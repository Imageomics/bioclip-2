"""
Zero-shot taxonomy level evaluation with top-down constrained prediction.

Predicts kingdom first, then restricts candidates for each lower level to
children of the predicted parent. Stops evaluation for an image after the
first misclassification and reports first-misclassified counts.
"""

import argparse
import warnings
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
)
from ..open_clip import lorentz as L
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from ..training.logger import setup_logging
from .data import DatasetFromFile
from .params import parse_args
from .utils import init_device, random_seed


LEVEL_ORDER = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]


def _parse_level_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--target-kingdom", type=str, nargs="+", default=None)
    parser.add_argument(
        "--use-hyperbolic",
        action="store_true",
        help="Use hyperbolic embeddings and similarity.",
    )
    parser.add_argument(
        "--hyperbolic-similarity",
        type=str,
        choices=["inner", "angle", "dist"],
        default="inner",
        help="Hyperbolic similarity: inner (default), angle (oxy-angle), or dist (negative distance).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=LEVEL_ORDER,
        help="Levels to evaluate from coarse to fine.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write per-level summary CSV.",
    )
    parser.add_argument(
        "--max-images-per-species",
        type=int,
        default=-1,
        help="Max images per species; -1 for all rows in CSV.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=512,
        help="Number of text embeddings per chunk when scoring.",
    )
    return parser.parse_known_args(argv)[0]


def img_loader(filepath: str):
    try:
        return Image.open(filepath).convert("RGB")
    except Exception as e:
        logging.warning(f"Error loading image '{filepath}': {e}.")
        return None


def get_dataloader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base_dataset[real_idx]
        return image, real_idx


def _parent_label(label: str) -> Optional[str]:
    if not label:
        return None
    if ", " not in label:
        return None
    return label.rsplit(", ", 1)[0]


def _compute_nlca(first_error: List[Optional[str]], levels: List[str]) -> tuple[List[float], float]:
    total_depth = len(levels)
    if total_depth <= 0:
        return [], 0.0

    level_to_index = {level: idx for idx, level in enumerate(levels)}
    scores: List[float] = []
    for err in first_error:
        matched_depth = total_depth if err is None else level_to_index.get(err, 0)
        scores.append(matched_depth / total_depth)

    return scores, float(np.mean(scores)) if scores else 0.0


def main():
    warnings.filterwarnings(
        "ignore",
        message="Converting mask without torch.bool dtype to bool; this will negatively affect performance.*",
        category=UserWarning,
    )
    args = parse_args(sys.argv[1:])
    level_args = _parse_level_args(sys.argv[1:])
    args.metadata_path = level_args.metadata_path
    args.target_kingdom = level_args.target_kingdom
    args.use_hyperbolic = level_args.use_hyperbolic
    args.hyperbolic_similarity = level_args.hyperbolic_similarity
    args.levels = level_args.levels
    args.output_csv = level_args.output_csv
    args.max_images_per_species = level_args.max_images_per_species
    args.text_batch_size = level_args.text_batch_size

    if args.data_root is None:
        logging.error("Error: --data-root is required.")
        sys.exit(1)

    target_kingdom = args.target_kingdom[0].lower() if args.target_kingdom else None

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_device(args)
    args.save_logs = args.logs and args.logs.lower() != "none"

    if args.save_logs and args.name is None:
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot_taxonomy_levels_topdown",
            ]
        )
    if args.save_logs is None:
        args.log_path = None
    else:
        log_base_path = os.path.join(args.logs, args.name)
        args.log_path = None
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    mode_label = "hyperbolic" if args.use_hyperbolic else "euclidean"
    logging.info(
        f"Zero-shot taxonomy levels ({mode_label}) top-down "
        f"for kingdom: {target_kingdom or 'ALL'}"
    )
    logging.info(f"Params: {vars(args)}")
    logging.info(f"Max images per species: {args.max_images_per_species}")

    random_seed(args.seed, 0)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        weights_only=False,
        use_hyperbolic=args.use_hyperbolic,
        hyperbolic_load_nonstrict=args.use_hyperbolic,
    )
    model.eval()
    tokenizer = get_tokenizer(args.model)
    cast_dtype = get_cast_dtype(args.precision)

    label_filename = args.label_filename
    if label_filename is None and args.metadata_path is not None:
        if os.path.isabs(args.metadata_path) and args.metadata_path.startswith(args.data_root):
            label_filename = os.path.relpath(args.metadata_path, args.data_root)
        else:
            label_filename = args.metadata_path
    if label_filename is None:
        logging.error("Error: --label_filename or --metadata-path is required.")
        sys.exit(1)

    logging.info(f"Loading metadata: {label_filename}")
    dataset_text_type = args.text_type if args.text_type != "asis" else "taxon"
    base_dataset = DatasetFromFile(
        args.data_root,
        label_filename,
        transform=preprocess_val,
        classes=dataset_text_type,
    )
    df_full = base_dataset.data

    df = df_full
    if target_kingdom is not None:
        df = df[df["kingdom"].astype(str).str.lower() == target_kingdom]
        if df.empty:
            logging.error(f"No data for kingdom '{target_kingdom}'.")
            sys.exit(1)
    # Keep original row indices so selected_indices still map to df_full/base_dataset.
    df = df.copy()

    required_cols = ["kingdom", "phylum", "cls", "order", "family", "genus", "species", "filepath"]
    for c in required_cols:
        if c not in df.columns:
            logging.error(f"Missing required column: {c}")
            sys.exit(1)
        df[c] = df[c].fillna("").astype(str)
    df["full_path"] = df.apply(lambda r: os.path.join(args.data_root, r["filepath"]), axis=1)
    df["file_exists"] = df["full_path"].apply(os.path.exists)
    df = df[
        (df["kingdom"] != "")
        & (df["phylum"] != "")
        & (df["cls"] != "")
        & (df["order"] != "")
        & (df["family"] != "")
        & (df["genus"] != "")
        & (df["species"] != "")
        & (df["filepath"] != "")
        & (df["file_exists"])
    ]
    if df.empty:
        logging.error("No valid rows after cleaning.")
        sys.exit(1)

    level_texts: Dict[str, Dict[str, List]] = {
        level: {"labels": [], "label_to_idx": {}} for level in LEVEL_ORDER
    }
    records: pd.DataFrame
    index_to_pos: Dict[int, int]

    def add_label(level: str, label: str):
        if label is None or label == "":
            return
        label_to_idx = level_texts[level]["label_to_idx"]
        if level != "species":
            if label in label_to_idx:
                return
            label_to_idx[label] = len(level_texts[level]["labels"])
        level_texts[level]["labels"].append(label)

    selected_indices = []
    species_counts: Dict[str, int] = {}
    for original_idx, rr in df.iterrows():
        species_name = rr["species"]
        if args.max_images_per_species > 0:
            count = species_counts.get(species_name, 0)
            if count >= args.max_images_per_species:
                continue
            species_counts[species_name] = count + 1
        selected_indices.append(int(original_idx))

    records = df_full.loc[selected_indices, required_cols].copy()
    records["index"] = selected_indices
    records = records.reset_index(drop=True)
    index_to_pos = {idx: pos for pos, idx in enumerate(selected_indices)}

    def build_hier_label(row: pd.Series, level: str) -> str:
        level_order = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]
        if level not in level_order:
            return ""
        parts = []
        for name in level_order:
            value = row.get(name, "")
            if value:
                parts.append(value)
            if name == level:
                break
        return ", ".join(parts)

    for level in args.levels:
        records[f"{level}_label"] = records.apply(lambda r, lv=level: build_hier_label(r, lv), axis=1)

    for _, rr in records.iterrows():
        add_label("kingdom", rr["kingdom_label"])
        add_label("phylum", rr["phylum_label"])
        add_label("cls", rr["cls_label"])
        add_label("order", rr["order_label"])
        add_label("family", rr["family_label"])
        add_label("genus", rr["genus_label"])
        add_label("species", rr["species_label"])

    indexed_dataset = IndexedDataset(base_dataset, selected_indices)
    dataloader = get_dataloader(indexed_dataset, args.batch_size, args.workers)

    total_rows = len(selected_indices)
    if total_rows == 0:
        logging.error("No images collected for evaluation.")
        sys.exit(1)

    logging.info("Encoding images once for all levels.")
    image_embs = None
    processed = 0
    for images, idxs in dataloader:
        processed += images.size(0)
        if processed == images.size(0) or processed % 1000 == 0:
            logging.info(f"Encoding images: {processed}/{total_rows}")
        images = images.to(device)
        if cast_dtype is not None:
            images = images.to(dtype=cast_dtype)
        with torch.no_grad():
            out = model.encode_image(images, normalize=True)
        emb_img = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()
        if image_embs is None:
            image_embs = np.zeros((total_rows, emb_img.shape[1]), dtype=emb_img.dtype)
        for b, idx in enumerate(idxs):
            pos = index_to_pos[int(idx)]
            image_embs[pos] = emb_img[b]

    if image_embs is None:
        logging.error("No images collected for evaluation.")
        sys.exit(1)

    image_embs_t = torch.from_numpy(image_embs)
    if args.use_hyperbolic:
        curv = model.curv.exp()

    def build_text_embeddings(labels: List[str]) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for label in labels:
                texts = [template(label) for template in openai_imagenet_template]
                tokens = tokenizer(texts).to(device)
                class_feats = model.encode_text(tokens)
                if args.use_hyperbolic:
                    tangent = L.log_map0(class_feats, curv)
                    class_feat = tangent.mean(dim=0)
                    class_feat = L.exp_map0(class_feat, curv)
                else:
                    class_feat = F.normalize(class_feats, dim=-1).mean(dim=0)
                    class_feat = class_feat / class_feat.norm()
                embeddings.append(class_feat)
                del tokens, class_feats
        return torch.stack(embeddings, dim=0).cpu()

    def chunked_predict(emb_img: torch.Tensor, text_embs: torch.Tensor) -> List[int]:
        best_scores = None
        best_indices = None
        num_labels = text_embs.shape[0]
        for start in range(0, num_labels, args.text_batch_size):
            end = min(num_labels, start + args.text_batch_size)
            chunk_embs = text_embs[start:end].to(device)
            if args.use_hyperbolic:
                if args.hyperbolic_similarity == "angle":
                    sims = -L.pairwise_oxy_angle(emb_img, chunk_embs, curv)
                elif args.hyperbolic_similarity == "dist":
                    sims = -L.pairwise_dist(emb_img, chunk_embs, curv)
                else:
                    sims = L.pairwise_inner(emb_img, chunk_embs, curv)
            else:
                sims = emb_img @ chunk_embs.T
            chunk_scores, chunk_idx = sims.max(dim=1)
            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_idx + start
            else:
                better = chunk_scores > best_scores
                best_scores = torch.where(better, chunk_scores, best_scores)
                best_indices = torch.where(better, chunk_idx + start, best_indices)
            del chunk_embs, sims, chunk_scores, chunk_idx
        return best_indices.cpu().tolist()

    level_results: Dict[str, Dict[str, int]] = {}
    first_error = [None] * total_rows
    prev_pred = [None] * total_rows
    eval_batch = args.batch_size

    for level_idx, level in enumerate(args.levels):
        labels = level_texts[level]["labels"]
        if not labels:
            logging.warning(f"No text nodes for level {level}, skipping.")
            continue
        logging.info(f"Encoding text for level {level} ({len(labels)} labels).")
        text_embs = build_text_embeddings(labels)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        parent_to_label_indices: Dict[Optional[str], List[int]] = {}
        for lab, idx in label_to_idx.items():
            parent = _parent_label(lab)
            parent_to_label_indices.setdefault(parent, []).append(idx)

        correct = 0
        processed = 0
        predicted = [None] * total_rows
        active_indices = [i for i, err in enumerate(first_error) if err is None]
        if not active_indices:
            logging.info(f"No active images left at level {level}.")
            level_results[level] = {"correct": correct, "total": total_rows}
            continue

        if level_idx == 0:
            grouped = {None: active_indices}
        else:
            grouped: Dict[Optional[str], List[int]] = {}
            for i in active_indices:
                parent_label = prev_pred[i]
                grouped.setdefault(parent_label, []).append(i)

        for parent_label, idxs in grouped.items():
            candidate_indices = parent_to_label_indices.get(parent_label, [])
            if not candidate_indices:
                for pos in idxs:
                    if first_error[pos] is None:
                        first_error[pos] = level
                continue

            for i in range(0, len(idxs), eval_batch):
                j = min(len(idxs), i + eval_batch)
                processed += j - i
                if processed == (j - i) or processed >= len(active_indices):
                    logging.info(f"Level {level}: {processed}/{len(active_indices)}")
                batch_positions = idxs[i:j]
                emb_img = image_embs_t[batch_positions].to(device)
                cand_embs = text_embs[candidate_indices]
                pred_offsets = chunked_predict(emb_img, cand_embs)
                for offset, pred_off in enumerate(pred_offsets):
                    pos = batch_positions[offset]
                    pred_idx = candidate_indices[pred_off]
                    pred_label = labels[pred_idx]
                    predicted[pos] = pred_label
                    true_label = records.iloc[pos][f"{level}_label"]
                    if pred_label == true_label:
                        correct += 1
                    elif first_error[pos] is None:
                        first_error[pos] = level

        evaluated = len(active_indices)
        level_results[level] = {"correct": correct, "total": evaluated, "all": total_rows}
        level_acc = correct / max(1, evaluated)
        logging.info(
            f"Level {level} accuracy: {correct}/{evaluated} ({level_acc:.4f})"
        )
        prev_pred = predicted

    first_error_counts: Dict[str, int] = {level: 0 for level in args.levels}
    first_error_counts["none"] = 0
    for err in first_error:
        if err is None:
            first_error_counts["none"] += 1
        else:
            first_error_counts[err] = first_error_counts.get(err, 0) + 1
    _, mean_nlca = _compute_nlca(first_error, args.levels)

    logging.info("Level-wise accuracy:")
    for level in args.levels:
        if level in level_results:
            res = level_results[level]
            acc = res["correct"] / max(1, res["total"])
            logging.info(f"  {level}: {res['correct']}/{res['total']} ({acc:.4f})")

    logging.info("First misclassification level counts:")
    for level, count in first_error_counts.items():
        logging.info(f"  {level}: {count}")
    logging.info(f"nLCA: {mean_nlca:.4f}")

    rows = []
    for level in args.levels:
        res = level_results.get(level, {"correct": 0, "total": 0})
        total = res["total"]
        correct = res["correct"]
        acc = correct / max(1, total)
        rows.append(
            {
                "level": level,
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "all_samples": res.get("all", total_rows),
                "first_misclassified_count": first_error_counts.get(level, 0),
            }
        )
    rows.append(
        {
            "level": "none",
            "correct": "",
            "total": "",
            "accuracy": "",
            "first_misclassified_count": first_error_counts.get("none", 0),
            "nlca_score": "",
        }
    )
    rows.append(
        {
            "level": "nLCA",
            "correct": "",
            "total": "",
            "accuracy": "",
            "all_samples": total_rows,
            "first_misclassified_count": "",
            "nlca_score": mean_nlca,
        }
    )
    summary_df = pd.DataFrame(rows)
    if args.output_csv is None and args.save_logs:
        args.output_csv = os.path.join(os.path.dirname(args.log_path), "taxonomy_level_summary.csv")
    if args.output_csv:
        summary_df.to_csv(args.output_csv, index=False)
        logging.info(f"Saved summary CSV to {args.output_csv}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
