#!/usr/bin/env python3
"""
Visualize text embeddings for a target species lineage plus sibling labels at each level.

Example:
python scripts/visualize_species_ancestors_siblings_tsne.py \
  --model ViT-B-16 \
  --pretrained /path/to/checkpoint.pt \
  --metadata-path /path/to/metadata.csv \
  --species-binomial "Cardinalis cardinalis" \
  --logs ./plots_species_context
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.open_clip import create_model_and_transforms, get_tokenizer
from src.open_clip import lorentz as L


LEVELS = ["kingdom", "phylum", "cls", "order", "family", "genus", "species"]
COLORS = {
    "kingdom": "#000000",
    "phylum": "#11A579",
    "cls": "#3969AC",
    "order": "#F2B701",
    "family": "#E73F74",
    "genus": "#80BA5A",
    "species": "#7F3C8D",
}


@dataclass
class TaxonomyNode:
    idx: int
    level: str
    label: str
    prompt: str
    is_ancestor: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE for target species lineage + siblings for every taxonomy level."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", default=None)
    p.add_argument(
        "--compare-pretrained",
        nargs="+",
        default=None,
        help="Optional list of checkpoints to compare side-by-side (e.g., ckpt_a ckpt_b).",
    )
    p.add_argument(
        "--compare-names",
        nargs="+",
        default=None,
        help="Optional subplot names for --compare-pretrained.",
    )
    p.add_argument(
        "--compare-modes",
        nargs="+",
        choices=["euclidean", "hyperbolic"],
        default=None,
        help="Optional per-checkpoint mode list for --compare-pretrained.",
    )
    p.add_argument(
        "--compare-hyperbolic-similarities",
        nargs="+",
        choices=["inner", "angle", "dist"],
        default=None,
        help="Optional per-checkpoint hyperbolic similarity list for --compare-pretrained.",
    )
    p.add_argument("--metadata-path", required=True)
    p.add_argument("--species-binomial", required=True, help='Example: "Cardinalis cardinalis"')
    p.add_argument("--target-kingdom", default=None, help="Optional disambiguation filter.")
    p.add_argument("--lineage-index", type=int, default=0, help="Pick lineage if multiple matches.")
    p.add_argument("--use-hyperbolic", action="store_true")
    p.add_argument(
        "--hyperbolic-similarity",
        choices=["inner", "angle", "dist"],
        default="inner",
        help="Only used when --use-hyperbolic is set.",
    )
    p.add_argument(
        "--max-siblings-per-level",
        type=int,
        default=-1,
        help="<=0 means use all siblings at each level.",
    )
    p.add_argument("--text-batch-size", type=int, default=256)
    p.add_argument("--tsne-perplexity", type=float, default=10.0)
    p.add_argument("--tsne-n-iter", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--panel-title-fontsize", type=int, default=20)
    p.add_argument("--logs", default="./plots_species_context")
    p.add_argument("--no-log-file", action="store_true")
    return p.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in LEVELS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = normalize_text(df[c])
    for c in LEVELS:
        df = df[df[c] != ""]
    return df.reset_index(drop=True)


def split_binomial(binomial: str) -> tuple[str, str]:
    parts = [p.strip() for p in binomial.strip().split() if p.strip()]
    if len(parts) < 2:
        raise ValueError("--species-binomial must include genus and species epithet.")
    genus = parts[0]
    species = " ".join(parts[1:])
    return genus, species


def resolve_lineage(
    df: pd.DataFrame,
    species_binomial: str,
    target_kingdom: str | None,
    lineage_index: int,
) -> Dict[str, str]:
    genus, species = split_binomial(species_binomial)
    query_l = species_binomial.strip().lower()
    genus_l = genus.lower()
    species_l = species.lower()

    mask = (df["genus"].str.lower() == genus_l) & (
        (df["species"].str.lower() == species_l) | (df["species"].str.lower() == query_l)
    )
    if target_kingdom:
        mask = mask & (df["kingdom"].str.lower() == target_kingdom.lower())
    candidates = df.loc[mask, LEVELS].drop_duplicates().reset_index(drop=True)

    if candidates.empty:
        raise ValueError(f"No lineage found for species '{species_binomial}'.")
    if lineage_index < 0 or lineage_index >= len(candidates):
        raise ValueError(f"--lineage-index must be in [0, {len(candidates) - 1}] for this query.")

    if len(candidates) > 1:
        logging.warning("Found %d distinct lineages, using lineage-index=%d.", len(candidates), lineage_index)
    row = candidates.iloc[lineage_index]
    return {lvl: str(row[lvl]) for lvl in LEVELS}


def top_labels_with_anchor(series: pd.Series, k: int, anchor: str) -> List[str]:
    labels = series.dropna().astype(str)
    if k <= 0:
        out = sorted(labels.unique().tolist())
    else:
        counts = labels.value_counts()
        out = counts.head(k).index.tolist()
    if anchor not in out:
        out.append(anchor)
    return sorted(set(out))


def build_nodes(
    df: pd.DataFrame, lineage: Dict[str, str], max_siblings_per_level: int
) -> List[TaxonomyNode]:
    nodes: List[TaxonomyNode] = []

    for level_i, level in enumerate(LEVELS):
        if level == "kingdom":
            labels = [lineage[level]]
        else:
            parent_levels = LEVELS[:level_i]
            filt = pd.Series(True, index=df.index)
            for p in parent_levels:
                filt &= df[p] == lineage[p]
            labels = top_labels_with_anchor(df.loc[filt, level], max_siblings_per_level, lineage[level])

        for label in labels:
            prefix = [lineage[l] for l in LEVELS[:level_i]]
            prompt_parts = prefix + [label]
            prompt = "An image of " + ", ".join(prompt_parts)
            is_ancestor = label == lineage[level]
            nodes.append(
                TaxonomyNode(
                    idx=len(nodes),
                    level=level,
                    label=label,
                    prompt=prompt,
                    is_ancestor=is_ancestor,
                )
            )

    return nodes


@torch.no_grad()
def encode_texts(model: torch.nn.Module, tokenizer, texts: Sequence[str], device: str, batch_size: int) -> torch.Tensor:
    out: List[torch.Tensor] = []
    for i0 in range(0, len(texts), batch_size):
        batch = texts[i0 : i0 + batch_size]
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens, normalize=True)
        out.append(emb)
    return torch.cat(out, dim=0)


def compute_distance_matrix(
    x: torch.Tensor,
    use_hyperbolic: bool,
    curv: torch.Tensor | None,
    hyperbolic_similarity: str,
) -> np.ndarray:
    n = x.shape[0]
    distances = torch.empty((n, n), device=x.device, dtype=torch.float32)
    for i0 in range(0, n, 512):
        i1 = min(n, i0 + 512)
        if use_hyperbolic:
            if hyperbolic_similarity == "angle":
                sims = -L.pairwise_oxy_angle(x[i0:i1], x, curv)
            elif hyperbolic_similarity == "dist":
                sims = -L.pairwise_dist(x[i0:i1], x, curv)
            else:
                sims = L.pairwise_inner(x[i0:i1], x, curv)
        else:
            sims = x[i0:i1] @ x.T
        distances[i0:i1] = 1.0 - sims
    distances = 0.5 * (distances + distances.T)
    distances.fill_diagonal_(0)
    return distances.detach().cpu().numpy()


def run_tsne(dist: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "metric": "precomputed",
        "init": "random",
        "random_state": seed,
        "verbose": 1,
    }
    tsne_sig = inspect.signature(TSNE.__init__).parameters
    if "n_iter" in tsne_sig:
        kwargs["n_iter"] = n_iter
    else:
        kwargs["max_iter"] = n_iter
    return TSNE(**kwargs).fit_transform(dist)


def _sanitize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "species"


def plot_result(
    xy_list: List[np.ndarray],
    nodes: List[TaxonomyNode],
    out_png: str,
    panel_names: List[str] | None = None,
    panel_title_fontsize: int = 20,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend_fontsize = 25
    n_panels = len(xy_list)
    if n_panels <= 0:
        raise ValueError("xy_list must contain at least one t-SNE result.")
    if panel_names is None:
        panel_names = ["" for _ in range(n_panels)]
    if len(panel_names) != n_panels:
        raise ValueError("panel_names must match xy_list length.")

    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(14, 11))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(12 * n_panels, 11))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = list(axes)

    for panel_i, (ax, xy) in enumerate(zip(axes, xy_list)):
        for level in LEVELS:
            idx = [n.idx for n in nodes if n.level == level]
            if idx:
                marker = "*" if level == "kingdom" else "o"
                size = 500 if level == "kingdom" else 200
                ax.scatter(
                    xy[idx, 0],
                    xy[idx, 1],
                    s=size,
                    marker=marker,
                    c=COLORS[level],
                    alpha=0.85,
                    edgecolors="none",
                    zorder=2,
                )
        ax.tick_params(
            axis="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        if panel_names[panel_i]:
            ax.set_title(panel_names[panel_i], fontsize=panel_title_fontsize)

    legend_handles = []
    for level in LEVELS:
        marker = "*" if level == "kingdom" else "o"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=COLORS[level],
                markeredgecolor="none",
                markersize=28 if level == "kingdom" else 20,
                label=f"{'class' if level == 'cls' else level}",
            )
        )

    if n_panels == 1:
        axes[0].legend(handles=legend_handles, loc="best", fontsize=legend_fontsize, frameon=True)
        fig.tight_layout()
    else:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.7, 0.5),
            fontsize=legend_fontsize,
            frameon=True,
        )
        fig.subplots_adjust(right=0.68, wspace=0.06)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.logs, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if not args.no_log_file:
        ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        handlers.append(logging.FileHandler(os.path.join(args.logs, f"{ts}-species_context_tsne.log")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision == "fp16" and device == "cpu":
        raise ValueError("fp16 is not supported on CPU for this script.")

    df = load_metadata(args.metadata_path)
    lineage = resolve_lineage(df, args.species_binomial, args.target_kingdom, args.lineage_index)
    logging.info("Resolved lineage: %s", lineage)

    nodes = build_nodes(df, lineage, args.max_siblings_per_level)
    logging.info("Built %d total text nodes.", len(nodes))
    for lvl in LEVELS:
        count_lvl = sum(1 for n in nodes if n.level == lvl)
        logging.info("Level %s: %d labels", lvl, count_lvl)

    prompts = [n.prompt for n in nodes]
    if args.compare_pretrained:
        if len(args.compare_pretrained) < 2:
            raise ValueError("--compare-pretrained requires at least 2 checkpoints.")
        checkpoints = args.compare_pretrained
    else:
        if not args.pretrained:
            raise ValueError("Either --pretrained or --compare-pretrained must be provided.")
        checkpoints = [args.pretrained]

    if args.compare_names:
        if len(args.compare_names) != len(checkpoints):
            raise ValueError("--compare-names must match number of checkpoints.")
        panel_names = args.compare_names
    else:
        panel_names = [os.path.splitext(os.path.basename(p))[0] for p in checkpoints]

    if args.compare_pretrained:
        if args.compare_modes:
            if len(args.compare_modes) != len(checkpoints):
                raise ValueError("--compare-modes must match number of checkpoints.")
            panel_use_hyperbolic = [m == "hyperbolic" for m in args.compare_modes]
        else:
            panel_use_hyperbolic = [args.use_hyperbolic] * len(checkpoints)

        if args.compare_hyperbolic_similarities:
            if len(args.compare_hyperbolic_similarities) != len(checkpoints):
                raise ValueError("--compare-hyperbolic-similarities must match number of checkpoints.")
            panel_hyperbolic_similarity = args.compare_hyperbolic_similarities
        else:
            panel_hyperbolic_similarity = [args.hyperbolic_similarity] * len(checkpoints)
    else:
        panel_use_hyperbolic = [args.use_hyperbolic]
        panel_hyperbolic_similarity = [args.hyperbolic_similarity]

    tokenizer = get_tokenizer(args.model)
    xy_list: List[np.ndarray] = []
    for ckpt, ckpt_use_hyperbolic, ckpt_hyperbolic_similarity in zip(
        checkpoints, panel_use_hyperbolic, panel_hyperbolic_similarity
    ):
        logging.info(
            "Encoding texts with checkpoint: %s | mode=%s | similarity=%s",
            ckpt,
            "hyperbolic" if ckpt_use_hyperbolic else "euclidean",
            ckpt_hyperbolic_similarity,
        )
        model_kwargs = {
            "pretrained": ckpt,
            "force_custom_text": False,
            "weights_only": False,
        }
        if ckpt_use_hyperbolic:
            model_kwargs["use_hyperbolic"] = True
            model_kwargs["hyperbolic_load_nonstrict"] = True
        model, _, _ = create_model_and_transforms(args.model, **model_kwargs)
        model = model.to(device).eval()

        emb = encode_texts(model, tokenizer, prompts, device=device, batch_size=args.text_batch_size)
        curv = model.curv.exp().detach() if ckpt_use_hyperbolic else None
        dist = compute_distance_matrix(
            emb,
            use_hyperbolic=ckpt_use_hyperbolic,
            curv=curv,
            hyperbolic_similarity=ckpt_hyperbolic_similarity,
        )
        xy = run_tsne(dist, args.tsne_perplexity, args.tsne_n_iter, args.seed)
        xy_list.append(xy)
        del model, emb
        if device == "cuda":
            torch.cuda.empty_cache()

    base = _sanitize_name(args.species_binomial)
    out_png = os.path.join(args.logs, f"{base}_ancestors_siblings_tsne.png")
    plot_result(
        xy_list,
        nodes,
        out_png,
        panel_names=panel_names,
        panel_title_fontsize=args.panel_title_fontsize,
    )
    logging.info("Saved plot: %s", out_png)

    for panel_name, xy in zip(panel_names, xy_list):
        panel_slug = _sanitize_name(panel_name)
        out_csv = os.path.join(args.logs, f"{base}_ancestors_siblings_nodes_{panel_slug}.csv")
        pd.DataFrame(
            {
                "idx": [n.idx for n in nodes],
                "level": [n.level for n in nodes],
                "label": [n.label for n in nodes],
                "is_ancestor": [n.is_ancestor for n in nodes],
                "prompt": [n.prompt for n in nodes],
                "tsne_x": xy[:, 0],
                "tsne_y": xy[:, 1],
            }
        ).to_csv(out_csv, index=False)
        logging.info("Saved node table: %s", out_csv)


if __name__ == "__main__":
    main()
