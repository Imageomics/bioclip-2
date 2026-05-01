#!/usr/bin/env python3
"""
Download and convert Hugging Face dataset `gmanolache/CrypticBio` to the
BioCLIP evaluation folder format:

<out_dir>/
  metadata.csv
  images/<split>/<index>.jpg

The generated metadata includes taxonomy columns expected by evaluation scripts:
kingdom, phylum, cls, order, family, genus, species, common_name, filepath, class
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image as PILImage

try:
    from datasets import Dataset, DatasetDict, Image, load_dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: 'datasets'. Install with:\n"
        "  pip install datasets pillow pyarrow\n"
        "or inside conda:\n"
        "  conda install -c conda-forge datasets pillow pyarrow"
    ) from e


TAXONOMY_COLUMN_CANDIDATES = {
    "kingdom": ["kingdom"],
    "phylum": ["phylum", "division"],
    "cls": ["cls", "class", "class_name", "tax_class"],
    "order": ["order"],
    "family": ["family"],
    "genus": ["genus"],
    "species": [
        "species",
        "species_name",
        "scientific_name",
        "taxon",
        "taxon_name",
        "label_name",
    ],
    "common_name": ["common_name", "vernacular_name", "vernacularName", "name"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CrypticBio for BioCLIP evaluation.")
    parser.add_argument("--dataset-id", default="gmanolache/CrypticBio")
    parser.add_argument("--out-dir", required=True, help="Output directory for metadata.csv and images/")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache dir")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to export (default: all splits in dataset)",
    )
    parser.add_argument("--image-column", default=None, help="Override image column name")
    parser.add_argument("--label-column", default=None, help="Override label/class column name")
    parser.add_argument("--max-per-split", type=int, default=-1, help="Limit rows per split; -1 for all")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--download-timeout", type=float, default=30.0)
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode for large datasets (default: enabled).",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for x in value:
            text = normalize_text(x)
            if text:
                return text
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "sample"


def choose_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def find_image_column(ds: Dataset, override: Optional[str]) -> str:
    if override:
        if override not in ds.column_names:
            raise ValueError(f"--image-column '{override}' not in dataset columns: {ds.column_names}")
        return override

    for col, feat in ds.features.items():
        if isinstance(feat, Image):
            return col
    for cand in ("image", "img", "url", "image_url", "image_path", "filepath", "path"):
        if cand in ds.column_names:
            return cand
    raise ValueError("Could not find image column. Use --image-column to set it explicitly.")


def to_pil_image(value: Any, timeout: float) -> Optional[PILImage.Image]:
    if value is None:
        return None
    if hasattr(value, "convert") and hasattr(value, "save"):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            try:
                return PILImage.open(io.BytesIO(value["bytes"])).convert("RGB")
            except Exception:
                return None
        path = value.get("path")
        if path:
            try:
                return PILImage.open(path).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(value, str):
        source = value.strip()
        if not source:
            return None
        try:
            if source.startswith("http://") or source.startswith("https://"):
                with urllib.request.urlopen(source, timeout=timeout) as resp:
                    data = resp.read()
                return PILImage.open(io.BytesIO(data)).convert("RGB")
            return PILImage.open(source).convert("RGB")
        except Exception:
            return None
    return None


def extract_label_value(
    row: Dict[str, Any],
    ds: Dataset,
    label_column: Optional[str],
) -> str:
    if label_column and label_column in row:
        value = row[label_column]
        feat = ds.features.get(label_column)
        if hasattr(feat, "int2str") and isinstance(value, int):
            return normalize_text(feat.int2str(value))
        return normalize_text(value)

    for cand in ("label", "class", "species", "scientific_name", "taxon_name", "name"):
        if cand in row:
            value = row[cand]
            feat = ds.features.get(cand)
            if hasattr(feat, "int2str") and isinstance(value, int):
                return normalize_text(feat.int2str(value))
            return normalize_text(value)
    return ""


def fill_taxonomy_from_row(row: Dict[str, Any], ds: Dataset) -> Dict[str, str]:
    out = {k: "" for k in TAXONOMY_COLUMN_CANDIDATES}
    for key, candidates in TAXONOMY_COLUMN_CANDIDATES.items():
        col = choose_column(ds.column_names, candidates)
        if col is None:
            continue
        value = row.get(col)
        feat = ds.features.get(col)
        if hasattr(feat, "int2str") and isinstance(value, int):
            out[key] = normalize_text(feat.int2str(value))
        else:
            out[key] = normalize_text(value)
    return out


def ensure_species_and_class(tax: Dict[str, str], label_value: str) -> Dict[str, str]:
    if not tax["species"]:
        tax["species"] = label_value
    class_label = tax["species"] or tax["common_name"] or label_value or "unknown"
    tax["class"] = class_label

    if not tax["genus"] and tax["species"] and " " in tax["species"]:
        tax["genus"] = tax["species"].split(" ")[0].strip()
    return tax


def load_dataset_dict(args: argparse.Namespace) -> DatasetDict:
    ds = load_dataset(
        args.dataset_id,
        cache_dir=args.cache_dir,
        token=args.token,
        streaming=args.streaming,
    )
    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})
    return ds


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    images_root = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    ds_dict = load_dataset_dict(args)
    split_names = args.splits if args.splits else list(ds_dict.keys())
    for split_name in split_names:
        if split_name not in ds_dict:
            raise ValueError(f"Split '{split_name}' not found. Available splits: {list(ds_dict.keys())}")

    rows = []
    row_id = 0
    for split_name in split_names:
        ds = ds_dict[split_name]
        image_col = find_image_column(ds, args.image_column)

        split_dir = images_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if args.max_per_split < 0:
            print(
                f"[warn] --max-per-split is -1 for split '{split_name}'. "
                "This may run for a very long time on CrypticBio."
            )

        for i, item in enumerate(ds):
            if args.max_per_split >= 0 and i >= args.max_per_split:
                break

            img = to_pil_image(item.get(image_col), timeout=args.download_timeout)
            if img is None:
                continue

            label_value = extract_label_value(item, ds, args.label_column)
            tax = fill_taxonomy_from_row(item, ds)
            tax = ensure_species_and_class(tax, label_value)

            stem = sanitize_filename(f"{i:08d}_{tax['class']}")
            rel_path = Path("images") / split_name / f"{stem}.jpg"
            abs_path = out_dir / rel_path
            img.save(abs_path, format="JPEG", quality=args.jpg_quality)

            rows.append(
                {
                    "id": row_id,
                    "kingdom": tax["kingdom"],
                    "phylum": tax["phylum"],
                    "cls": tax["cls"],
                    "order": tax["order"],
                    "family": tax["family"],
                    "genus": tax["genus"],
                    "species": tax["species"],
                    "common_name": tax["common_name"],
                    "filepath": str(rel_path).replace("\\", "/"),
                    "class": tax["class"],
                    "split": split_name,
                }
            )
            row_id += 1

    if not rows:
        raise RuntimeError("No rows were exported. Check dataset access and image column settings.")

    metadata_path = out_dir / "metadata.csv"
    fieldnames = [
        "id",
        "kingdom",
        "phylum",
        "cls",
        "order",
        "family",
        "genus",
        "species",
        "common_name",
        "filepath",
        "class",
        "split",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved images:   {images_root}")


if __name__ == "__main__":
    main()
