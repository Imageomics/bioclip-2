# BioCLIP 2 for Hierarchical Fine-Grained Evaluation

This repository contains a BioCLIP 2 based codebase for hierarchical fine-grained classification experiments. It includes Euclidean and hyperbolic training, taxonomy-only contrastive loss, top-down and coarse-to-fine evaluation, nLCA reporting, taxonomy embedding visualization, and dataset preparation utilities for CrypticBio-style evaluation.

The current setup is intended to support the main experiments described in the hierarchical contrastive learning workflow:
- level-restricted contrastive training
- taxonomy-only hierarchical contrastive loss
- Euclidean and hyperbolic variants
- coarse-to-fine evaluation
- top-down constrained evaluation
- normalized Lowest Common Ancestor (`nLCA`) reporting
- text embedding visualization across taxonomy levels

## Table of Contents

1. [Overview](#overview)
2. [Environment](#environment)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Visualization](#visualization)
6. [CrypticBio Preparation](#crypticbio-preparation)
7. [Repository Layout](#repository-layout)
8. [Notes](#notes)
9. [License](#license)

## Overview

The main additions in this repository are:
- taxonomy-only training via `--taxonomy-loss-only`
- level-restricted comparisons via `--taxonomy-compare-same-level`
- all-level training via `--taxonomy-use-all-level-data`
- configurable taxonomy image-side weighting via `--taxonomy-image-weighting`
- hyperbolic CLIP support via `--use-hyperbolic`
- hierarchical evaluation scripts for both unrestricted coarse-to-fine and top-down constrained prediction
- `nLCA` reporting in the top-down evaluation script

Key implementation files:
- training arguments: [src/training/params.py](src/training/params.py)
- taxonomy loss: [src/open_clip/loss.py](src/open_clip/loss.py)
- hyperbolic model: [src/open_clip/model.py](src/open_clip/model.py), [src/open_clip/lorentz.py](src/open_clip/lorentz.py)
- training entrypoint: [src/training/main.py](src/training/main.py)
- top-down evaluation: [src/evaluation/zero_shot_taxonomy_levels_topdown.py](src/evaluation/zero_shot_taxonomy_levels_topdown.py)
- coarse-to-fine evaluation: [src/evaluation/zero_shot_taxonomy_levels_coarse_to_fine.py](src/evaluation/zero_shot_taxonomy_levels_coarse_to_fine.py)

## Environment

Create the two main conda environments from the repository root:

```bash
conda env create -f bioclip-train.yml
conda env create -f bioclip-test.yml
```

Environment roles:
- `bioclip-train`: training
- `bioclip-test`: evaluation, dataset preparation, and visualization

This workflow uses `bioclip-train.yml` and `bioclip-test.yml` as the maintained environment definitions. The older `requirements.txt` and `requirements.yml` files are not part of the current branch setup.

## Training

A single configurable training launcher is included:
- [slurm/train.sh](slurm/train.sh)

The script is a template. Before running it, update:
- repository path
- training and validation shard paths
- checkpoint path
- output log directory
- `USE_HYPERBOLIC`

### Core taxonomy-only recipe

The main taxonomy-only configuration is:

```bash
--taxonomy-loss-only \
--taxonomy-compare-same-level 1 \
--taxonomy-use-all-level-data 1 \
--taxonomy-group-same-text 1 \
--taxonomy-image-weighting standard
```

This means:
- use all taxonomy levels during training
- compare each level only against text from the same level
- merge duplicate text labels within a batch
- use the standard taxonomy image-side weighting from the paper setup

### Example launcher settings

Euclidean:

```bash
USE_HYPERBOLIC=0
```

Hyperbolic:

```bash
USE_HYPERBOLIC=1
HYPERBOLIC_SIMILARITY="dist"
```

Run:

```bash
sbatch slurm/train.sh
```

## Evaluation

Two taxonomy evaluation launchers are included:
- [slurm/eval_which_level.sh](slurm/eval_which_level.sh)
- [slurm/eval_topdown.sh](slurm/eval_topdown.sh)

Both scripts support:
- Euclidean or hyperbolic evaluation
- configurable checkpoint path
- dataset root and metadata file
- optional `target_kingdom`
- optional CSV export

### Coarse-to-fine evaluation

This runs independent taxonomy-level retrieval and reports:
- per-level accuracy
- first misclassification depth counts

Launcher:

```bash
sbatch slurm/eval_which_level.sh
```

Backend:

```bash
python -m src.evaluation.zero_shot_taxonomy_levels_coarse_to_fine
```

### Top-down evaluation

This runs constrained prediction from `kingdom -> ... -> species`, where the candidate set at each level is restricted to the descendants of the predicted parent.

It reports:
- per-level accuracy
- first misclassification depth counts
- `nLCA`

Launcher:

```bash
sbatch slurm/eval_topdown.sh
```

Backend:

```bash
python -m src.evaluation.zero_shot_taxonomy_levels_topdown
```

### nLCA

`nLCA` in the top-down script is computed as:
- the depth of the longest correct prefix shared by predicted and ground-truth taxonomy paths
- divided by the total number of evaluated levels

Interpretation:
- `1.0` means the whole taxonomy path is correct
- `0.0` means the prediction is already wrong at the first level

## Visualization

The repository includes a taxonomy text embedding visualization script:
- [scripts/visualize_species_ancestors_siblings_tsne.py](scripts/visualize_species_ancestors_siblings_tsne.py)

SLURM launcher:
- [slurm/visualize_species_ancestors_siblings_tsne_euclidean.sh](slurm/visualize_species_ancestors_siblings_tsne_euclidean.sh)

This tool can:
- plot a target species together with its ancestors and sibling labels
- compare multiple checkpoints side by side
- compare Euclidean and hyperbolic checkpoints in the same figure

Run:

```bash
sbatch slurm/visualize_species_ancestors_siblings_tsne_euclidean.sh
```

Before launching, update:
- `PRETRAINED`
- `METADATA_PATH`
- `SPECIES_BINOMIAL`
- optional `USE_HYPERBOLIC`
- optional `COMPARE_*` arrays

## CrypticBio Preparation

To build a `CrypticBio_eval` folder compatible with the evaluation scripts, use:
- [scripts/prepare_crypticbio_eval.py](scripts/prepare_crypticbio_eval.py)
- [slurm/prepare_crypticbio_eval.sh](slurm/prepare_crypticbio_eval.sh)

This utility downloads `gmanolache/CrypticBio` from Hugging Face and exports:

```text
<out_dir>/
  metadata.csv
  images/<split>/*.jpg
```

Run:

```bash
sbatch slurm/prepare_crypticbio_eval.sh
```

Or directly:

```bash
python scripts/prepare_crypticbio_eval.py \
  --out-dir /path/to/CrypticBio_eval
```

If required, install missing dependencies:

```bash
pip install datasets pillow pyarrow
```

## Repository Layout

Important paths in this repository:

- `src/training/`
- `src/open_clip/`
- `src/evaluation/`
- `slurm/train.sh`
- `slurm/eval_which_level.sh`
- `slurm/eval_topdown.sh`
- `slurm/visualize_species_ancestors_siblings_tsne_euclidean.sh`
- `slurm/prepare_crypticbio_eval.sh`
- `scripts/visualize_species_ancestors_siblings_tsne.py`
- `scripts/prepare_crypticbio_eval.py`

## Notes

- Several provided `slurm/*.sh` files are templates with placeholder paths. Update them before use.
- Some training scripts still contain hardcoded local paths from the development environment. Treat them as examples, not drop-in universal launchers.
- Hyperbolic evaluation supports `inner`, `angle`, and `dist` similarity options.
- Euclidean evaluation uses the default normalized dot product.

## License

BioCLIP 2 is released under the MIT License. See [LICENSE](LICENSE).
