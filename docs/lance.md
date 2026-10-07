# Training from Lance datasets

BioCLIP 2 can train from [Lance](https://lancedb.github.io/lance/) datasets instead of webdataset tar shards. A Lance
dataset holds the TreeOfLife tar shards as is: one column per tar member, same bytes. Only data loading changes.

## Why Lance

- **Same model, same speed** as training from tar shards ([Validation](#validation)).
- **Less to store and read:** about a third less disk space and about 40% less data read in a full training run,
  so less load on shared storage ([Data read during training](#data-read-during-training)).
- **Reproducible batches:** Lance shuffles are seeded, so the same settings give the same Lance batches, even after
  resuming. Webdataset seeds its shuffle from the clock.
- **Direct access:** read any row by number, or one column alone (all captions, no images).
- **Shuffle by flag:** webdataset-like, chunked or global, without rewriting the data.

## Install

```
pip install pylance
```

The package is `pylance`; `pip install lance` is an unrelated project. Only Lance datasets need it.

## Data

Each row is one tar sample, in tar-shard order:

| column | content |
|---|---|
| `jpg` | JPEG bytes |
| `sci`, `com`, `taxon`, `sci_com`, `taxon_com`, `taxonTag`, `taxonTag_com` | captions |
| `scientific_name`, `common_name`, `taxonomic_name` | other tar members |
| `uuid`, `source_shard`, `source_index` | sample key, source tar shard and position |

**Keep the manifest next to the dataset.** Copy each dataset's `.manifest.json` with it, e.g.
`train.lance.manifest.json` with `train.lance`. The loader reads the size, shard layout and conversion status from
it. Without it, the loader scans the dataset first (slower) and logs a warning.

## Run

Point `--train-data` and `--val-data` at `.lance` directories. In `slurm/train.sh`, change these lines:

```
  --train-data '[training-dir]/train.lance' \
  --val-data '[evaluation-dir]/val.lance' \
  --dataset-type 'auto' \
```

`auto` (the default) picks the loader per path: `.lance` uses Lance, `.tar` uses webdataset, so the LAION replay
data (`--continual-data`) stays on tar shards. With `--dataset-type 'lance'`, also pass
`--continual-dataset-type 'webdataset'`, or the LAION tars fail to load.

## Options

| flag | default | meaning |
|---|---|---|
| `--lance-sampler` | `wds` | how training batches are drawn: `wds`, `chunked` or `global` (see [How batches are drawn](#how-batches-are-drawn)) |
| `--lance-shuffle-buffer` | `5000` | shuffle buffer for `wds` (as in the webdataset loader) |
| `--lance-shuffle-chunk` | `32768` | `chunked` shuffles within chunks of this many consecutive rows; `0` = global |
| `--lance-allow-partial` | off | allow a dataset the manifest marks as partial (incomplete conversion) |
| `--lance-mp-context` | `fork` | DataLoader worker start method |
| `--lance-pin-memory` | off | pin memory in the DataLoader (webdataset does not) |
| `--continual-dataset-type` | same as `--dataset-type` | dataset type of `--continual-data` |

`--text_type` works as with webdataset: `random` picks from `sci`, `com`, `taxon`, `sci_com` and `taxon_com`, and any
other value names a column (case-insensitive). `''` (LAION's generic `txt` caption) fails: Lance datasets have no
such column.

## How batches are drawn

TreeOfLife shards are clustered by taxon, so the sampling sets how many species share a batch, which changes the
contrastive task. The default, `wds`, replays webdataset's sampling on row numbers: shards drawn the same way (with
replacement under `--dataset-resampled`), the same 5,000-sample shuffle buffer, one batch stream per worker.
`chunked` and `global` are simpler seeded shuffles, **not** like-for-like with webdataset. As with webdataset,
batches per epoch (and so the learning-rate schedule) depend on `--workers`; keep it fixed when comparing runs.

## Validation

BioCLIP 2 was trained on a TreeOfLife subset three times with the same settings: from Lance (`wds` sampler), from
webdataset, and from webdataset again to measure run-to-run variation. On the same zero-shot and few-shot
species-classification benchmarks, Lance stayed within run-to-run variation of webdataset. Training speed was the
same. This is one run per format, on one model size and one data subset.

## Data read during training

TreeOfLife 224×224 shards, `wds` sampler. Run totals are from Slurm job accounting for the Lance run and one
webdataset run in [Validation](#validation) (100 epochs, about 1 million images per epoch).

| | webdataset (tar) | Lance |
|---|---|---|
| Size on disk | 1× | about 0.68× |
| **Data read in the 100-epoch run** | **about 8.3 TB** | **about 4.9 TB (about 40% less)** |

Two causes:
- **Tar packaging.** Webdataset reads every file in a sample, unused captions included, plus a tar header and
  padding for each. Lance reads only the image and the captions in use.
- **Shuffle-buffer leftovers.** Webdataset drops what is still in its shuffle buffer at each epoch end, after
  reading it. Lance shuffles row numbers, so it reads only rows that end up in a batch.
