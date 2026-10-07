import ast
import io
import itertools
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        # hasattr, not isinstance(DistributedSampler): the Lance batch samplers below implement set_epoch too.
        # With the isinstance gate they would be silently skipped and every epoch would replay identical data.
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    for shard_file in shards_list:
        if not os.path.exists(shard_file):
            shards_list.remove(shard_file)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        # Lisa wants the shard sizes under the per_shard key.
        # But this is not always the case.
        if "per_shard" in sizes:
            sizes = sizes["per_shard"]
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list if os.path.basename(shard) in sizes])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = any('txt' in key for key in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, is_continual=False):
    if is_continual:
        input_shards = args.continual_data
    elif is_train:
        input_shards = args.train_data
    else:
        input_shards = args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
        else:
            # Eval will just exhaust the iterator if the size is not specified.
            num_samples = args.val_num_samples or 0 
    
    logging.info(
        f"Finish counting shard total size: {num_samples}.")

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    text_type = args.continual_text_type if is_continual else args.text_type
    batch_size = args.continual_batch_size if is_continual else args.batch_size
    if text_type == 'random':
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp",sci="sci.txt", com="com.txt",taxon="taxon.txt", sci_com="sci_com.txt", taxon_com = "taxon_com.txt"),
            wds.map_dict(image=preprocess_img, sci=lambda sci: tokenizer(sci)[0], com=lambda com: tokenizer(com)[0], taxon=lambda taxon: tokenizer(taxon)[0], sci_com=lambda sci_com: tokenizer(sci_com)[0], taxon_com=lambda taxon_com: tokenizer(taxon_com)[0]),
            wds.to_tuple("image", "sci", "com", "taxon", "sci_com", "taxon_com"),
            wds.batched(batch_size, partial=not is_train)
        ])
    elif text_type == '':
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text='txt'),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(batch_size, partial=not is_train)
        ])
    else:
        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text=text_type+'.txt'),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(batch_size, partial=not is_train)
        ])


    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)
    
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


# =============================================================================================
# Lance (v1: verbatim columns).  Datasets written by the wds_to_lance.py converter:
# one column per tar member (uuid, jpg, scientific_name, common_name, taxonomic_name, sci, com, taxon,
# sci_com, taxon_com, taxonTag, taxonTag_com, source_shard, source_index).  Selected with
# --dataset-type lance.  Contract identical to get_wds_dataset: the DataLoader yields
#   (images, sci, com, taxon, sci_com, taxon_com)  for --text_type random   (train.py:120-126 unchanged)
#   (images, text)                                 otherwise, text = column named by --text_type
# with images float32 [B,3,H,W] and texts int64 [B,77]; dataloader.num_batches / num_samples use the same
# padded-epoch arithmetic as webdataset; DataInfo.set_epoch reaches the sampler.
# Which webdataset semantics are replicated, approximated or dropped is documented on the samplers below.
# =============================================================================================
LANCE_V1_IMAGE_COLUMN = "jpg"
LANCE_V1_RANDOM_POOL = ("sci", "com", "taxon", "sci_com", "taxon_com")  # order matters: train.py unpacks it


def _lance_identity_collate(batch):
    """__getitems__ already returns a collated batch; whole batches are assembled inside the worker, like webdataset."""
    return batch


def _lance_decode_pilrgb(data):
    """Exactly what wds.decode('pilrgb') does per image: PIL open, load, convert('RGB') (mode-'L' and CMYK rows included)."""
    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        return img.convert("RGB")


def _lance_pick(buf, rng):
    # webdataset.filters.pick
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def lance_wds_shuffle(data, bufsize, initial, rng):
    """Verbatim port of webdataset.filters._shuffle (0.2.86) over an iterator of row ids.

    `data` must be an iterator.  The buffer grows by one per emitted sample until bufsize, then streams;
    a sample can linger in the buffer for a long time, which is why webdataset batches span more than
    bufsize + batch_size tar rows."""
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield _lance_pick(buf, rng)
    while len(buf) > 0:
        yield _lance_pick(buf, rng)


def lance_epoch_arithmetic(num_samples, batch_size, world_size, workers, floor=False):
    """The padded-epoch arithmetic of get_wds_dataset (lines 'roll over and repeat ...'), factored out so the
    Lance path and the unit test use the identical formula.  NOTE --workers changes the epoch length
    (e.g. 956,203 rows, 4096 x 4: workers 8 -> 64 batches, 12 -> 60, 16 -> 64, 20 -> 60, 24 -> 72)."""
    round_fn = math.floor if floor else math.ceil
    global_batch_size = batch_size * world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    return num_batches, num_worker_batches, num_batches * global_batch_size


def lance_dataset_info(uri):
    """(num_rows, columns, shards=[(name, rows, row_offset), ...], source, manifest) for a v1 Lance dataset.

    Prefers the plain-JSON manifest wds_to_lance.py writes next to the dataset (<uri>.manifest.json), so the
    parent process never imports lance before forking DataLoader workers.  Falls back to opening the dataset,
    scanning the source_shard column and synthesizing the manifest fields from the schema metadata (so the
    partial/verify checks still apply), then drops the handle."""
    uri = uri.rstrip('/')
    manifest = uri + ".manifest.json"
    if os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        shards = [(s["name"], int(s["rows"]), int(s["offset"])) for s in m["shards"]]
        return int(m["num_rows"]), list(m["columns"]), shards, manifest, m
    logging.warning(f"{manifest} not found; opening {uri} in the parent process to read its size and shard layout")
    import lance
    ds = lance.dataset(uri)
    num_rows = ds.count_rows()
    columns = list(ds.schema.names)
    sh = ds.to_table(columns=["source_shard"]).column("source_shard").to_numpy(zero_copy_only=False)
    change = np.flatnonzero(sh[1:] != sh[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(int)
    ends = np.concatenate([change, [len(sh)]]).astype(int)
    shards = [(str(sh[s]), int(e - s), int(s)) for s, e in zip(starts, ends)]
    md = dict(ds.schema_metadata)
    del sh, ds
    synth = None
    if md.get("tol.spec"):  # written by wds_to_lance.py: rebuild what the manifest would have said
        def _int(k):
            v = md.get(k)
            return int(v) if v not in (None, "") else None
        synth = {
            "partial": md.get("tol.partial_commit") == "true",
            "shards_committed": _int("tol.shards_committed"), "shards_listed": _int("tol.shards_listed"),
            "limit_rows": _int("tol.limit_rows"),
            "verify": {"status": md.get("tol.verify.status")} if md.get("tol.verify.status") else None,
            "sizes_json_total": _int("tol.sizes_json_total"), "synthesized_from": "schema metadata (no manifest)",
        }
    return num_rows, columns, shards, uri, synth


def _lance_resolve_column(text_type, columns):
    if text_type in columns:
        return text_type
    lower = {c.lower(): c for c in columns}
    if text_type.lower() in lower:  # webdataset lowercases member suffixes; accept the same spelling
        logging.info(f"--text_type {text_type!r} matched Lance column {lower[text_type.lower()]!r} case-insensitively")
        return lower[text_type.lower()]
    raise ValueError(f"--text_type {text_type!r}: no such column in the Lance dataset; available: {columns}")


class LanceVerbatimDataset(Dataset):
    """Map-style view of a v1 Lance dataset that fetches WHOLE BATCHES per __getitems__ call.

    The DataLoader is built with a batch sampler, so torch's fetcher calls __getitems__(list_of_row_ids) once
    per batch inside the worker: one ds.take(), PIL decode (pilrgb), preprocess, stack, tokenize -- the same
    per-worker work the webdataset pipeline does, with the same transforms and tokenizer objects.  The Lance
    handle is opened lazily in whichever process first fetches (never in the parent), which keeps fork-started
    workers safe."""

    def __init__(self, uri, num_rows, image_column, text_columns, preprocess_img, tokenizer, allow_duplicates=False):
        self.uri = uri
        self.num_rows = num_rows
        self.image_column = image_column
        self.text_columns = list(text_columns)
        self.preprocess_img = preprocess_img
        self.tokenizer = tokenizer
        # Duplicate rows inside one batch are false negatives for InfoNCE.  The chunked/global samplers cannot
        # produce them (hard error).  The wds-emulation sampler CAN, exactly as webdataset does: when a worker
        # draws the same shard twice in a row, a row lingering in the 5000 buffer meets its second copy
        # ((1 - 1/5000)^rows_per_shard ~ 0.3% of a 29k-row shard, ~1% of worker-epochs) -- so there it is
        # tolerated and counted, never silently deduplicated (that would change the batch size).
        self.allow_duplicates = bool(allow_duplicates)
        self._ds = None
        self._pid = None
        self.duplicate_batches = 0

    def _dataset(self):
        if self._ds is None or self._pid != os.getpid():
            import lance
            ds = lance.dataset(self.uri)
            n = ds.count_rows()
            if n != self.num_rows:
                raise RuntimeError(f"{self.uri}: dataset has {n} rows but the manifest/loader expects {self.num_rows}; "
                                   f"re-run wds_to_lance.py --commit-only or delete the stale manifest")
            missing = [c for c in [self.image_column, *self.text_columns] if c not in ds.schema.names]
            if missing:
                raise RuntimeError(f"{self.uri}: missing columns {missing}; have {ds.schema.names}")
            self._ds, self._pid = ds, os.getpid()
        return self._ds

    def __len__(self):
        return self.num_rows

    def __getitems__(self, indices):
        idx = [int(i) for i in indices]
        uniq = sorted(set(idx))
        if len(uniq) != len(idx):
            if not self.allow_duplicates:
                raise RuntimeError("a batch contains the same row twice (duplicates inside a batch are false negatives "
                                   "for InfoNCE); this sampler mode must never produce them")
            self.duplicate_batches += 1
            if self.duplicate_batches <= 3:
                logging.warning(f"Lance batch with {len(idx) - len(uniq)} duplicate row(s) (webdataset-like: same shard "
                                f"drawn twice in a row); kept as-is, batch #{self.duplicate_batches} in this worker")
        tbl = self._dataset().take(uniq, columns=[self.image_column, *self.text_columns])
        if tbl.num_rows != len(uniq):
            raise RuntimeError(f"take({len(uniq)}) returned {tbl.num_rows} rows")
        if len(uniq) != len(idx):  # expand back to the requested multiset (Arrow take on positions)
            pos = {r: i for i, r in enumerate(uniq)}
            tbl = tbl.take([pos[r] for r in idx])
        images = torch.stack([self.preprocess_img(_lance_decode_pilrgb(b))
                              for b in tbl.column(self.image_column).to_pylist()])
        texts = tuple(self.tokenizer(tbl.column(c).to_pylist()) for c in self.text_columns)
        return (images, *texts)

    def __getitem__(self, index):
        images, *texts = self.__getitems__([index])
        return (images[0], *[t[0] for t in texts])


class LanceWdsTrainBatchSampler(torch.utils.data.Sampler):
    """Reproduces the webdataset TRAIN sampling PROCESS of get_wds_dataset on row ids, per (epoch, rank, worker):

      resampled  : ResampledShards2 -- shards drawn uniformly WITH replacement, forever;
      otherwise  : detshuffle2(seed + epoch) over the shard list, split_by_node, split_by_worker, and the pipeline
                   restarting from the top when the worker's shards run out (DataPipeline.iterator repeats iterator1);
      then       : rows of each shard in tar order -> the 5000/1000 sample shuffle buffer (_shuffle, ported verbatim)
                   -> batches of batch_size with partial=False -> with_epoch(num_worker_batches);
    and interleaves the workers' batch streams round-robin, which is the order a DataLoader returns batches from
    iterable-dataset workers.  Batch i of an epoch therefore has the same composition statistics as webdataset's
    batch i (single-shard windows, shard multiplicity, cross-rank shard collisions); only the RNG draws differ
    (webdataset seeds its sample shuffle from time.time(), so its own draws are not reproducible either).
    Row ids inside a batch are sorted (Lance takes them in one pass); order within a batch carries no meaning."""

    def __init__(self, shards, batch_size, num_worker_batches, num_workers, rank, world_size, seed, epoch=0,
                 resampled=True, bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL,
                 shard_bufsize=_SHARD_SHUFFLE_SIZE, shard_initial=_SHARD_SHUFFLE_INITIAL, vary_shuffle_per_pass=False):
        # vary_shuffle_per_pass: re-seed the SAMPLE shuffle on every __iter__ (the shard draws stay fixed for the
        # epoch).  Used for the --continual-data stream, which train.py re-iterates (IterLoader) without ever calling
        # set_epoch: webdataset's time-seeded wds.shuffle differs per pass there while its shard sequence is frozen.
        self.vary_shuffle_per_pass = bool(vary_shuffle_per_pass)
        self._pass = 0
        self.shards = [(str(n), int(r), int(o)) for n, r, o in shards]
        self.batch_size = int(batch_size)
        self.num_worker_batches = int(num_worker_batches)
        self.num_workers = max(1, int(num_workers))
        self.rank, self.world_size, self.seed, self.epoch = int(rank), int(world_size), int(seed), int(epoch)
        self.resampled = bool(resampled)
        self.bufsize, self.initial = int(bufsize), min(int(initial), int(bufsize))
        self.shard_bufsize, self.shard_initial = int(shard_bufsize), int(shard_initial)
        if not self.shards:
            raise ValueError("no shards")
        if not self.resampled and len(self.shards) < self.num_workers * self.world_size:
            raise ValueError("number of shards must be >= total workers")  # same assert as get_wds_dataset

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_worker_batches * self.num_workers

    def _shard_indices(self, worker, rng_shards):
        if self.resampled:
            n = len(self.shards)
            while True:  # ResampledShards2: rng.choice(urls) for _ in range(sys.maxsize)
                yield rng_shards.randrange(n)
        else:
            rng = random.Random()
            rng.seed(self.seed + self.epoch)  # detshuffle2: same order on every rank/worker for this epoch
            order = lance_wds_shuffle(iter(range(len(self.shards))), self.shard_bufsize, self.shard_initial, rng)
            order = itertools.islice(order, self.rank, None, self.world_size)   # split_by_node
            order = itertools.islice(order, worker, None, self.num_workers)     # split_by_worker
            yield from order

    def _row_stream(self, worker, rng_shards):
        for si in self._shard_indices(worker, rng_shards):
            _, rows, offset = self.shards[si]
            yield from range(offset, offset + rows)  # tarfile_to_samples: tar order

    def worker_batches(self, worker, pass_id=0):
        tag = f"tol-lance-v1|seed={self.seed}|epoch={self.epoch}|rank={self.rank}|worker={worker}"
        rng_shards = random.Random(tag + "|shards")
        rng_buf = random.Random(tag + f"|buffer|pass={pass_id}" if pass_id else tag + "|buffer")
        return self.batches_with_rngs(worker, rng_shards, rng_buf)

    def batches_with_rngs(self, worker, rng_shards, rng_buf):
        """The replay itself, with the two RNGs passed in (so a test can inject the exact RNG objects handed to the
        real webdataset stages and compare index for index)."""
        batches = []
        passes = 0
        while len(batches) < self.num_worker_batches:  # DataPipeline.iterator(): repeat iterator1() until with_epoch cuts
            passes += 1
            if passes > 10_000:
                raise RuntimeError("worker produced no full batches; batch_size larger than the worker's rows?")
            shuffled = lance_wds_shuffle(self._row_stream(worker, rng_shards), self.bufsize, self.initial, rng_buf)
            batch = []
            for r in shuffled:
                batch.append(r)
                if len(batch) == self.batch_size:  # wds.batched(partial=False): trailing partial batch is dropped
                    batches.append(sorted(batch))
                    batch = []
                    if len(batches) == self.num_worker_batches:
                        break
        return batches

    def __iter__(self):
        self._pass += 1
        pass_id = self._pass if self.vary_shuffle_per_pass else 0
        streams = [self.worker_batches(w, pass_id) for w in range(self.num_workers)]
        for i in range(len(self)):  # DataLoader round-robin: batch i comes from worker i % num_workers
            yield streams[i % self.num_workers][i // self.num_workers]


class LanceChunkedBatchSampler(torch.utils.data.Sampler):
    """NOT webdataset-like (a cleaner sampler that gives different batches): one seeded permutation of all rows per
    epoch, identical on every rank.  chunk == 0: global permutation.  chunk > 0: rows are permuted WITHIN contiguous
    chunks of `chunk` rows (tar order) and the chunk order is permuted, so a batch is drawn from ~chunk contiguous
    tar rows (one to three shards; train_small shards are 28-30k rows), approximating webdataset's single-shard
    windows.  Padding to the padded epoch
    length wraps around (those rows are seen twice per epoch, >= num_rows draws apart).  Global step s uses rows
    order[s*G:(s+1)*G] with G = batch_size * world_size, and rank r takes its slice of that."""

    def __init__(self, num_rows, batch_size, num_batches, world_size, rank, seed, epoch=0, chunk=0, vary_per_pass=False):
        self.num_rows, self.batch_size, self.num_batches = int(num_rows), int(batch_size), int(num_batches)
        self.world_size, self.rank, self.seed, self.epoch, self.chunk = int(world_size), int(rank), int(seed), int(epoch), int(chunk)
        self.vary_per_pass = bool(vary_per_pass)  # continual stream: a new permutation on every __iter__
        self._pass = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def epoch_order(self, pass_id=0):
        rng = np.random.default_rng([self.seed, self.epoch, 0x1A9CE, int(pass_id)])
        n = self.num_rows
        if self.chunk <= 0 or self.chunk >= n:
            order = rng.permutation(n)
        else:
            parts = []
            for c in rng.permutation(-(-n // self.chunk)):
                start = int(c) * self.chunk
                parts.append(start + rng.permutation(min(self.chunk, n - start)))
            order = np.concatenate(parts)
        need = self.num_batches * self.batch_size * self.world_size
        return np.resize(order, need) if need > n else order[:need]

    def __iter__(self):
        self._pass += 1
        order = self.epoch_order(self._pass if self.vary_per_pass else 0)
        G = self.batch_size * self.world_size
        for s in range(self.num_batches):
            g = order[s * G:(s + 1) * G]
            yield sorted(int(x) for x in g[self.rank * self.batch_size:(self.rank + 1) * self.batch_size])


class LanceWdsValBatchSampler(torch.utils.data.Sampler):
    """Reproduces get_wds_dataset's VAL batch order: SimpleShardList in sizes.json order -> split_by_worker (worker w
    gets shards w, w+W, w+2W, ...) -> rows in tar order -> batched(batch_size, partial=True) per worker -> the
    DataLoader's round-robin over workers with exhausted workers skipped.  With the same --workers as the
    webdataset run, batch i holds exactly the same rows (including the per-worker partial batches), so
    evaluate()'s random.seed(i) text-type choice lands on the same batches."""

    def __init__(self, shards, batch_size, num_workers):
        W = max(1, int(num_workers))
        self.streams = []
        for w in range(W):
            rows = []
            for name, n, offset in shards[w::W]:
                rows.extend(range(int(offset), int(offset) + int(n)))
            self.streams.append([rows[i:i + batch_size] for i in range(0, len(rows), batch_size)])
        self.total = sum(len(s) for s in self.streams)

    def __len__(self):
        return self.total

    def __iter__(self):
        pos = [0] * len(self.streams)
        remaining = self.total
        while remaining:
            for w, stream in enumerate(self.streams):
                if pos[w] < len(stream):
                    yield stream[pos[w]]
                    pos[w] += 1
                    remaining -= 1


def get_lance_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, is_continual=False):
    if is_continual:
        uri = args.continual_data
    elif is_train:
        uri = args.train_data
    else:
        uri = args.val_data
    assert uri is not None
    uri = uri.rstrip('/')
    text_type = args.continual_text_type if is_continual else args.text_type
    batch_size = args.continual_batch_size if is_continual else args.batch_size
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_rows, columns, shards, info_source, manifest = lance_dataset_info(uri)
    if manifest is not None:
        # A dataset committed with --allow-partial-commit (or a truncated --limit-rows smoke dataset) has fewer
        # rows: num_batches / total_steps / the LR schedule would silently differ from the webdataset arm.
        partial = bool(manifest.get("partial")) or manifest.get("shards_committed") != manifest.get("shards_listed") \
            or manifest.get("limit_rows") is not None
        if partial:
            msg = (f"{uri}: manifest says the dataset is partial (shards {manifest.get('shards_committed')}/"
                   f"{manifest.get('shards_listed')}, limit_rows={manifest.get('limit_rows')}); its epoch length would "
                   f"not match the webdataset run")
            if not getattr(args, 'lance_allow_partial', False):
                raise RuntimeError(msg + " (pass --lance-allow-partial to train on it anyway)")
            logging.warning(msg + " -- continuing because --lance-allow-partial was given")
        v = manifest.get("verify") or {}
        if v.get("status") != "pass":
            logging.warning(f"{uri}: the dataset has not passed wds_to_lance.py --verify (manifest verify={v or None}); "
                            f"byte-fidelity to the tars is unverified")
        if manifest.get("sizes_json_total") not in (None, num_rows):
            logging.warning(f"{uri}: {num_rows} rows but sizes.json said {manifest['sizes_json_total']}; the epoch "
                            f"length derives from the row count here, webdataset used sizes.json")
    if LANCE_V1_IMAGE_COLUMN not in columns:
        raise ValueError(f"{uri}: no {LANCE_V1_IMAGE_COLUMN!r} column; is this a v1 (verbatim) Lance dataset? columns={columns}")
    if text_type == 'random':
        text_columns = list(LANCE_V1_RANDOM_POOL)
    elif text_type == '':
        # A live value: slurm/train.sh passes --continual_text_type '' for LAION shards, where it means the
        # generic 'txt' member.  The v1 TreeOfLife schema has no such column, so never alias it to anything.
        raise ValueError(f"{uri}: --{'continual_' if is_continual else ''}text_type '' selects the generic 'txt' "
                         f"webdataset member; this Lance dataset has no 'txt' column (columns={columns}). "
                         f"Pass an explicit column name, e.g. taxon.")
    else:
        text_columns = [_lance_resolve_column(text_type, columns)]
    missing = [c for c in text_columns if c not in columns]
    if missing:
        raise ValueError(f"{uri}: missing text columns {missing}; available {columns}")

    # Same precedence as get_wds_dataset: the dataset's own size first, --train-num-samples only as a fallback.
    num_samples = num_rows
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError('Currently, the number of dataset samples must be specified for the training dataset. '
                                   'Please specify it via `--train-num-samples` if no dataset length info is present.')
        else:
            num_samples = args.val_num_samples or 0
    elif is_train and args.train_num_samples and args.train_num_samples != num_samples:
        logging.warning(f"--train-num-samples {args.train_num_samples} ignored: the Lance dataset has {num_samples} rows")
    logging.info(f"Finish counting shard total size: {num_samples}.")
    logging.info(f"Lance v1 dataset {uri}: {num_rows} rows, {len(shards)} source shards (layout from {info_source}), "
                 f"text columns {text_columns}")

    workers = args.workers
    mode = getattr(args, 'lance_sampler', 'wds') if is_train else 'val'
    dataset = LanceVerbatimDataset(uri, num_rows, LANCE_V1_IMAGE_COLUMN, text_columns, preprocess_img, tokenizer,
                                   allow_duplicates=(mode == 'wds'))
    bufsize = getattr(args, 'lance_shuffle_buffer', None)
    if bufsize is None:
        bufsize = _SAMPLE_SHUFFLE_SIZE
    if bufsize < 1:
        raise ValueError(f"--lance-shuffle-buffer must be >= 1 (got {bufsize}); webdataset uses {_SAMPLE_SHUFFLE_SIZE}")
    if is_train:
        num_batches, num_worker_batches, num_samples = lance_epoch_arithmetic(
            num_samples, batch_size, args.world_size, workers, floor)
        if mode == 'wds':
            sampler = LanceWdsTrainBatchSampler(
                shards, batch_size, num_worker_batches, workers, args.rank, args.world_size, args.seed, epoch=epoch,
                resampled=resampled, bufsize=bufsize, initial=min(_SAMPLE_SHUFFLE_INITIAL, bufsize),
                vary_shuffle_per_pass=is_continual)
        elif mode in ('chunked', 'global'):
            chunk = getattr(args, 'lance_shuffle_chunk', 0) if mode == 'chunked' else 0
            sampler = LanceChunkedBatchSampler(num_rows, batch_size, num_batches, args.world_size, args.rank, args.seed,
                                               epoch=epoch, chunk=chunk, vary_per_pass=is_continual)
        else:
            raise ValueError(f"unknown --lance-sampler {mode!r}")
        logging.info(f"Lance train sampler: {type(sampler).__name__} (resampled={resampled}, shuffle_buffer={bufsize}, "
                     f"chunk={getattr(sampler, 'chunk', None)}), {num_batches} batches/rank/epoch = "
                     f"{num_worker_batches} x {max(1, workers)} workers, padded epoch {num_samples} samples")
    else:
        sampler = LanceWdsValBatchSampler(shards, batch_size, workers)
        # get_wds_dataset reports ceil(num_samples / batch_size); the loop actually iterates len(sampler) batches
        # (per-worker partial batches, like webdataset's split_by_worker + batched(partial=True)).
        num_batches = math.ceil(num_samples / batch_size)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_lance_identity_collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=bool(getattr(args, 'lance_pin_memory', False)),   # webdataset's WebLoader never pins; off by default
        multiprocessing_context=(getattr(args, 'lance_mp_context', None) or 'fork') if workers > 0 else None,
        prefetch_factor=2 if workers > 0 else None,                   # DataLoader default, same as WebLoader
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    dataloader.num_batches_iterated = len(sampler)
    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "lance":
        # v1 verbatim datasets written by wds_to_lance.py (defined above).
        return get_lance_dataset
    elif dataset_type == "auto":
        ext = data_path.rstrip('/').split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        elif ext in ['lance']:
            return get_lance_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.continual_data:
        # --continual-dataset-type lets the LAION replay stream stay on webdataset while the main stream is Lance.
        continual_type = getattr(args, 'continual_dataset_type', None) or args.dataset_type
        data["continual"] = get_dataset_fn(args.continual_data, continual_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer, is_continual=True)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
