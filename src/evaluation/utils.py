import datetime
import json
import logging
import os
import random
import numpy as np
import torch

from ..training.logger import setup_logging


def get_dataloader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        return None


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    args.device = device
    device = torch.device(device)
    return device


def configure_torch_backends(deterministic):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic


def normalize_force_image_size(args):
    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]


def build_experiment_name(args, suffix, include_workers=True):
    model_name_safe = args.model.replace("/", "-")
    date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    parts = [
        date_str,
        f"model_{model_name_safe}",
        f"b_{args.batch_size}",
    ]
    if include_workers:
        parts.append(f"j_{args.workers}")
    parts.append(f"p_{args.precision}")
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


def configure_logging(args, name_suffix, include_workers=True, log_filename="out.log"):
    args.save_logs = bool(args.logs) and args.logs.lower() != "none"
    if args.save_logs and args.name is None:
        args.name = build_experiment_name(
            args, name_suffix, include_workers=include_workers
        )
    if args.save_logs:
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        args.log_path = os.path.join(log_base_path, log_filename)
    else:
        log_base_path = None
        args.log_path = None

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    return log_base_path


def log_params(args, log_base_path=None):
    logging.info("Params:")
    if args.save_logs:
        if log_base_path is None:
            log_base_path = os.path.join(args.logs, args.name)
        params_file = os.path.join(log_base_path, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
        return params_file

    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")
    return None
