"""
Modified from [biobench](https://github.com/samuelstevens/biobench/blob/main/biobench/newt/__init__.py).
---
# NeWT: Natural World Tasks
NeWT is a collection of 164 binary classification tasks related to visual understanding of the natural world ([CVPR 2021 paper](https://arxiv.org/abs/2103.16483), [code](https://github.com/visipedia/newt/tree/main)).
We evaluate a vision model by extracting visual features for each image, fitting a linear SVM to the training examples, and evaluating on the test data.
We aggregate scores across all 164 tasks.
If you use this evaluation, be sure to cite the original work:
```
@inproceedings{van2021benchmarking,
  title={Benchmarking Representation Learning for Natural World Image Collections},
  author={Van Horn, Grant and Cole, Elijah and Beery, Sara and Wilber, Kimberly and Belongie, Serge and Mac Aodha, Oisin},
  booktitle={Computer Vision and Pattern Recognition},
  year={2021}
}
```
"""

import os
import sys
import logging
import datetime
import scipy.stats
import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

import sklearn.svm
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.model_selection
import torch

from .params import parse_args
from .utils import init_device, random_seed
from ..open_clip import (
    create_model_and_transforms,
    trace_model,
)
from ..training.logger import setup_logging


class Dataset(torch.utils.data.Dataset):
    """
    A dataset that returns `(example id, image tensor)` tuples.
    """

    def __init__(self, dir: str, df, transform):
        self.transform = transform
        self.image_ids = df.get_column("id").to_list()
        self.dir = dir

    def __getitem__(self, i: int):
        image_id = self.image_ids[i]
        image = Image.open(os.path.join(self.dir, f"{image_id}.jpg"))
        if self.transform is not None:
            image = self.transform(image)
        return image_id, image

    def __len__(self) -> int:
        return len(self.image_ids)


class Task:
    """
    Task is a group of features and labels for an SVM + a train/test split.
    """

    def __init__(self, name, cluster, sub_cluster, features, labels, is_train, example_ids):
        self.name = name
        self.cluster = cluster
        self.sub_cluster = sub_cluster
        self.features = features
        self.labels = labels
        self.is_train = is_train
        self.example_ids = example_ids

    @property
    def splits(
        self,
    ):
        """
        The features and labels for train and test splits.

        Returned as `(x_train, y_train), (x_test, y_test)`.
        """
        x_train = self.features[self.is_train]
        y_train = self.labels[self.is_train]
        x_test = self.features[~self.is_train]
        y_test = self.labels[~self.is_train]

        return (x_train, y_train), (x_test, y_test)


@torch.no_grad()
def get_all_task_specific_features(args, backbone, img_transform):
    """ """
    labels_csv_name = "newt2021_labels.csv"
    labels_csv_path = os.path.join(args.data_root, labels_csv_name)
    images_dir_name = "newt2021_images"
    images_dir_path = os.path.join(args.data_root, images_dir_name)

    df = pl.read_csv(labels_csv_path).with_row_index()

    backbone = backbone.to(args.device)

    dataset = Dataset(images_dir_path, df, img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    all_features, all_ids = [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    for b in tqdm(range(total)):
        ids, images = next(it)
        images = images.to(args.device)

        features, _ = backbone.encode_image(images)
        features = torch.nn.functional.normalize(features, dim=-1)
        all_features.append(features.cpu())

        all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)

    for task in df.get_column("task").unique():
        task_df = df.filter(pl.col("task") == task)

        task_idx = task_df.get_column("index").to_numpy().astype(int)
        features = all_features[task_idx].numpy()
        ids = all_ids[task_idx]

        labels = task_df.get_column("label").to_numpy()
        is_train = task_df.select(pl.col("split") == "train").get_column("split")

        cluster = task_df.item(row=0, column="task_cluster")
        sub_cluster = task_df.item(row=0, column="task_subcluster")
        if not sub_cluster:
            sub_cluster = "none"
        yield Task(task, cluster, sub_cluster, features, labels, is_train.to_numpy(), ids)


def l2_normalize(features):
    """Normalizes a batch of vectors to have L2 unit norm."""
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


def init_svc(seed: int = 42):
    """Create a new, randomly initialized SVM with a random hyperparameter search over kernel, C and gamma. It uses only 16 jobs in parallel to prevent overloading the CPUs on a shared machine."""
    return sklearn.model_selection.RandomizedSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.svm.SVC(C=1.0, kernel="rbf"),
        ),
        {
            "svc__C": scipy.stats.loguniform(a=1e-3, b=1e1),
            "svc__kernel": ["rbf", "linear", "sigmoid", "poly"],
            "svc__gamma": scipy.stats.loguniform(a=1e-4, b=1e-3),
        },
        n_iter=100,
        n_jobs=16,
        random_state=seed,
    )


if __name__ == "__main__":
    """
    The NeWT benchmark.
    First, get features for all images.
    Second, select the subsets of features that correspond to different tasks and train an SVM.
    Third, evaluate the SVM and report results.
    """
    # 1. Load model
    args = parse_args(sys.argv[1:])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_device(args)

    args.save_logs = args.logs and args.logs.lower() != "none"

    # get the name of the experiments
    if args.save_logs and args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot_iid",
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

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=None,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        load_weights_only=False,
    )

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    if  args.save_logs is None:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    else:
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # 2. Get features.
    all_task_features = get_all_task_specific_features(args, model, preprocess_val)

    # Fit SVMs.
    y_preds = []
    y_trues = []
    for task in all_task_features:
        (x_train, y_train), (x_test, y_test) = task.splits

        x_mean = x_train.mean(axis=0, keepdims=True)

        x_train = x_train - x_mean
        x_train = l2_normalize(x_train)

        x_test = x_test - x_mean
        x_test = l2_normalize(x_test)

        svc = init_svc(args.seed)

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        y_preds.append(y_pred)
        y_trues.append(y_test)
    
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    acc = np.mean(y_preds == y_trues)
    logging.info(f"Accuracy: {acc:.4f}")
