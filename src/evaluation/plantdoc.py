import sys
import os.path
import logging
import numpy as np
from tqdm import tqdm

import torch
import torchvision.datasets

from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)
from .simpleshot import simpleshot
from ..open_clip import (
    create_model_and_transforms,
    trace_model,
)


class Features(torch.utils.data.Dataset):
    """
    A dataset of learned features (dense vectors).
    x: Float[Tensor, " n dim"] Dense feature vectors from a vision backbone.
    y: Int[Tensor, " n 85"] 0/1 labels of absence/presence of 85 different traits.
    ids: Shaped[np.ndarray, " n"] Image ids.
    """

    def __init__(
        self, x, y, ids,
    ):
        self.x = x
        self.y = y
        self.ids = ids

    @property
    def dim(self) -> int:
        """Dimension of the dense feature vectors."""
        _, dim = self.x.shape
        return dim

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.ids[index]


class Dataset(torchvision.datasets.ImageFolder):
    """
    Subclasses ImageFolder so that `__getitem__` includes the path, which we use as the ID.
    """

    def __getitem__(self, index: int) -> tuple[str, object, object]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


@torch.no_grad
def get_features(
    args, backbone, img_transform, *, is_train: bool
) -> Features:
    """
    Get a block of features from a vision backbone for a split (either train or test).

    Args:
        args: PlantDoc arguments.
        backbone: visual backbone.
        is_train: whether you want training data or the test data.
    """
    backbone = backbone.to(args.device)

    split = "train" if is_train else "test"

    root = os.path.join(args.data_root, split)
    dataset = Dataset(os.path.join(args.data_root, split), img_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
        shuffle=True,
    )

    all_features, all_labels, all_ids = [], [], []

    total = len(dataloader) if not args.debug else 2
    it = iter(dataloader)
    logging.debug("Need to embed %d batches of %d images.", total, args.batch_size)
    for b in tqdm(range(total)):
        ids, images, labels = next(it)
        images = images.to(args.device)

        with torch.amp.autocast("cuda"):
            features, _ = backbone.encode_image(images)

        all_features.append(features.cpu())
        all_labels.extend(labels)
        all_ids.extend(ids)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_ids = np.array(all_ids)
    all_labels = torch.tensor(all_labels)
    logging.info("Got features for %d images.", len(all_ids))

    return Features(all_features, all_labels, all_ids)


def choose_k_per_class(labels, *, k: int):
    """
    Returns indices for a label set that include at most `k` examples per class.

    Args:
        labels: a list of integer labels for a set of data.
        k: the maximum number of examples per class.

    Returns:
        indices for `labels` such that at most `k` examples per class are in the data.
    """
    classes = np.unique(labels)

    train_indices = np.array([], dtype=int)

    # Iterate through each class to select indices
    for cls in classes:
        # Indices corresponding to the current class
        cls_indices = np.where(labels == cls)[0]
        # Randomly shuffle the indices
        np.random.shuffle(cls_indices)
        # Select the first K indices for the train set
        cls_train_indices = cls_indices[:k]
        # Append the selected indices to the train array
        train_indices = np.concatenate((train_indices, cls_train_indices))

    # Shuffle the indices to mix classes
    np.random.shuffle(train_indices)

    return torch.from_numpy(train_indices)


if __name__ == "__main__":
    """
    Runs simpleshot `Args.n_repeats` times (default 100) with 1 training example per class, then evaluates on the validation split.
    """
    # 1. Load model
    args = parse_args(sys.argv[1:])

    configure_torch_backends(deterministic=False)

    device = init_device(args)

    log_base_path = configure_logging(
        args, "PlantDoc", include_workers=True, log_filename="out.log"
    )

    normalize_force_image_size(args)

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
    )

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    logging.info("Model:")
    logging.info(f"{str(model)}")
    log_params(args, log_base_path)

    train_features = get_features(args, model, preprocess_val, is_train=True)
    test_features = get_features(args, model, preprocess_val, is_train=False)

    all_scores = []
    for r in range(args.n_repeats):
        i = choose_k_per_class(train_features.y, k=1)

        scores = simpleshot(
            train_features.x[i],
            train_features.y[i],
            test_features.x,
            test_features.y,
            args.batch_size,
            args.device,
        )
        all_scores.append(torch.mean(scores).cpu())
        logging.info(
            "%d/%d simpleshot finished (%.1f%%)",
            r + 1,
            args.n_repeats,
            (r + 1) / args.n_repeats * 100,
        )

    all_scores = np.array(all_scores)
    logging.info(
        "Mean accuracy: %.1f%%, std: %.1f%%",
        np.mean(all_scores) * 100,
        np.std(all_scores) * 100,
    )
