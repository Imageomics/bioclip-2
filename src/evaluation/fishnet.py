import sys
import logging
import os.path
import datetime
import numpy as np
import polars as pl
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils
import torch.utils.data

from .params import parse_args
from .utils import init_device, random_seed
from ..open_clip import (
    create_model_and_transforms,
    trace_model,
)
from ..training.logger import setup_logging


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


def init_classifier(input_dim: int) -> torch.nn.Module:
    """A simple MLP classifier consistent with the design in AWA2."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 9),
    )


def evaluate(
    args, classifier: torch.nn.Module, dataloader
):
    """
    Evaluates the trained classifier on a test split.

    Returns:
        a list of Examples.
    """
    total = 2 if args.debug else len(dataloader)
    it = iter(dataloader)
    y_pred, y_true = [], []
    for b in range(total):
        features, labels, ids = next(it)
        features = features.to(args.device)
        labels = labels.numpy()
        ids = ids.numpy()
        with torch.no_grad():
            pred_logits = classifier(features)
        pred_logits = (pred_logits > 0.5).cpu().numpy()
        y_pred.append(pred_logits)
        y_true.append(labels)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    correct = np.all(y_pred == y_true, axis=1)
    acc = np.sum(correct) / len(y_pred)

    return acc


@torch.no_grad()
def get_features(
    args, backbone, img_transform, *, is_train: bool
) -> Features:
    """Extract visual features."""
    backbone = backbone.to(args.device)

    file = "train.csv" if is_train else "test.csv"
    dataset = ImageDataset(args.data_root, file, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers
    )

    all_features, all_labels, all_ids = [], [], []

    total = 2 if args.debug else len(dataloader)
    it = iter(dataloader)
    for b in tqdm(range(total)):
        images, labels, _ = next(it)
        images = images.to(args.device)

        features, _ = backbone.encode_image(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

        ids = np.arange(len(labels)) + b * args.batch_size
        all_ids.append(ids)

    # Keep the Tensor data type for subsequent training
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_ids = np.concatenate(all_ids, axis=0)

    logging.info(f"Extracted {len(all_features)} features.")

    return Features(all_features, all_labels, all_ids)


class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset that loads the required attribute labels.
    """

    def __init__(self, root_dir: str, csv_file: str, transform):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.df = pl.read_csv(self.csv_file).with_row_index()
        self.all_columns = [
            "FeedingPath",
            "Tropical",
            "Temperate",
            "Subtropical",
            "Boreal",
            "Polar",
            "freshwater",
            "saltwater",
            "brackish",
        ]
        for col in self.all_columns:
            self.df = self.df.filter(self.df[col].is_not_null())
        self.transform = transform

        # Corresponding column indices
        self.image_col = 4
        self.folder_col = 13
        self.label_cols = [15, 16, 17, 18, 19, 20, 21, 22, 23]
        logging.info("csv file: %s has %d item.", csv_file, len(self.df))

    def __getitem__(self, index: int):
        row_data = self.df.row(index)
        image_name = row_data[self.image_col]
        image_name = image_name.split("/")[-1]
        folder = row_data[self.folder_col]
        image_path = os.path.join(self.root_dir, "Image_Library", folder, image_name)
        image = Image.open(image_path)

        # Extract the required attribute labels.
        label = []
        for col in self.label_cols:
            value = row_data[col]
            if col == 15:
                if value == "pelagic":
                    value = 1
                elif value == "benthic":
                    value = 0
                else:
                    raise ValueError("FeedingPath can only be pelagic or benthic.")
            label.append(value)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":
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
    train_dataset = get_features(args, model, preprocess_val, is_train=True)
    test_dataset = get_features(args, model, preprocess_val, is_train=False)

    # 3. Set up classifier.
    classifier = init_classifier(train_dataset.dim).to(args.device)

    # 4. Load datasets for classifier.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 5. Fit the classifier.
    for epoch in range(args.epochs):
        total = 2 if args.debug else len(train_loader)
        it = iter(train_loader)
        for b in range(total):
            features, labels, _ = next(it)
            features = features.to(args.device)
            labels = labels.to(args.device, dtype=torch.float)
            output = classifier(features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate the classifier.
        if (epoch + 1) % args.eval_every == 0:
            score = evaluate(args, classifier, test_loader)
            logging.info(f"Epoch {epoch + 1}/{args.epochs}: {score:.3f}")
