import sys
import logging
import os.path
import sklearn
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils
import torch.utils.data

from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)
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


def init_classifier(input_dim: int) -> torch.nn.Module:
    """A simple MLP classifier consistent with the design in AWA2."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 512),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 85),
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

    return sklearn.metrics.f1_score(
        y_true, y_pred, average="macro", labels=np.unique(y_true)
    )


@torch.no_grad()
def get_features(
    args, backbone, img_transform, *, is_train: bool
) -> Features:
    """Extract visual features."""
    backbone = backbone.to(args.device)

    file = "trainclasses.txt" if is_train else "testclasses.txt"
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

    def __init__(self, root_dir: str, class_file: str, transform):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "JPEGImages")
        self.class_file = os.path.join(self.root_dir, "classes.txt")

        with open(self.class_file, 'r') as fp:
            all_classes = fp.readlines()
        self.all_classes = [all_class.strip() for all_class in all_classes]
        self.subset_class_file = os.path.join(self.root_dir, class_file)
        with open(self.subset_class_file, 'r') as fp:
            subset_classes = fp.readlines()
        self.subset_classes = [subset_class.strip() for subset_class in subset_classes]

        with open(os.path.join(self.root_dir, "predicate-matrix-binary.txt"), 'r') as fp:
            class_labels = fp.readlines()
        self.labeldict = {}
        for class_name, class_label in zip(self.all_classes, class_labels):
            if class_name in self.subset_classes:
                self.labeldict[class_name] = np.array([int(item) for item in class_label.strip().split(" ")])
        
        self.all_images = []
        for class_name in self.subset_classes:
            image_list = os.listdir(os.path.join(self.image_dir, class_name))
            self.all_images += [os.path.join(self.image_dir, class_name, image_name) for image_name in image_list]

        self.transform = transform

    def __getitem__(self, index: int):
        image_path = self.all_images[index]
        class_name = image_path.split("/")[-2]
        label = self.labeldict[class_name]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

    def __len__(self) -> int:
        return len(self.all_images)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    configure_torch_backends(deterministic=False)

    device = init_device(args)

    log_base_path = configure_logging(
        args, "AWA2", include_workers=True, log_filename="out.log"
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

    # 2. Get features.
    train_dataset = get_features(args, model, preprocess_val, is_train=True)
    test_dataset = get_features(args, model, preprocess_val, is_train=False)

    scores = []
    num_runs = 3
    for run in range(num_runs):
        random_seed(args.seed + run, args.rank)

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
                logging.info(
                    "Run %d/%d Epoch %d/%d: %.3f",
                    run + 1,
                    num_runs,
                    epoch + 1,
                    args.epochs,
                    score,
                )

        final_score = evaluate(args, classifier, test_loader)
        logging.info(
            "Run %d/%d final score: %.3f", run + 1, num_runs, final_score
        )
        scores.append(final_score)

    scores = np.array(scores)
    logging.info(
        "Mean score over %d runs: %.3f, std: %.3f",
        num_runs,
        np.mean(scores),
        np.std(scores),
    )
