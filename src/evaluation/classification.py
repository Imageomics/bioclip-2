"""
Run zero-shot and few-shot classification from a shared image feature extraction pass.
"""

import datetime
import logging
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from numpy import linalg as LA
from scipy.stats import mode
from tqdm import tqdm

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from ..training.precision import get_autocast

from .data import DatasetFromFile
from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    get_dataloader,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)


FEATURE_PICKLE_NAME = "pickle.p"
ZERO_SHOT_TASK = "zero_shot"
FEW_SHOT_TASK = "few_shot"


def save_feature_bundle(base_path, bundle):
    os.makedirs(base_path, exist_ok=True)
    filepath = os.path.join(base_path, FEATURE_PICKLE_NAME)
    with open(filepath, "wb") as handle:
        pickle.dump(
            [
                bundle["features"],
                bundle["target"],
                bundle["samples"],
                bundle["class_to_idx"],
            ],
            handle,
        )
    return filepath


def load_feature_bundle(filepath):
    with open(filepath, "rb") as handle:
        data = pickle.load(handle)

    if isinstance(data, dict):
        return {
            "features": data["features"],
            "target": data["target"],
            "samples": data["samples"],
            "class_to_idx": data["class_to_idx"],
        }

    if isinstance(data, (list, tuple)) and len(data) >= 4:
        features, target, samples, class_to_idx = data[:4]
        return {
            "features": features,
            "target": target,
            "samples": samples,
            "class_to_idx": class_to_idx,
        }

    raise ValueError(f"Unsupported feature bundle format in {filepath}.")


def resolve_feature_file(args):
    if args.feature_file:
        return args.feature_file
    if args.task_type == "eval" and args.classification_tasks == [FEW_SHOT_TASK]:
        return args.pretrained
    return None


def build_dataloader(args, preprocess_val):
    dataset = DatasetFromFile(
        args.data_root,
        args.label_filename,
        transform=preprocess_val,
        classes=args.text_type,
    )
    return get_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


def extract_feature_bundle(model, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    feature_list = []
    target_list = []

    with torch.no_grad():
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            target_list.append(target.numpy())
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)

            with autocast():
                image_features, _ = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            feature_list.append(image_features.detach().cpu().numpy())

    return {
        "features": np.vstack(feature_list),
        "target": np.hstack(target_list),
        "samples": dataloader.dataset.samples,
        "class_to_idx": dataloader.dataset.class_to_idx,
    }


def build_classnames(class_to_idx):
    classnames = [None] * len(class_to_idx)
    for classname, idx in class_to_idx.items():
        classnames[idx] = classname
    return classnames


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]
            texts = tokenizer(texts).to(args.device)
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return {
        k: float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    }


def evaluate_zero_shot(model, bundle, args):
    logging.info("Starting zero-shot.")
    classnames = build_classnames(bundle["class_to_idx"])
    classifier = zero_shot_classifier(
        model,
        classnames,
        openai_imagenet_template,
        args,
    )

    features = bundle["features"]
    target = bundle["target"]
    num_classes = len(classnames)
    topk_keys = sorted({1, min(num_classes, 3), min(num_classes, 5)})
    topk = {k: 0.0 for k in topk_keys}
    n = 0.0
    logit_scale = model.logit_scale.exp()

    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            end = start + args.batch_size
            image_features = torch.from_numpy(features[start:end]).to(
                args.device, dtype=classifier.dtype
            )
            target_batch = torch.from_numpy(target[start:end]).to(args.device)
            logits = logit_scale * image_features @ classifier
            acc = accuracy(logits, target_batch, topk=topk_keys)
            for k, value in acc.items():
                topk[k] += value
            n += target_batch.size(0)

    for k in topk:
        topk[k] /= n

    results = {f"val-unseen-top{k}": value for k, value in topk.items()}
    logging.info("Finished zero-shot.")
    return results


def split_few_shot_examples(select, kshot, nfold, n_class, idx_to_class):
    test = []
    target = []
    train = []
    label = []

    random.seed(nfold)
    for class_idx, class_data in select.items():
        features = class_data["feature"][:]
        num_features = len(features)
        class_name = idx_to_class.get(class_idx, class_idx)
        if num_features < kshot:
            raise ValueError(
                f"{class_name} has only {num_features} images, fewer than the requested {kshot}-shot evaluation."
            )
        elif num_features < kshot + 5:
            logging.info(
                "%s has only %d images. Not enough for evaluation.",
                class_name,
                num_features,
            )

        random.shuffle(features)
        train.append(features[:kshot])
        test.extend(features[kshot:])
        label.extend([class_idx for _ in range(kshot)])
        target.extend([class_idx for _ in range(max(num_features - kshot, 0))])

    flatten_train = np.vstack(train)
    label = np.array(label)
    test = np.vstack(test)
    target = np.array(target)

    assert kshot * len(train) == flatten_train.shape[0] == label.shape[0]
    assert len(train) == n_class
    assert len(test) == len(target)

    return flatten_train, label, test, target


def cl2n(x_flatten, x_mean):
    x_flatten = x_flatten - x_mean
    x_flatten = x_flatten / LA.norm(x_flatten, 2, 1)[:, None]
    return x_flatten


def get_few_shot_accuracy(flatten_train, label, test, target, n_class, kshot):
    train_mean = flatten_train.mean(axis=0)

    flatten_train = cl2n(flatten_train, train_mean)
    test = cl2n(test, train_mean)
    train_center = flatten_train.reshape(n_class, kshot, flatten_train.shape[-1]).mean(1)

    num_nn = 1
    label = label[::kshot]
    subtract = train_center[:, None, :] - test
    distance = LA.norm(subtract, 2, axis=-1)
    idx = np.argpartition(distance, num_nn, axis=0)[:num_nn]
    nearest_samples = np.take(label, idx)
    out = mode(nearest_samples, axis=0, keepdims=True)[0]
    out = out.astype(int)
    return float((out == target).mean())


def build_few_shot_index(bundle):
    select = {}
    for feature, sample, class_idx in zip(
        bundle["features"], bundle["samples"], bundle["target"]
    ):
        if class_idx not in select:
            select[class_idx] = {"feature": [], "sample": []}
        select[class_idx]["feature"].append(feature)
        select[class_idx]["sample"].append(sample)
    return select


def evaluate_few_shot(bundle, args):
    logging.info("Starting few-shot.")
    select = build_few_shot_index(bundle)
    n_class = len(select)
    count = sum(len(v["feature"]) for v in select.values())
    idx_to_class = {value: key for key, value in bundle["class_to_idx"].items()}

    logging.info("Num of classes: %d. Num of samples: %d.", n_class, count)

    results = {}
    for kshot in args.kshot_list:
        acc_list = []
        for fold in range(args.nfold):
            flatten_train, label, test, target = split_few_shot_examples(
                select,
                kshot,
                fold,
                n_class,
                idx_to_class,
            )
            acc = get_few_shot_accuracy(
                flatten_train,
                label,
                test,
                target,
                n_class,
                kshot,
            )
            acc_list.append(acc)
            logging.info("%d shot No.%d ACC: %.4f", kshot, fold, acc)

        mean_acc = float(np.mean(acc_list))
        std_acc = float(np.std(acc_list))
        results[f"few-shot-{kshot}-mean"] = mean_acc
        results[f"few-shot-{kshot}-std"] = std_acc
        logging.info("Dataset: %s", args.data_root)
        logging.info("Model: %s", args.pretrained)
        logging.info("%d shot AVG ACC: %.2f", kshot, mean_acc * 100)
        logging.info("%d shot STD: %.4f", kshot, std_acc * 100)

    logging.info("Finished few-shot.")
    return results


def create_model(args, device):
    model, _, preprocess_val = create_model_and_transforms(
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

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    model.eval()
    logging.info("Model:")
    logging.info("%s", str(model))
    return model, preprocess_val


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    random_seed(args.seed, 0)
    configure_torch_backends(deterministic=args.task_type == "eval")
    device = init_device(args)

    log_base_path = configure_logging(
        args,
        "classification",
        include_workers=True,
        log_filename="out.log",
    )
    normalize_force_image_size(args)
    log_params(args, log_base_path)

    needs_model = args.task_type != "eval" or ZERO_SHOT_TASK in args.classification_tasks
    model = None
    preprocess_val = None
    if needs_model:
        model, preprocess_val = create_model(args, device)

    feature_file = resolve_feature_file(args)
    if args.task_type == "eval":
        if feature_file is None:
            raise ValueError(
                "--feature-file is required for classification eval-only runs that include zero-shot."
            )
        bundle = load_feature_bundle(feature_file)
        logging.info("Loaded image features from %s", feature_file)
    else:
        dataloader = build_dataloader(args, preprocess_val)
        start_time = time.monotonic()
        bundle = extract_feature_bundle(model, dataloader, args)
        output_dir = log_base_path or os.getcwd()
        elapsed = datetime.timedelta(seconds=time.monotonic() - start_time)
        logging.info("Image feature extraction took: %s", elapsed)

    if args.task_type == "retrieve":
        feature_file = save_feature_bundle(output_dir, bundle)
        elapsed = datetime.timedelta(seconds=time.monotonic() - start_time)
        logging.info("Saved image features to %s", feature_file)
        logging.info("Done.")
        return

    metrics = {}
    if ZERO_SHOT_TASK in args.classification_tasks:
        metrics.update(evaluate_zero_shot(model, bundle, args))
    if FEW_SHOT_TASK in args.classification_tasks:
        metrics.update(evaluate_few_shot(bundle, args))

    logging.info("Results:")
    for key, value in metrics.items():
        logging.info("  %s: %.2f", key, value * 100)
    logging.info("Done.")


if __name__ == "__main__":
    main()
