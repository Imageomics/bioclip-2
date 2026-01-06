"""
Do zero-shot classification on IID data with both seen and unseen classes.

Single-process. If you want to run all evaluations of a single model at once, look
in scripts/.

Writes the output to a plaintext and JSON format in the logs directory.
"""

import logging
import sys
import torch
import torch.nn.functional as F
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


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return dict([
        (k,float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()))
        for k in topk
    ])


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        n = 0.0
        topk = dict()
        for i in (1,min(len(dataloader.dataset.classes),3), min(len(dataloader.dataset.classes),5)):
            topk[i] = 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device) #images.shape: torch.Size([batch_size, 3 rgb channels, image_height, image_width])
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                image_features, _ = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                # logits = 100.0 * image_features @ classifier
                logits = model.logit_scale.exp() * image_features @ classifier

            # measure accuracy
            acc = accuracy(logits, target, topk=topk.keys())
            for k,v in acc.items():
                topk[k] += v
            n += images.size(0)

    for k,v in acc.items():
        topk[k] /= n
    return topk


def zero_shot_eval(model, data, args):
    results = {}

    logging.info("Starting zero-shot.")

    for split in data:
        logging.info("Building zero-shot %s classifier.", split)
        classnames = [c for c in data[split].dataset.classes]

        classifier = zero_shot_classifier(
            model, classnames, openai_imagenet_template, args
        )

        topk = run(model, classifier, data[split], args)

        for k,v in topk.items():
            results[f"{split}-top{k}"] = v

        logging.info("Finished zero-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished zero-shot.")

    return results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    configure_torch_backends(deterministic=False)

    device = init_device(args)

    log_base_path = configure_logging(
        args, "zero_shot_iid", include_workers=True, log_filename="out.log"
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

    # initialize datasets
    data = {
        "val-unseen": get_dataloader(
            DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val, classes=args.text_type),
            batch_size=args.batch_size,num_workers=args.workers
        ),
    }

    model.eval()
    metrics = zero_shot_eval(model, data, args)
    
    logging.info("Results:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value*100:.2f}")
    logging.info("Done.")
