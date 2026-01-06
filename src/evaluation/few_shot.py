"""
Do few-shot classification.

Single-process. If you want to run all evaluations of a single model at once, look in scripts/.
"""

import datetime
import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import pickle
import numpy as np
import random
from numpy import linalg as LA
from scipy.stats import mode

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

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    trace_model,
)
from ..training.precision import get_autocast



def save_pickle(base_path, data):
    print('base_path:',base_path)
    os.makedirs(base_path, exist_ok=True)
    file = os.path.join(base_path,'pickle.p')
    print('pickle file location:',file)
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    return file


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def run(model, dataloader, args, output_dir):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        feature_list = []
        target_list = []
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            target_list.append(target.numpy())
            images = images.to(args.device) #images.shape: torch.Size([batch_size, 3 rgb channels, image_height, image_width])
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                image_features, _ = model.encode_image(images) #batch_size x emb_size
                image_features = F.normalize(image_features, dim=-1)
                feature_list.append(image_features.detach().cpu().numpy())

        file = save_pickle(
            output_dir,
            [
                np.vstack(feature_list),
                np.hstack(target_list),
                dataloader.dataset.samples,
                dataloader.dataset.class_to_idx,
            ],
        )

    return file



def few_shot_eval(model, data, args, output_dir):
    results = {}

    logging.info("Starting few-shot.")

    for split in data:
        logging.info("Building few-shot %s classifier.", split)
        
        file = run(model, data[split], args, output_dir)
        
        logging.info("Finished few-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished few-shot.")

    return results, file

def split(select, kshot, nfold, i2c, filepath=None):    
    
    test = [] #N
    target = [] #N
    train = [] #kshot x class
    label = []
    test_sample = []
    random.seed(nfold)
    for k,v in select.items():
        num_v = len(v['feature'])
        if num_v < kshot:
            logging.info(f'{i2c[k]} has only {num_v} images. Less than {kshot} images for few-shot. ')
        elif num_v < kshot+5:
            logging.info(f'{i2c[k]} has only {num_v} images. Not enough for evaluation.')
            
        random.shuffle(v['feature'])
        train.append(v['feature'][:kshot])
        test+=v['feature'][kshot:]
        test_sample+=v['sample'][kshot:]
        label+=[k for i in range(kshot)]
        test_num = num_v-kshot
        target+=[k for i in range(test_num)]
    
    flatten_train = np.vstack(train)
    label = np.array(label)
    assert kshot*len(train) == flatten_train.shape[0] == label.shape[0]
    assert len(train) == n_class
    assert len(test) == len(target) == len(test_sample)

    return flatten_train, label, test, target

def CL2N(x_flatten, x_mean):
    x_flatten = x_flatten - x_mean #(class, emb) = (class, emb) - (emb,)
    x_flatten = x_flatten / LA.norm(x_flatten, 2, 1)[:, None] #(class, emb) = (class, emb) / (class,1)
    return x_flatten

def get_acc(flatten_train, label, test, target, n_class, kshot, nfold):
    train_mean = flatten_train.mean(axis=0)
    
    flatten_train = CL2N(flatten_train,train_mean)
    test = CL2N(test,train_mean)
    train_center = flatten_train.reshape(n_class, kshot, flatten_train.shape[-1]).mean(1)

    num_NN = 1
    label = label[::kshot] #num of class
    subtract = train_center[:, None, :] - test
    distance = LA.norm(subtract, 2, axis=-1) #(train num:test num)
    idx = np.argpartition(distance, num_NN, axis=0)[:num_NN] #(num_NN:train num)
    nearest_samples = np.take(label, idx) #(num_NN:train num)
    out = mode(nearest_samples, axis=0, keepdims=True)[0]
    out = out.astype(int)
    test_label = np.array(target)
    acc = (out == test_label).mean()

    return acc

if __name__ == "__main__":
    global args
    args = parse_args(sys.argv[1:])
    random_seed(args.seed, 0)

    configure_torch_backends(deterministic=True)

    device = init_device(args)

    log_base_path = configure_logging(
        args, "few_shot", include_workers=False, log_filename="out.log"
    )

    normalize_force_image_size(args)

    log_params(args, log_base_path)

    if args.task_type == 'eval':
        feature_file = args.pretrained        
    else:
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

        if args.trace:
            model = trace_model(model, batch_size=args.batch_size, device=device)

        logging.info("Model:")
        logging.info(f"{str(model)}")

        # initialize datasets
        data = {
            "val-unseen": get_dataloader(
                DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val),
                batch_size=args.batch_size,num_workers=args.workers
            ),
        }

        start_time = time.monotonic()
                   
        model.eval()
        output_dir = log_base_path or os.getcwd()
        _, feature_file = few_shot_eval(model, data, args, output_dir)             

        end_time = time.monotonic()
        logging.info(f"feature extraction takes: {datetime.timedelta(seconds=end_time - start_time)}")

    if args.task_type == 'eval' or args.task_type == 'all':
        feature, target, samples, c2i = load_pickle(feature_file)

        i2c = dict([(v,k) for k,v in c2i.items()])

        if args.debug:
            for i in range(len(target)):
                assert target[i] == samples[i][1]


        select = dict()
        for idx in range(len(feature)):
            f = feature[idx]
            s = samples[idx]
            cat = target[idx]
            if cat in select:
                select[cat]['feature'].append(f)
                select[cat]['sample'].append(s)
            else:
                select[cat] = dict()
                select[cat]['feature'] = [f]
                select[cat]['sample'] = [s]
                
        count = sum([len(v['feature']) for v in select.values()])
        n_class = len(select)

        logging.info("Num of classes: %d.\nNum of samples: %d.", n_class, count)

        for kshot in args.kshot_list:
           
            acc_list = []
            for n in range(args.nfold):
                #split
                flatten_train, label, test, target = split(select, kshot, n, i2c)
                acc = get_acc(flatten_train, label, test, target, n_class, kshot, n)
                acc_list.append(acc)
                logging.info(f"{kshot} shot No.{n} ACC: {acc:.4f}")
            logging.info("!!!!!!Result: ")
            logging.info("Dataset:  %s", args.data_root)
            logging.info("Model:  %s", args.pretrained)
            logging.info(f"{kshot} shot AVG ACC: {np.mean(acc_list)*100:.2f}")
            logging.info(f"{kshot} shot STD: {np.std(acc_list)*100:.4f}")
    
    logging.info(f"Done!")
