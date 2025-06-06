"""
Sourced from [generalized-category-discovery](https://github.com/sgvaze/generalized-category-discovery/blob/main/data/herbarium_19.py).
[Generalized Category Discovery](https://www.robots.ox.ac.uk/~vgg/research/gcd/)
Given a dataset, some of which is labelled, *Generalized Category Discovery* is the task of assigning a category to all the unlabelled instances. Unlabelled instances could come from labelled or 'New' classes.
Requested ciation:
```
@InProceedings{vaze2022gcd,
               title={Generalized Category Discovery},
               author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
               booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
               year={2022}}
```
"""
import os.path

import torch.utils
import torch.utils.data

import numpy as np
import torch
import torchvision.datasets
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as linear_assignment


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    assert y_pred.shape == y_true.shape
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    y_pred_new = np.array([ind[i][1] for i in y_pred])

    return y_pred_new == y_true


class HerbariumDataset19(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx


def subsample_instances(dataset, prop_indices_to_subsample=0.8, seed=0):
    np.random.seed(seed)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))
    return subsample_indices


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_herbarium_datasets(herbarium_dataroot, train_transform, test_transform, train_classes=range(500), prop_train_labels=0.8,
                           seed=0, split_train_val=False):

    np.random.seed(seed)

    # Init entire training set
    train_dataset = HerbariumDataset19(transform=train_transform,
                                       root=os.path.join(herbarium_dataroot, 'small-train'))

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    # Get unlabelled data
    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    # Get test dataset
    test_dataset = HerbariumDataset19(transform=test_transform,
                                      root=os.path.join(herbarium_dataroot, 'small-validation'))

    # Transform dict
    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


class MergedDataset(torch.utils.data.Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return img, label, uq_idx, labeled_or_not

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
