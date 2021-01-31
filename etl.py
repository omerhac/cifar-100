import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


def load_train_dataset():
    "Return CIFAR-100 train dataset"
    return tfds.load('cifar100', split='train', as_supervised=True)


def load_test_dataset():
    "Return CIFAR-100 test dataset"
    return tfds.load('cifar100', split='train', download=False, as_supervised=True)


def get_lables():
    """Return CIFAR-100 labels tfds object"""
    _, info = tfds.load('cifar100', with_info=True, download=False)
    return info.features['label']


def get_coarse_lables():
    """Return CIFAR-100 labels tfds object"""
    _, info = tfds.load('cifar100', with_info=True, download=False)
    return info.features['coarse_label']


def get_label_name(label):
    """Return an int label string name"""
    labels = get_lables()
    return labels.int2str(label)


def get_coarse_label_name(label):
    """Return a coarse int label string name"""
    labels = get_coarse_lables()
    return labels.int2str(label)


if __name__=='__main__':
    print(get_label_name(5))
    print(get_coarse_label_name(5))