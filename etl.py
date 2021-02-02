import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


def load_train_dataset():
    "Return CIFAR-100 train dataset"
    return tfds.load('cifar100', split='train', as_supervised=True, download=False, data_dir='data')


def load_test_dataset():
    "Return CIFAR-100 test dataset"
    return tfds.load('cifar100', split='train', download=False, as_supervised=True, data_dir='data')


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


def normalize_image(image, label):
    """Normalize jpg image"""
    return image / 255, label


def get_train_dataset():
    """Return train dataset after manipulations"""
    train_dataset = load_train_dataset()
    prep_train = train_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return prep_train


def get_test_dataset():
    """Return test dataset after manipulations"""
    test_dataset = load_test_dataset()
    prep_test = test_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return prep_test


if __name__=='__main__':
    ds = load_train_dataset()
    for image, label in ds.take(1):
        print(label.numpy())
        print(image.numpy())


    ds = get_train_dataset()
    for image, label in ds.take(1):
        print(label.numpy())
        print(image.numpy())