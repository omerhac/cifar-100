import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def load_train_dataset():
    "Return CIFAR-100 train dataset"
    return tfds.load('cifar100', split='train', as_supervised=True, download=False, data_dir='data')


def load_test_dataset():
    "Return CIFAR-100 test dataset"
    return tfds.load('cifar100', split='test', download=False, as_supervised=True, data_dir='data')


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


def random_rotate(image, label):
    """Randomly rotate the image in 35 deg range"""
    def rotate_image(image):
        angle = np.random.uniform() * 35  # random angle from [0, 35]
        return rotate(image.numpy(), angle, reshape=False)

    rotated = tf.py_function(rotate_image, [image], Tout=tf.float32)
    rotated.set_shape([32, 32, 3])

    return rotated, label


def random_flip(image, label):
    """Randomly flip the image left to right"""
    flip = np.random.uniform()  # flip probability
    if flip > 0.5:
        return tf.image.flip_left_right(image), label
    else:
        return image, label


def random_mask(image, label):
    """Put a random black mask of size up to third image size on the image"""
    def mask(x):
        mask_size = int(np.floor(np.random.uniform(high=0.33) * 32))
        mask_hight = int(np.floor(np.random.uniform() * (32 - mask_size)))
        mask_width = int(np.floor(np.random.uniform() * (32 - mask_size)))

        # mask
        x = x.numpy()
        x[mask_hight:mask_hight + mask_size, mask_width:mask_width + mask_size, :] = 0
        return x

    masked = tf.py_function(mask, [image], Tout=tf.float32)
    masked.set_shape([32, 32, 3])

    return masked, label


def compute_black_percent(image, label):
    """Compute the percentage of the image that is black"""
    black_pix = tf.cast(image == 0, tf.float32)
    black_sum = tf.reduce_sum(black_pix)
    return image, label, black_sum / (32 * 32 * 3)


def get_train_dataset(with_mask_percent=False):
    """Return train dataset after manipulations"""
    train_dataset = load_train_dataset()
    prep_train = train_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # normalize
    prep_train = prep_train.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # rotate
    prep_train = prep_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # flip
    prep_train = prep_train.map(random_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # mask

    if with_mask_percent:
        prep_train = prep_train.map(compute_black_percent, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # masked percent
    return prep_train


def get_test_dataset():
    """Return test dataset after manipulations"""
    test_dataset = load_test_dataset()
    prep_test = test_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return prep_test


if __name__=='__main__':
    ds = get_train_dataset()
    plt.figure()
    fig, ax = plt.subplots(1, 3)
    for i, (image, label, bp) in enumerate(ds.take(3)):
        ax[i].imshow(image)
        ax[i].set_title(bp.numpy())
        #print(image.numpy().shape)
    plt.show()



