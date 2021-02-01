import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import model
import etl
import os


def train(model, train_generator, batch_size=64, epochs=50, log_dir='logs', save_dir='weights', save_path=None, seed=42,
          log_name=None):
    """Train the provided model on the provided train dataset generator.
    Args:
        model: tf keras compiled Model to train.
        train_generator: train dataset generator
        batch_size: train batch size, defaults to 64
        epochs: number of epochs, defaults to 50
        log_dir: directory for saving train logging
        save_dir: directory for saving model weights
        save_path: path for saving model .h5 weights. defaults to None for non saving session
        seed: train shuffling seed
        log_name: train logging name, defaults to None for non logging session
    """

    # prepare dataset
    prep_train = train_generator.shuffle(seed)
    prep_train = prep_train.repeat()
    prep_train = prep_train.batch(batch_size)

    test_generator = etl.load_test_dataset().batch(batch_size)

    # train and log
    hist = model.fit(prep_train, epochs=epochs, validation_data=test_generator, steps_per_epoch=50000//batch_size)
    hist = hist.history

    # save training logs
    if log_name:
        if not os.path.exists(log_dir): os.mkdir(log_dir)

        # loss log
        plot_log(hist, epochs, ['loss', 'val_loss'], save_path=log_dir+'/loss_log_'+log_name)

        # accuracy log
        plot_log(hist, epochs, ['sparse_categorical_accuracy', 'sparse_categorical_accuracy'],
                 save_path=log_dir+'/acc_log_'+log_name)

        # top k accuracy log
        plot_log(hist, epochs, ['sparse_top_k_categorical_accuracy', 'val_sparse_top_k_categorical_accuracy'],
                 save_path=log_dir+'/acc_log_'+log_name)

    # save weights
    if save_path:
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        model.save(save_path, save_format='.h5')


def plot_log(hist_dict, epochs, val_names, save_path='log.jpg'):
    """Plot training logs and save the figure.
    Args:
        hist_dict: training dictionary
        epochs: number of training epochs
        val_names: list of names of training values to plot
        save_path: path to save logs, optional
    """
    epochs_range = range(epochs)

    plt.figure()
    for val in val_names:
        plt.plot(epochs_range, hist_dict[val], label=val)

    plt.xlabel('epochs')
    plt.ylabel(val_names[0])

    plt.savefig(save_path)


if __name__ == '__main__':
    model = tf.keras.Sequential([tf.keras.layers.Input([32, 32, 3]),
                                 tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
                                 tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(100, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                                                                                     tf.keras.metrics.SparseTopKCategoricalAccuracy(5)])
    ds = etl.load_train_dataset()
    train(model, ds, epochs=2, log_name='bla.jpeg', save_path='weights/bla.h5')