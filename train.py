import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import models
import etl
import os
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy


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

    test_generator = etl.get_test_dataset().batch(batch_size)

    # train and log
    loss_function = models.LossFunction(beta=20)
    optimizer = tf.optimizers.Adam()

    # initialize aggregators
    mean_loss = tf.keras.metrics.Mean(name='mean_loss')
    mean_categorical_accuracy = tf.keras.metrics.Mean(name='mean_categorical_accuracy')
    mean_top_k = tf.metrics.Mean(name='mean_top_k_accuracy')
    mean_val_cat_acc = tf.metrics.Mean(name='mean_val_categorical_accuracy')
    mean_val_top_k = tf.metrics.Mean(name='mean_val_top_k_accuracy')
    losses, cat_accs, top_ks, val_accs, val_topks = [], [], [], [], []

    @tf.function
    def train_step(batch_images, batch_labels, batch_percent):
        with tf.GradientTape() as tape:
            batch_preds = model(batch_images)
            loss = loss_function([batch_labels, batch_percent], batch_preds)  # compute loss

            # apply gradients
            grads_and_vars = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads_and_vars, model.trainable_variables))

        # calculate metrics
        cat_acc = tf.metrics.sparse_categorical_accuracy(batch_labels, batch_preds)
        topk_acc = tf.metrics.sparse_top_k_categorical_accuracy(batch_labels, batch_preds)
        return loss, cat_acc, topk_acc

    for epoch in range(epochs):
        print(f'Epoch number {epoch}')

        # train
        for batch_num, (batch_images, batch_labels, batch_percent) in enumerate(prep_train):
            # calculate and apply gradients
            loss, cat_acc, top_k_acc = train_step(batch_images, batch_labels, batch_percent)

            # aggregate
            mean_loss(loss)
            mean_categorical_accuracy(cat_acc)
            mean_top_k(top_k_acc)

            # print loss and accuracy
            if batch_num % 50 == 0:
                print(f'Batch {batch_num}/{50000 // batch_size - 1}, Mean loss is: {mean_loss.result():.4f}, '
                      f'Mean Categorical Accuracy is: {mean_categorical_accuracy.result():.4f},'
                      f'Mean Top K Accuracy is: {mean_top_k.result():.4f}')

            # finished dataset break rule
            if batch_num == 50000 // batch_size - 1:
                break

        # evaluate
        for batch_num, (batch_images, batch_labels, batch_percent) in enumerate(test_generator):
            batch_preds = model(batch_images)
            # aggregate metric
            val_cat_acc = tf.metrics.sparse_categorical_accuracy(batch_labels, batch_preds)
            val_top_k = tf.metrics.sparse_top_k_categorical_accuracy(batch_labels, batch_preds)

            mean_val_cat_acc(val_cat_acc)
            mean_val_top_k(val_top_k)

        print(f'Validation Categorical Accuracy score is: {mean_val_cat_acc.result():.4f},'
              f'Validation Top K Accuracy is: {mean_val_top_k.result():.4f}')

        # aggregate and reset
        cat_accs.append(mean_categorical_accuracy.result())
        losses.append(mean_loss.result())
        top_ks.append(mean_top_k.result())
        val_accs.append(mean_val_top_k.result())
        val_topks.append(mean_val_top_k.result())
        mean_categorical_accuracy.reset_states()
        mean_loss.reset_states()
        mean_top_k.reset_states()
        mean_val_top_k.reset_states()
        mean_val_cat_acc.reset_states()

    # save training logs
    if log_name:
        if not os.path.exists(log_dir): os.mkdir(log_dir)

        # loss log
        plot_log(hist, epochs, ['loss', 'val_loss'], save_path=log_dir+'/loss_log_'+log_name)

        # accuracy log
        plot_log(hist, epochs, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy'],
                 save_path=log_dir+'/acc_log_'+log_name)

        # top k accuracy log
        plot_log(hist, epochs, ['sparse_top_k_categorical_accuracy', 'val_sparse_top_k_categorical_accuracy'],
                 save_path=log_dir+'/topk_acc_log_'+log_name)

    # save weights
    if save_path:
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        model.save(save_path, save_format='tf')


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
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


if __name__ == '__main__':
    ds = etl.get_train_dataset(with_mask_percent=True)
    model = models.get_simple_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=[SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy(k=5)])
    train(model, ds, epochs=70, log_name='InceptionBN_augment.jpeg', save_path='weights/InceptionBN_eugment.tf')
