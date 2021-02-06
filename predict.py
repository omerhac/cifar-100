import models
import etl
import tensorflow as tf


def predict(image, model=None, load_dir='weights/InceptionBN_aug', string_label=True):
    """Predict the label of image with model
    Args:
        image: jpeg image to predict as numpy array (not normalized)
        model: model to predict with
        load_dir: model path to load from if model=None
        string_label: whether to return the actual label or just int label

    Return:
        image_label -> int / string
    """

    if not model:
        model = tf.keras.models.load_model(load_dir)

    image = image / 255  # normalize

    # predict
    int_label = tf.argmax(model([image]), axis=1)

    if string_label:
        return etl.get_label_name(int_label.numpy()[0])
    else:
        return int_label.numpy()[0]


def build_confusion_metrix(model_path, n_examples):
    """Build confusion matrix
    Args:
        model_path: path to saved model
        n_examples: number of examples to base on
    Return:
        confusion_matrix -> numpy array
    """

    model = tf.keras.models.load_model(model_path)

    ds = etl.get_test_dataset().batch(10)

    # predict
    sample_labels, sample_images = next(ds)
    pred_labels = tf.argmax(model(sample_images), axis=1)


if __name__ == '__main__':
    a = models.InceptionBN()
    ds = etl.load_train_dataset()
    a.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    #a.fit(ds, epochs=1, steps_per_epoch=1)
    #a.save('check')
    for image, label in ds.take(1):
        i = image.numpy()
    print(i)
    print(predict(i, load_dir='weights/check'))