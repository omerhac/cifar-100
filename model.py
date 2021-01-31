import tensorflow as tf
import numpy as np
import etl
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy


def get_simple_model():
    """Create and return a simple CNN model for benchmark"""
    x = Input(shape=[32, 32, 3], name='input')

    c1 = Conv2D(32, (3,3), padding='same', activation='relu', name='conv1')(x)

    mp1 = MaxPooling2D((2, 2), padding='valid', name='max_pool1')(c1)
    c2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(mp1)
    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(c2)

    mp2 = MaxPooling2D((2, 2), padding='valid', name='max_pool2')(c3)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu',name='conv4')(mp2)
    c5 = Conv2D(128, (3, 3), padding='same', activation='relu',name='conv5')(c4)

    mp3 = MaxPooling2D((2, 2), padding='valid', name='max_pool3')(c5)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv6')(mp3)
    c7 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv7')(c6)

    flat = Flatten(name='flatten')(c7)

    d1 = Dense(1028, activation='relu', name='dense_1')(flat)

    output = Dense(100, activation='softmax', name='output_layer')(d1)

    model = tf.keras.Model(x, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=[SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy(k=5)])

    return model


if __name__ == '__main__':
    model = get_simple_model()

    ds = etl.load_train_dataset().batch(100)
    model.fit(ds, epochs=4)
