import tensorflow as tf
import numpy as np
import etl
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, Flatten


def get_simple_model():
    """Create and return a simple CNN model for benchmark"""
    x = Input(shape=[32, 32, 3], name='input')

    c1 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv1')(x)
    c2 = Conv2D(64, (3,3), padding='same', activation='relu',name='conv2')(c1)

    mp1 = MaxPooling2D((2, 2), padding='valid', name='max_pool1')(c2)
    c3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(mp1)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4')(c3)

    mp2 = MaxPooling2D((2, 2), padding='valid', name='max_pool2')(c4)
    c5 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv5')(mp2)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv6')(c5)
    c7 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv7')(c6)
    c8 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv8')(c7)

    mp3 = MaxPooling2D((2, 2), padding='valid', name='max_pool3')(c8)
    c9 = Conv2D(512, (3, 3), padding='same', activation='relu',name='conv9')(mp3)
    c10 = Conv2D(512, (3, 3), padding='same', activation='relu',name='conv10')(c9)
    c11 = Conv2D(512, (3, 3), padding='same', activation='relu',name='conv11')(c10)
    c12 = Conv2D(512, (3, 3), padding='same', activation='relu',name='conv12')(c11)

    flat = Flatten(name='flatten')(c12)

    d1 = Dense(4096, activation='relu', name='dense_1')(flat)
    do1 = Dropout(0.2, name='dropout_1')(d1)
    d2 = Dense(4096, activation='relu', name='dense_2')(do1)
    do2 = Dropout(0.2, name='dropout_2')(d2)

    output = Dense(100, activation='softmax', name='output_layer')(do2)

    model = tf.keras.Model(x, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    return model


if __name__ == '__main__':
    model = get_simple_model()

    ds = etl.load_train_dataset().batch(100)
    model.fit(ds, epochs=4)
