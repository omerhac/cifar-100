import tensorflow as tf
import numpy as np
import etl
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Input, Flatten, concatenate,\
    GlobalAveragePooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy


def get_simple_model():
    """Create and return a simple CNN model for benchmark"""
    # root
    x = Input(shape=[32, 32, 3], name='input')
    c1 = Conv2D(32, (3,3), padding='same', activation='relu', name='conv1')(x)

    # first block
    mp1 = MaxPooling2D((2, 2), padding='valid', name='max_pool1')(c1)
    c2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(mp1)
    c3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3')(c2)

    # second block
    mp2 = MaxPooling2D((2, 2), padding='valid', name='max_pool2')(c3)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu',name='conv4')(mp2)
    c5 = Conv2D(128, (3, 3), padding='same', activation='relu',name='conv5')(c4)

    # third block
    mp3 = MaxPooling2D((2, 2), padding='valid', name='max_pool3')(c5)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv6')(mp3)
    c7 = Conv2D(256, (3, 3), padding='same', activation='relu',name='conv7')(c6)

    # dense
    flat = Flatten(name='flatten')(c7)
    d1 = Dense(1028, activation='relu', name='dense_1')(flat)

    output = Dense(100, activation='softmax', name='output_layer')(d1)  # num-classes

    # compile
    model = tf.keras.Model(x, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=[SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy(k=5)])

    return model


class FireModule(tf.keras.layers.Layer):
    """Squeezenet like fire module as in https://arxiv.org/pdf/1602.07360.pdf"""
    def __init__(self, squeeze_filters, expand_filters, name='fire'):
        super(FireModule, self).__init__(name=name)
        self._num_squeeze_filters = squeeze_filters
        self._num_expand_filters = expand_filters

        # create layers
        self._squeeze = Conv2D(squeeze_filters, (1,1), activation='relu', padding='same', name='squeeze_filters')
        self._expand1 = Conv2D(expand_filters, (3,3), activation='relu', padding='same', name='expand_1x1_filters')
        self._expand3 = Conv2D(expand_filters, (3, 3), activation='relu', padding='same', name='expand_3x3_filters')

    def call(self, x):
        """Forward pass on input x"""
        squeeze = self._squeeze(x)
        expand1 = self._expand1(squeeze)
        expand3 = self._expand3(squeeze)
        return concatenate([expand3, expand1])


class Squeezenet(tf.keras.Model):
    """Squeezenet like model as is https://arxiv.org/pdf/1602.07360.pdf"""
    def __init__(self):
        super(Squeezenet, self).__init__()

        # root
        self._root_conv = Conv2D(96, (5,5), input_shape=[32, 32, 3], padding='valid', activation='relu', name='root_conv')

        # fire modules
        # block 1
        self._fire1 = FireModule(16, 32, name='fire1')
        self._fire2 = FireModule(16, 32, name='fire2')
        self._fire3 = FireModule(32, 128, name='fire3')
        self._mp1 = MaxPooling2D((3,3), strides=(2,2), name='maxpool1')

        # block 2
        self._fire4 = FireModule(32, 128, name='fire4')
        self._fire5 = FireModule(32, 128, name='fire5')
        self._fire6 = FireModule(64, 256, name='fire6')
        self._mp2 = MaxPooling2D((3, 3), strides=(2,2), name='maxpool2')

        # top
        self._top_conv = Conv2D(1000, (1, 1), padding='same', activation='relu', name='top_conv')
        self._avg_pool = GlobalAveragePooling2D(name='top_avg_pool')
        self._output = Dense(100, activation='softmax', name='output_layer')

    def call(self, x):
        """Forward pass on input x"""
        root_x = self._root_conv(x)

        fire1 = self._fire1(root_x)
        fire2 = fire1 + self._fire2(fire1) # residual connection
        fire3 = self._fire3(fire2)
        mp1 = self._mp1(fire3)

        fire4 = self._fire4(mp1)
        fire5 = fire4 + self._fire5(fire4)  # residual connection
        fire6 = self._fire6(fire5)
        mp2 = self._mp2(fire6)

        output = self._output(self._avg_pool(self._top_conv(mp2)))

        return output


class InceptionModule(tf.keras.layers.Layer):
    """Inception style module as in https://arxiv.org/pdf/1409.4842v1.pdf"""
    def __init__(self, filters1, filters3_red, filters3, filters5_red, filters5, filters_proj):
        super(InceptionModule, self).__init__()
        self._num_filters1 = filters1
        self._num_filters3_red = filters3_red
        self._num_filters5_red = filters5_red
        self.num_filters5 = filters5
        self._num_filters_proj = filters_proj

        self._conv1 = Conv2D(filters1, (1, 1), activation='relu', padding='same', name='1x1_conv')
        self._conv3_red = Conv2D(filters3_red, (1, 1), activation='relu', padding='same',name='3x3_reduce_conv')
        self._conv5_red = Conv2D(filters5_red, (1, 1), activation='relu', padding='same',name='5x5_reduce_conv')
        self._conv3 = Conv2D(filters3, (3, 3), activation='relu', padding='same', name='3x3_conv')
        self._conv5 = Conv2D(filters5, (1, 1), activation='relu', padding='same', name='5x5_conv')
        self._maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='3x3_maxpool')
        self._project = Conv2D(filters_proj, (1, 1), activation='relu', padding='same', name='projection')

    def call(self, x):
        """Forward pass on input x"""
        # perform 1x1 convolutions per all branches for dimensionality reduction
        conv1 = self._conv1(x)
        conv3_red = self._conv3_red(x)
        conv5_red = self._conv5_red(x)

        # perform branches "expensive" convolutions on reduced maps
        conv3 = self._conv3(conv3_red)
        conv5 = self._conv5(conv5_red)

        # maxpooling branch
        maxpool = self._maxpool(x)
        maxpool_proj = self._project(maxpool)

        return concatenate([conv1, conv3, conv5, maxpool_proj])



if __name__ == '__main__':
    model = Squeezenet()
    x = tf.random.uniform([1, 32, 32, 3])
    print(model(x).shape)
    model.build((None, 32, 32, 3))
    #print(model.summary())
    l = InceptionModule(64, 96, 128, 16, 32, 32)
    x = tf.random.uniform([1, 28, 28, 192])
    print(l(x).shape)
