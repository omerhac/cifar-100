import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Input, Flatten, concatenate,\
    GlobalAveragePooling2D, Dropout


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
    """Squeezenet like model as in https://arxiv.org/pdf/1602.07360.pdf, with residual connections"""
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
        fire2 = fire1 + self._fire2(fire1)  # residual connection
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
    def __init__(self, filters1, filters3_red, filters3, filters5_red, filters5, filters_proj, name='inception'):
        super(InceptionModule, self).__init__(name=name)
        self._num_filters1 = filters1
        self._num_filters3_red = filters3_red
        self._num_filters5_red = filters5_red
        self.num_filters5 = filters5
        self._num_filters_proj = filters_proj

        self._conv1 = Conv2D(filters1, (1, 1), activation='relu', padding='same', name='1x1_conv')
        self._conv3_red = Conv2D(filters3_red, (1, 1), activation='relu', padding='same', name='3x3_reduce_conv')
        self._conv5_red = Conv2D(filters5_red, (1, 1), activation='relu', padding='same', name='5x5_reduce_conv')
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


class InceptionBNModule(tf.keras.layers.Layer):
    """Inception style module with batch normalization"""
    def __init__(self, filters1, filters3_red, filters3, filters5_red, filters5, filters_proj, bn_moment=0.9, name='inception'):
        super(InceptionBNModule, self).__init__(name=name)
        self._num_filters1 = filters1
        self._num_filters3_red = filters3_red
        self._num_filters5_red = filters5_red
        self.num_filters5 = filters5
        self._num_filters_proj = filters_proj

        self._conv1 = Conv2D(filters1, (1, 1), activation='relu', padding='same', name='1x1_conv')
        self._conv3_red = Conv2D(filters3_red, (1, 1), activation='relu', padding='same', name='3x3_reduce_conv')
        self._conv5_red = Conv2D(filters5_red, (1, 1), activation='relu', padding='same', name='5x5_reduce_conv')
        self._conv3 = Conv2D(filters3, (3, 3), activation='relu', padding='same', name='3x3_conv')
        self._conv5 = Conv2D(filters5, (1, 1), activation='relu', padding='same', name='5x5_conv')
        self._maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='3x3_maxpool')
        self._project = Conv2D(filters_proj, (1, 1), activation='relu', padding='same', name='projection')

        self._bn_conv1 = BatchNormalization(momentum=bn_moment, name='conv1_batch_norm')
        self._bn_conv3_red = BatchNormalization(momentum=bn_moment, name='conv3_red_batch_norm')
        self._bn_conv5_red = BatchNormalization(momentum=bn_moment, name='conv5_red_batch_norm')
        self._bn_conv3 = BatchNormalization(momentum=bn_moment, name='conv3_batch_norm')
        self._bn_conv5 = BatchNormalization(momentum=bn_moment, name='conv5_batch_norm')
        self._bn_mp = BatchNormalization(momentum=bn_moment, name='maxpool')
        self._bn_proj = BatchNormalization(momentum=bn_moment, name='project_batch_norm')

    def call(self, x, training=True):
        """Forward pass on input x"""
        # perform 1x1 convolutions per all branches for dimensionality reduction
        conv1 = self._bn_conv1(self._conv1(x), training=training)
        conv3_red = self._bn_conv3_red(self._conv3_red(x), training=training)
        conv5_red = self._bn_conv5_red(self._conv5_red(x), training=training)

        # perform branches "expensive" convolutions on reduced maps
        conv3 = self._bn_conv3(self._conv3(conv3_red), training=training)
        conv5 = self._bn_conv5(self._conv5(conv5_red), training=training)

        # maxpooling branch
        maxpool = self._bn_mp(self._maxpool(x), training=training)
        maxpool_proj = self._bn_proj(self._project(maxpool), training=training)

        return concatenate([conv1, conv3, conv5, maxpool_proj])


class Inception(tf.keras.Model):
    """Inception style net as in https://arxiv.org/pdf/1409.4842v1.pdf"""
    def __init__(self):
        super(Inception, self).__init__()

        # root
        self._root_conv = Conv2D(64, (5, 5), input_shape=[32, 32, 3], padding='valid', activation='relu',
                                 name='root_conv')

        # inception modules
        # block 1
        self._inception1 = InceptionModule(64, 96, 128, 16, 32, 32, name='inception1')
        self._inception2 = InceptionModule(128, 128, 192, 32, 96, 34, name='inception2')
        self._mp1 = MaxPooling2D((3,3), strides=(2,2), name='maxpool1')

        # block 2
        self._inception3 = InceptionModule(192, 96, 208, 16, 48, 64, name='inception3')
        self._inception4 = InceptionModule(60, 112, 224, 24, 64, 64, name='inception4')
        self._mp2 = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool2')

        # top
        self._avg_pool = GlobalAveragePooling2D(name='top_avg_pool')
        self._dropout = Dropout(0.4)
        self._output = Dense(100, activation='softmax', name='output_layer')

    def call(self, x):
        "Forward pass on input x"
        root_x = self._root_conv(x)

        inception1 = self._inception1(root_x)
        inception2 = self._inception2(inception1)
        mp1 = self._mp1(inception2)

        inception3 = self._inception3(mp1)
        inception4 = self._inception4(inception3)
        mp2 = self._mp2(inception4)

        output = self._output(self._dropout(self._avg_pool(mp2)))

        return output


class InceptionBN(tf.keras.Model):
    """Inception style net with batch normalization"""

    def __init__(self, predict_mask=False):
        super(InceptionBN, self).__init__()

        # root
        self._root_conv = Conv2D(64, (5, 5), input_shape=[32, 32, 3], padding='valid', activation='relu',
                                 name='root_conv')
        self._bn_root = BatchNormalization(momentum=0.9, name='bn_root')

        # inception modules
        # block 1
        self._inception1 = InceptionBNModule(64, 96, 128, 16, 32, 32, name='inception1')
        self._inception2 = InceptionBNModule(128, 128, 192, 32, 96, 34, name='inception2')
        self._mp1 = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool1')

        # block 2
        self._inception3 = InceptionBNModule(192, 96, 208, 16, 48, 64, name='inception3')
        self._inception4 = InceptionBNModule(160, 112, 224, 24, 64, 64, name='inception4')
        self._mp2 = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool2')

        # top
        self._avg_pool = GlobalAveragePooling2D(name='top_avg_pool')
        self._dropout = Dropout(0.4)

        if predict_mask:
            self._output = Dense(101, activation='softmax', name='output_layer')
        else:
            self._output = Dense(100, activation='softmax', name='output_layer')

    def call(self, x, training=True):
        "Forward pass on input x"
        root_x = self._bn_root(self._root_conv(x), training=training)

        inception1 = self._inception1(root_x, training=training)
        inception2 = self._inception2(inception1, training=training)
        mp1 = self._mp1(inception2)

        inception3 = self._inception3(mp1, training=training)
        inception4 = self._inception4(inception3, training=training)
        mp2 = self._mp2(inception4)

        output = self._output(self._dropout(self._avg_pool(mp2)))

        return output


class LossFunction(tf.keras.losses.Loss):
    """Loss function for models that predicts CIFAR-100 as well as pic mask percent
    Loss = alpha * sparse_categorical_crossentropy + beta * MSE(over mask prediction)
    """

    def __init__(self, alpha=1, beta=4):
        super(LossFunction, self).__init__()
        self._alpha = alpha
        self._beta = beta

    def call(self, y_true, y_pred):
        gt_class, gt_mask = y_true
        predicted_mask = y_pred[:, -1]  # last dimension predicts mask
        predicted_classes = y_pred[:, :-1]  # all other dimensions predicts classes

        # compute losses
        sparse_ce = tf.keras.losses.sparse_categorical_crossentropy(gt_class, predicted_classes)
        mask_mse = tf.keras.losses.mse(gt_mask, [predicted_mask])

        return self._alpha * sparse_ce + self._beta * mask_mse


if __name__ == '__main__':
    a = tf.random.uniform([2,3])
    c = tf.constant([1, 1])
    m = tf.constant([1, 2])

    l = LossFunction(beta=4)
    print(l([c, m], a))
