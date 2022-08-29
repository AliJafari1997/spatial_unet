import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from keras.regularizers import l2
weight_decay=5e-4


def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
            config = super(PAM, self).get_config().copy()
            config.update({
                'ratio': self.ratio,
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            })
            return config



    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input

        out = Conv2D(filters*2, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(out)
        out = BatchNormalization(axis=3)(out)
        out = Activation('relu')(out)
        return out
