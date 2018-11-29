'''
Utilities for DeepLab

Lei Mao
Department of Computer Science
University of Chicago

dukeleimao@gmail.com
'''

import tensorflow as tf


def atrous_spatial_pyramid_pooling(inputs, filters=256, regularizer=None):
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''

    # Atrous 1x1
    aspp1x1 = tf.layers.conv2d(inputs, filters, (1, 1), padding='same', kernel_regularizer=regularizer, name='aspp1x1')
    # Atrous 3x3, rate = 6
    aspp3x3_1 = tf.layers.conv2d(inputs, filters, (3, 3), padding='same', dilation_rate=(6, 6), kernel_regularizer=regularizer, name='aspp3x3_1')
    # Atrous 3x3, rate = 12
    aspp3x3_2 = tf.layers.conv2d(inputs, filters, (3, 3), padding='same', dilation_rate=(12, 12), kernel_regularizer=regularizer, name='aspp3x3_2')
    # Atrous 3x3, rate = 18
    aspp3x3_3 = tf.layers.conv2d(inputs, filters, (3, 3), padding='same', dilation_rate=(18, 18), kernel_regularizer=regularizer, name='aspp3x3_3')

    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    image_feature = tf.layers.conv2d(image_feature, filters, (1, 1), padding='same')
    image_feature = tf.image.resize_bilinear(image_feature, tf.shape(inputs)[1:3], align_corners=True, name='image_pool_feature')

    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature], axis=3, name='aspp_pools')
    outputs = tf.layers.conv2d(outputs, filters, (1, 1), padding='same', kernel_regularizer=regularizer, name='aspp_outputs')

    return outputs
