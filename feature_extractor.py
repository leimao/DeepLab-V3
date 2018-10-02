import tensorflow as tf
from nets import resnet_v2
from nets.mobilenet import mobilenet_v2


def Vgg16(imgs_in, weight_decay, batch_norm_momentum):
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    conv1_1 = tf.layers.conv2d(
        imgs_in,
        64,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv1_1',
    )
    conv1_2 = tf.layers.conv2d(
        conv1_1,
        64,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv1_2',
    )
    pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=[2, 2], name='pool1')

    conv2_1 = tf.layers.conv2d(
        pool1,
        128,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv2_1',
    )
    conv2_2 = tf.layers.conv2d(
        conv2_1,
        128,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv2_2',
    )
    pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=[2, 2], strides=[2, 2], name='pool2')

    conv3_1 = tf.layers.conv2d(
        pool2,
        256,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv3_1',
    )
    conv3_2 = tf.layers.conv2d(
        conv3_1,
        256,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv3_2',
    )
    conv3_3 = tf.layers.conv2d(
        conv3_2,
        256,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv3_3',
    )
    pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=[2, 2], strides=[2, 2], name='pool3')

    conv4_1 = tf.layers.conv2d(
        pool3,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv4_1',
    )
    conv4_2 = tf.layers.conv2d(
        conv4_1,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv4_2',
    )
    conv4_3 = tf.layers.conv2d(
        conv4_2,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv4_3',
    )
    pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=[2, 2], strides=[2, 2], name='pool4')

    conv5_1 = tf.layers.conv2d(
        pool4,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv5_1',
    )
    conv5_2 = tf.layers.conv2d(
        conv5_1,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv5_2',
    )
    conv5_3 = tf.layers.conv2d(
        conv5_2,
        512,
        [3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        name='conv5_3',
    )
    pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=[2, 2], strides=[2, 2], name='pool5')

    downsample = tf.layers.average_pooling2d(conv4_1, pool_size=[4, 4], strides=[4, 4], name='downsample')
    merged = tf.add(x=pool5, y=downsample, name='merged')

    return merged


def Resnet(n_layers, imgs_in, weight_decay, batch_norm_momentum, is_training):
    assert n_layers in {50, 101, 152, 200}, 'unsupported n_layers'

    network = getattr(resnet_v2, 'resnet_v2_{}'.format(n_layers))
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_momentum)):
        features, _ = network(imgs_in, is_training=is_training, global_pool=False, output_stride=16)

    return features


def MobileNet(depth_multiplier, imgs_in, weight_decay, batch_norm_momentum, is_training):
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training, weight_decay=weight_decay, bn_decay=batch_norm_momentum)):
        features, _ = mobilenet_v2.mobilenet_base(imgs_in, depth_multiplier=depth_multiplier, finegrain_classification_mode=depth_multiplier < 1, output_stride=16)

    return features
