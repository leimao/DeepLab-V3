import os
from datetime import datetime

import numpy as np

import tensorflow as tf
from feature_extractor import MobileNet, Resnet, Vgg16
from modules import atrous_spatial_pyramid_pooling


class DeepLab(object):
    def __init__(self, img_means, base_architecture='resnet_101', n_classes=21, ignore_label=255, learning_rate=1e-5, weight_decay=5e-4, batch_norm_momentum=0.9997, pre_trained_model=None, log_dir='data/logs/deeplab/'):
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.raw_weight_decay = weight_decay
        self.batch_norm_momentum = batch_norm_momentum

        self.imgs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='imgs')
        self.normalized_imgs = self.imgs - img_means
        self.lbls = tf.placeholder(tf.uint8, shape=[None, None, None], name='lbls')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.target_size = tf.placeholder(tf.int32, shape=[2], name='target_size')
        self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        self.regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        self.build_model(base_architecture)
        if pre_trained_model:
            self.initialize_backbone_from_pretrained_weights(pre_trained_model)
        self.lr = tf.train.exponential_decay(learning_rate, tf.train.create_global_step(), 3000, 0.8)
        self.build_optimizer()

        now = datetime.now()
        self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
        self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
        self.summary()
        self.val_step = tf.get_variable('val_step', initializer=tf.constant(0))
        self.inc_val_step = tf.assign_add(self.val_step, 1)
        self.saver = tf.train.Saver()
        # NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
        # self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, base_architecture):
        with tf.variable_scope('backbone'):
            if base_architecture == 'vgg16':
                feature_map = Vgg16(self.normalized_imgs, self.regularizer, self.batch_norm_momentum)
            elif base_architecture.startswith('resnet'):
                n_layers = int(base_architecture.split('_')[-1])
                feature_map = Resnet(n_layers, self.normalized_imgs, self.weight_decay, self.batch_norm_momentum, self.is_training)
            elif base_architecture.startswith('mobilenet'):
                depth_multiplier = float(base_architecture.split('_')[-1])
                feature_map = MobileNet(depth_multiplier, self.normalized_imgs, self.weight_decay, self.batch_norm_momentum, self.is_training)
            else:
                raise ValueError('Unknown backbone architecture!')
        pools = atrous_spatial_pyramid_pooling(feature_map, regularizer=self.regularizer)
        logits = tf.layers.conv2d(pools, self.n_classes, (1, 1), name='logits')
        self.logits = tf.image.resize_bilinear(logits, self.target_size, name='resized_logits')
        not_ignore_mask = tf.not_equal(self.lbls, self.ignore_label)
        # The locations represented by indices in indices take value on_value, while all other locations take value off_value. For example, ignore_label (255) will be set to zero vector with onehot encoding
        onehot_labels = tf.one_hot(self.lbls, self.n_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels, self.logits, weights=not_ignore_mask)

    def build_optimizer(self):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=tf.train.get_global_step())

    def summary(self):
        with tf.name_scope('loss'):
            self.train_summaries = tf.summary.scalar('train', self.loss)
            self.val_summaries = tf.summary.scalar('val', self.loss)

    def train(self, imgs, lbls, target_size=None):
        _, logits, loss, summaries, train_step = self.sess.run([self.optimizer, self.logits, self.loss, self.train_summaries, tf.train.get_global_step()], feed_dict={self.imgs: imgs, self.lbls: lbls, self.target_size: target_size if target_size else imgs.shape[1:3], self.weight_decay: self.raw_weight_decay * np.mean(lbls != self.ignore_label), self.is_training: True})
        self.writer.add_summary(summaries, train_step)
        return logits, loss

    def validate(self, imgs, lbls, target_size=None):
        logits, loss, summaries, val_step = self.sess.run([self.logits, self.loss, self.val_summaries, self.inc_val_step], feed_dict={self.imgs: imgs, self.lbls: lbls, self.target_size: target_size if target_size else imgs.shape[1:3], self.is_training: False})
        self.writer.add_summary(summaries, val_step)
        return logits, loss

    def test(self, imgs, target_size=None):
        logits = self.sess.run(self.logits, feed_dict={self.imgs: imgs, self.target_size: target_size if target_size else imgs.shape[1:3], self.is_training: False})
        return logits

    def save(self, directory, filename):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        self.saver.save(self.sess, filepath)
        return filepath

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)

    def initialize_backbone_from_pretrained_weights(self, path_to_pretrained_weights):
        variables_to_restore = tf.contrib.slim.get_variables_to_restore()
        valid_prefix = 'backbone/'
        tf.train.init_from_checkpoint(path_to_pretrained_weights, {v.name[len(valid_prefix):].split(':')[0]: v for v in variables_to_restore if v.name.startswith(valid_prefix)})

    def close(self):
        self.writer.close()
        self.sess.close()


if __name__ == '__main__':
    deeplab = DeepLab(np.zeros(3, dtype=np.float32), pre_trained_model='data/models/pretrained/resnet_101/resnet_v2_101.ckpt')
    print('Graph compiled successfully.')
    deeplab.close()
