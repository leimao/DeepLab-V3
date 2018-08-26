
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2

from modules import atrous_spatial_pyramid_pooling

class DeepLab(object):

    def __init__(self, is_training, num_classes, ignore_label = 255, base_architecture = 'resnet_v2_101', batch_norm_decay = 0.9997, pre_trained_model = './models/resnet_101/resnet_v2_101.ckpt', log_dir = './log'):

        self.is_training = is_training
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.base_architecture = base_architecture
        self.inputs_shape = [None, None, None, 3]
        self.labels_shape = [None, None, None, 1]
        self.inputs = tf.placeholder(tf.float32, shape = self.inputs_shape, name = 'inputs')
        self.labels = tf.placeholder(tf.uint8, shape = self.labels_shape
            , name = 'labels')

        self.target_height = tf.placeholder(tf.int32, None, name = 'target_image_height')
        self.target_width = tf.placeholder(tf.int32, None, name = 'target_image_width')

        self.pre_trained_model = pre_trained_model
        self.batch_norm_decay = batch_norm_decay

        self.feature_map = self.backbone_initializer()
        self.outputs = self.model_initializer()

        self.learning_rate = tf.placeholder(tf.float32, None, name = 'learning_rate')
        self.loss = self.loss_initializer()
        self.optimizer = self.optimizer_initializer()

        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.is_training == True:
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.train_summaries, self.valid_summaries = self.summary()

    def backbone_initializer(self):

        if self.base_architecture == 'resnet_v2_101':
            self.base_model = resnet_v2.resnet_v2_101
            feature_map = self.resnet_initializer()

        elif self.base_architecture == 'resnet_v2_50':
            self.base_model = resnet_v2.resnet_v2_50
            feature_map = self.resnet_initializer()
        else:
            raise Exception('Unknown backbone architecture')

        return feature_map

    def resnet_initializer(self):

        # Feature map extraction from backbone model
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay = self.batch_norm_decay)):
            logits, end_points = self.base_model(inputs = self.inputs, num_classes = None, is_training = self.is_training, global_pool = False, output_stride = 16)

        if self.is_training == True:
            exclude = [self.base_architecture + '/logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude = exclude)
            tf.train.init_from_checkpoint(self.pre_trained_model, {v.name.split(':')[0]: v for v in variables_to_restore})

        feature_map = end_points[self.base_architecture + '/block4']

        return feature_map

    def model_initializer(self):

        with tf.variable_scope('encoder', reuse = None):
            pools = atrous_spatial_pyramid_pooling(inputs = self.feature_map, filters = 256)
            logits = tf.layers.conv2d(inputs = pools, filters = self.num_classes, kernel_size = (1, 1), activation = None, name = 'logits')
            outputs = tf.image.resize_bilinear(images = logits, size = (self.target_height, self.target_width), name = 'resized_outputs')

        return outputs

    def loss_initializer(self):

        labels_linear = tf.reshape(tensor = self.labels, shape = [-1])
        not_ignore_mask = tf.to_float(tf.not_equal(labels_linear, self.ignore_label))
        # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
        # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
        onehot_labels = tf.one_hot(indices = labels_linear, depth = self.num_classes, on_value = 1.0, off_value = 0.0)

        loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights = not_ignore_mask)

        return loss

    def optimizer_initializer(self):

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

        return optimizer

    def summary(self):

        with tf.name_scope('train_summaries'):
            train_loss_summary = tf.summary.scalar('train_loss', self.loss)

        with tf.name_scope('valid_summaries'):
            valid_loss_summary = tf.summary.scalar('valid_loss', self.loss)

        return train_loss_summary, valid_loss_summary

    def train(self, inputs, labels, target_height, target_width, learning_rate):

        _, train_loss, outputs, summaries = self.sess.run([self.optimizer, self.loss, self.outputs, self.train_summaries], 
            feed_dict = {self.inputs: inputs, self.labels: labels, self.learning_rate: learning_rate, self.target_height: target_height, self.target_width: target_width})

        self.writer.add_summary(summaries, self.train_step)
        self.train_step += 1

        return outputs, train_loss

    def validate(self, inputs, labels, target_height, target_width):

        outputs, valid_loss, summaries = self.sess.run([self.outputs, self.loss, self.valid_summaries], feed_dict = {self.inputs: inputs, self.labels: labels, self.target_height: target_height, self.target_width: target_width})

        self.writer.add_summary(summaries, self.train_step)

        return outputs, valid_loss

    def test(self, inputs, target_height, target_width):

        outputs = self.sess.run(self.outputs, feed_dict = {self.inputs: inputs, self.target_height: target_height, self.target_width: target_width})

        return outputs

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


if __name__ == '__main__':
    
    deeplab = DeepLab(is_training = True, num_classes = 10, base_architecture = 'resnet_v2_101', batch_norm_decay = 0.9997, pre_trained_model = './models/resnet_101/resnet_v2_101.ckpt')
    print('Graph compile successful.')

