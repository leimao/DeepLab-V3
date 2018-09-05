

import os

import numpy as np
from tqdm import trange

import tensorflow as tf
from model import DeepLab
from utils import (DataPreprocessor, Dataset, Iterator, add_channel_means,
                   count_label_prediction_matches,
                   mean_intersection_over_union, multiscale_single_test,
                   multiscale_single_validate, save_load_means,
                   validation_demo, validation_single_demo)


def train(train_dataset_filename='./data/VOCdevkit/VOC2012/train_dataset.txt', valid_dataset_filename='./data/VOCdevkit/VOC2012/valid_dataset.txt', test_dataset_filename='./data/VOCdevkit/VOC2012/test_dataset.txt', images_dir='./data/VOCdevkit/VOC2012/JPEGImages', labels_dir='./data/VOCdevkit/VOC2012/SegmentationClass', pre_trained_model='./models/resnet_101/resnet_v2_101.ckpt', model_dir='./models/voc2012', results_dir='./results', log_dir='./log'):

    num_classes = 21
    ignore_label = 255
    num_epochs = 1000
    minibatch_size = 8  # Unable to do minibatch_size = 12 :(
    random_seed = 0
    # learning_rate = 0.00001
    learning_rate = 0.0001
    batch_norm_decay = 0.9997
    model_filename = 'deeplab.ckpt'
    image_shape = [513, 513]

    # validation_scales = [0.5, 1, 1.5]
    validation_scales = [1]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Prepare datasets
    train_dataset = Dataset(dataset_filename=train_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')
    valid_dataset = Dataset(dataset_filename=valid_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')
    test_dataset = Dataset(dataset_filename=test_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')

    # Calculate image channel means
    channel_means = save_load_means(means_filename='./models/channel_means.npz', image_filenames=train_dataset.image_filenames, recalculate=False)

    voc2012_preprocessor = DataPreprocessor(channel_means=channel_means, output_size=image_shape, min_scale_factor=0.5, max_scale_factor=2.0)

    # Prepare dataset iterators
    train_iterator = Iterator(dataset=train_dataset, minibatch_size=minibatch_size, process_func=voc2012_preprocessor.preprocess, random_seed=random_seed, scramble=True, num_jobs=1)
    valid_iterator = Iterator(dataset=valid_dataset, minibatch_size=minibatch_size, process_func=voc2012_preprocessor.preprocess, random_seed=None, scramble=False, num_jobs=1)
    test_iterator = Iterator(dataset=test_dataset, minibatch_size=minibatch_size, process_func=voc2012_preprocessor.preprocess, random_seed=None, scramble=False, num_jobs=1)

    model = DeepLab(is_training=True, num_classes=num_classes, ignore_label=ignore_label, base_architecture='resnet_v2_101', batch_norm_decay=batch_norm_decay, pre_trained_model=pre_trained_model, log_dir=log_dir)

    for i in range(num_epochs):

        print('Epoch number: {}'.format(i))

        print('Start validation ...')

        valid_loss_total = 0
        num_pixel_labels_total = np.zeros(num_classes)
        num_pixel_correct_predictions_total = np.zeros(num_classes)

        # Multi-scale inputs prediction

        for _ in trange(valid_iterator.dataset_size):
            image, label = valid_iterator.next_raw_data()

            output, valid_loss = multiscale_single_validate(image=image, label=label, input_scales=validation_scales, validator=model.validate)
            prediction = np.argmax(output, axis=-1)
            valid_loss_total += valid_loss
            num_pixel_labels, num_pixel_correct_predictions = count_label_prediction_matches(labels=[np.squeeze(label, axis=-1)], predictions=[prediction], num_classes=num_classes, ignore_label=ignore_label)

            num_pixel_labels_total += num_pixel_labels
            num_pixel_correct_predictions_total += num_pixel_correct_predictions

            # validation_single_demo(image = image, label = np.squeeze(label, axis = -1), prediction = prediction, demo_dir = os.path.join(results_dir, 'validation_demo'), filename = str(j))

        """
        for _ in trange(np.ceil(valid_iterator.dataset_size / minibatch_size).astype(int)):

            images, labels = valid_iterator.next_minibatch()
            num_samples = len(images)
            outputs, valid_loss = model.validate(inputs = images, target_height = image_shape[0], target_width = image_shape[1], labels = labels)
            predictions = np.argmax(outputs, axis = -1)
            valid_loss_total += valid_loss * num_samples

            num_pixel_labels, num_pixel_correct_predictions = count_label_prediction_matches(labels=np.squeeze(labels, axis=-1), predictions=predictions, num_classes=num_classes, ignore_label=ignore_label)

            num_pixel_labels_total += num_pixel_labels
            num_pixel_correct_predictions_total += num_pixel_correct_predictions

        validation_demo(images = images, labels = np.squeeze(labels, axis = -1), predictions = predictions, demo_dir = os.path.join(results_dir, 'validation_demo'))
        """

        print(num_pixel_labels_total)
        print(num_pixel_correct_predictions_total)
        print(num_pixel_correct_predictions_total / num_pixel_labels_total)

        mean_IOU = mean_intersection_over_union(num_pixel_labels=num_pixel_labels_total, num_pixel_correct_predictions=num_pixel_correct_predictions_total)

        valid_loss_ave = valid_loss_total / valid_iterator.dataset_size

        print('Validation loss: {:.4f} | mIOU: {:.4f}'.format(valid_loss_ave, mean_IOU))

        print('Start training ...')

        train_loss_total = 0

        for _ in trange(np.ceil(train_iterator.dataset_size / minibatch_size).astype(int)):
            images, labels = train_iterator.next_minibatch()
            num_samples = len(images)
            outputs, train_loss = model.train(inputs=images, labels=labels, target_height=image_shape[0], target_width=image_shape[1], learning_rate=learning_rate)
            predictions = np.argmax(outputs, axis=-1)
            train_loss_total += train_loss * num_samples

        train_iterator.shuffle_dataset()

        validation_demo(images=images, labels=np.squeeze(labels, axis=-1), predictions=predictions, demo_dir=os.path.join(results_dir, 'training_demo'))

        train_loss_ave = train_loss_total / train_iterator.dataset_size

        print('Training loss: {:.4f}'.format(train_loss_ave))


if __name__ == '__main__':

    tf.set_random_seed(0)
    np.random.seed(0)

    train()
