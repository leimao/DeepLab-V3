
import os

import numpy as np
from tqdm import trange

import tensorflow as tf
from model import DeepLab
from utils import (DataPreprocessor, Dataset, Iterator,
                   count_label_prediction_matches,
                   mean_intersection_over_union, multiscale_single_validate,
                   save_load_means, subtract_channel_means, validation_demo,
                   validation_single_demo)

import argparse


def train(network_backbone, pre_trained_model=None, train_dataset_filename='./data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', val_dataset_filename='./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', images_dir='./data/VOCdevkit/VOC2012/JPEGImages', labels_dir='./data/VOCdevkit/VOC2012/SegmentationClass', train_augmented_dataset_filename='./data/SBD/train_noval.txt', images_augmented_dir='./data/SBD/benchmark_RELEASE/dataset/img', labels_augmented_dir='./data/SBD/benchmark_RELEASE/dataset/cls', model_dir=None, results_dir='./results', log_dir='./log'):

    if not model_dir:
        model_dir = './models/deeplab/{}_voc2012'.format(network_backbone)
    num_classes = 21
    ignore_label = 255
    num_epochs = 1000
    minibatch_size = 8  # Unable to do minibatch_size = 12 :(
    random_seed = 0
    learning_rate = 1e-6
    weight_decay = 5e-4
    batch_norm_decay = 0.99
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
    valid_dataset = Dataset(dataset_filename=val_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')

    # Calculate image channel means
    channel_means = save_load_means(means_filename='./channel_means.npz', image_filenames=train_dataset.image_filenames, recalculate=False)

    voc2012_preprocessor = DataPreprocessor(channel_means=channel_means, output_size=image_shape, min_scale_factor=0.5, max_scale_factor=2.0)

    # Prepare dataset iterators
    train_iterator = Iterator(dataset=train_dataset, minibatch_size=minibatch_size, process_func=voc2012_preprocessor.preprocess, random_seed=random_seed, scramble=True, num_jobs=1)
    valid_iterator = Iterator(dataset=valid_dataset, minibatch_size=minibatch_size, process_func=voc2012_preprocessor.preprocess, random_seed=None, scramble=False, num_jobs=1)

    # Prepare augmented dataset
    train_augmented_dataset = Dataset(dataset_filename=train_augmented_dataset_filename, images_dir=images_augmented_dir, labels_dir=labels_augmented_dir, image_extension='.jpg', label_extension='.mat')

    channel_augmented_means = save_load_means(means_filename='./channel_augmented_means.npz', image_filenames=train_augmented_dataset.image_filenames, recalculate=False)

    voc2012_augmented_preprocessor = DataPreprocessor(channel_means=channel_augmented_means, output_size=image_shape, min_scale_factor=0.5, max_scale_factor=2.0)
    train_augmented_iterator = Iterator(dataset=train_augmented_dataset, minibatch_size=minibatch_size, process_func=voc2012_augmented_preprocessor.preprocess, random_seed=random_seed, scramble=True, num_jobs=1)

    model = DeepLab(network_backbone, num_classes=num_classes, ignore_label=ignore_label, batch_norm_momentum=batch_norm_decay, pre_trained_model=pre_trained_model, log_dir=log_dir)

    best_mIoU = 0

    for i in range(num_epochs):

        print('Epoch number: {}'.format(i))

        print('Start validation ...')

        valid_loss_total = 0
        num_pixels_union_total = np.zeros(num_classes)
        num_pixels_intersection_total = np.zeros(num_classes)

        # Multi-scale inputs prediction
        for _ in trange(valid_iterator.dataset_size):
            image, label = valid_iterator.next_raw_data()
            image = subtract_channel_means(image=image, channel_means=channel_means)

            output, valid_loss = multiscale_single_validate(image=image, label=label, input_scales=validation_scales, validator=model.validate)
            valid_loss_total += valid_loss

            prediction = np.argmax(output, axis=-1)
            num_pixels_union, num_pixels_intersection = count_label_prediction_matches(labels=[np.squeeze(label, axis=-1)], predictions=[prediction], num_classes=num_classes, ignore_label=ignore_label)

            num_pixels_union_total += num_pixels_union
            num_pixels_intersection_total += num_pixels_intersection

            # validation_single_demo(image=image, label=np.squeeze(label, axis=-1), prediction=prediction, demo_dir=os.path.join(results_dir, 'validation_demo'), filename=str(_))

        mean_IOU = mean_intersection_over_union(num_pixels_union=num_pixels_union_total, num_pixels_intersection=num_pixels_intersection_total)

        valid_loss_ave = valid_loss_total / valid_iterator.dataset_size

        print('Validation loss: {:.4f} | mIoU: {:.4f}'.format(valid_loss_ave, mean_IOU))

        if mean_IOU > best_mIoU:
            best_mIoU = mean_IOU
            model_savename = '{}_{:.4f}.ckpt'.format(network_backbone, best_mIoU)
            print('New best mIoU achieved, model saved as {}.'.format(model_savename))
            model.save(model_dir, model_savename)

        print('Start training ...')

        train_loss_total = 0
        num_pixels_union_total = np.zeros(num_classes)
        num_pixels_intersection_total = np.zeros(num_classes)

        print('Training using VOC2012 ...')
        for _ in trange(np.ceil(train_iterator.dataset_size / minibatch_size).astype(int)):
            images, labels = train_iterator.next_minibatch()
            balanced_weight_decay = weight_decay * sum(labels != ignore_label) / labels.size
            outputs, train_loss = model.train(inputs=images, labels=labels, target_height=image_shape[0], target_width=image_shape[1], learning_rate=learning_rate, weight_decay=balanced_weight_decay)
            train_loss_total += train_loss

            predictions = np.argmax(outputs, axis=-1)
            num_pixels_union, num_pixels_intersection = count_label_prediction_matches(labels=np.squeeze(labels, axis=-1), predictions=predictions, num_classes=num_classes, ignore_label=ignore_label)

            num_pixels_union_total += num_pixels_union
            num_pixels_intersection_total += num_pixels_intersection

            # validation_demo(images=images, labels=np.squeeze(labels, axis=-1), predictions=predictions, demo_dir=os.path.join(results_dir, 'training_demo'), batch_no=_)
        train_iterator.shuffle_dataset()

        print('Training using SBD ...')
        for _ in trange(np.ceil(train_augmented_iterator.dataset_size / minibatch_size).astype(int)):
            images, labels = train_augmented_iterator.next_minibatch()
            balanced_weight_decay = weight_decay * sum(labels != ignore_label) / labels.size
            outputs, train_loss = model.train(inputs=images, labels=labels, target_height=image_shape[0], target_width=image_shape[1], learning_rate=learning_rate, weight_decay=balanced_weight_decay)
            train_loss_total += train_loss

            predictions = np.argmax(outputs, axis=-1)
            num_pixels_union, num_pixels_intersection = count_label_prediction_matches(labels=np.squeeze(labels, axis=-1), predictions=predictions, num_classes=num_classes, ignore_label=ignore_label)

            num_pixels_union_total += num_pixels_union
            num_pixels_intersection_total += num_pixels_intersection

            # validation_demo(images=images, labels=np.squeeze(labels, axis=-1), predictions=predictions, demo_dir=os.path.join(results_dir, 'training_demo'), batch_no=_)
        train_augmented_iterator.shuffle_dataset()

        mIoU = mean_intersection_over_union(num_pixels_union=num_pixels_union_total, num_pixels_intersection=num_pixels_intersection_total)
        train_loss_ave = train_loss_total / (train_iterator.dataset_size + train_augmented_iterator.dataset_size)
        print('Training loss: {:.4f} | mIoU: {:.4f}'.format(train_loss_ave, mIoU))

    model.close()



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description = 'Train DeepLab v3 for image semantic segmantation.')

    network_backbone_default = 'resnet_101'
    pre_trained_model_default  = './models/pretrained/resnet_101/resnet_v2_101.ckpt'
    train_dataset_filename_default = './data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    val_dataset_filename_default = './data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    images_dir_default = './data/VOCdevkit/VOC2012/JPEGImages'
    labels_dir_default  = './data/VOCdevkit/VOC2012/SegmentationClass'
    train_augmented_dataset_filename_default = './data/SBD/train_noval.txt'
    images_augmented_dir_default = './data/SBD/benchmark_RELEASE/dataset/img'
    labels_augmented_dir_default = './data/SBD/benchmark_RELEASE/dataset/cls'
    model_dir_default = './models/deeplab/{}_voc2012'.format(network_backbone_default)
    results_dir_default = './results'
    log_dir_default = './log'
    random_seed_default = 0

    parser.add_argument('--network_backbone', type = str, help = 'Network backbones: resnet_50, resnet_101, mobilenet_1.0_224. Default: resnet_101', default = network_backbone_default)
    parser.add_argument('--pre_trained_model', type = str, help = 'Pretrained model directory', default = pre_trained_model_default)
    parser.add_argument('--train_dataset_filename', type = str, help = 'Train dataset filename', default = train_dataset_filename_default)
    parser.add_argument('--val_dataset_filename', type = str, help = 'Validation dataset filename', default = val_dataset_filename_default)
    parser.add_argument('--images_dir', type = str, help = 'Images directory', default = images_dir_default)
    parser.add_argument('--labels_dir', type = str, help = 'Labels directory', default = labels_dir_default)
    parser.add_argument('--train_augmented_dataset_filename', type = str, help = 'Train augmented dataset filename', default = train_augmented_dataset_filename_default)
    parser.add_argument('--images_augmented_dir', type = str, help = 'Images augmented directory', default = images_augmented_dir_default)
    parser.add_argument('--labels_augmented_dir', type = str, help = 'Labels augmented directory', default = labels_augmented_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Trained model saving directory', default = model_dir_default)
    parser.add_argument('--results_dir', type = str, help = 'Training demo directory', default = results_dir_default)
    parser.add_argument('--log_dir', type = str, help = 'TensorBoard log directory', default = log_dir_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)

    argv = parser.parse_args()

    network_backbone = argv.network_backbone
    pre_trained_model = argv.pre_trained_model
    train_dataset_filename = argv.train_dataset_filename
    val_dataset_filename = argv.val_dataset_filename
    images_dir = argv.images_dir
    labels_dir = argv.labels_dir
    train_augmented_dataset_filename = argv.train_augmented_dataset_filename
    images_augmented_dir = argv.images_augmented_dir
    labels_augmented_dir = argv.labels_augmented_dir
    model_dir = argv.model_dir
    results_dir = argv.results_dir
    log_dir = argv.log_dir
    random_seed = argv.random_seed


    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)


    train(network_backbone=network_backbone, pre_trained_model=pre_trained_model, train_dataset_filename=train_dataset_filename, val_dataset_filename=val_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, train_augmented_dataset_filename=train_augmented_dataset_filename, images_augmented_dir=images_augmented_dir, labels_augmented_dir=labels_augmented_dir, model_dir=model_dir, results_dir=results_dir, log_dir=log_dir)

