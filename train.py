import argparse
import os

import numpy as np
from tqdm import tqdm

import cv2
import tensorflow as tf
from model import DeepLab
from process_cs_data import compute_img_means, read_cs_tfrecords
from utils import (count_label_prediction_matches, fetch_batch,
                   mean_intersection_over_union, multiscale_validate,
                   validation_demo)


def validate(model, val_init, val_data, n_classes, scales, ignore_lbl, batch_size, valset_size=0, track_valset_size=True, best_mIoU=0):
    val_loss_total = 0
    num_pixels_union_total = np.zeros(n_classes)
    num_pixels_intersection_total = np.zeros(n_classes)
    p_bar = tqdm(desc='Validating...', total=None if track_valset_size else valset_size, unit_scale=True)
    model.sess.run(val_init)
    while True:
        try:
            imgs, lbls = fetch_batch(model.sess.run(val_data))
            logits, val_loss = multiscale_validate(model.validate, imgs, lbls, scales)
            val_loss_total += val_loss
            predictions = np.argmax(logits, axis=-1)
            num_pixels_union, num_pixels_intersection = count_label_prediction_matches(labels=lbls, predictions=predictions, num_classes=n_classes, ignore_label=ignore_lbl)
            num_pixels_union_total += num_pixels_union
            num_pixels_intersection_total += num_pixels_intersection
            # validation_demo(images=imgs, labels=lbls, predictions=predictions, demo_dir=os.path.join(results_dir, 'validation_demo'), batch_no=p_bar.n)
            if track_valset_size:
                valset_size += 1
            p_bar.update()
        except tf.errors.OutOfRangeError:
            if track_valset_size:
                track_valset_size = False
            p_bar.close()
            break

    mean_IOU = mean_intersection_over_union(num_pixels_union=num_pixels_union_total, num_pixels_intersection=num_pixels_intersection_total)
    val_loss_ave = val_loss_total / valset_size / batch_size
    print('Validation loss: {:.4f} | mIoU: {:.4f}'.format(val_loss_ave, mean_IOU))
    if mean_IOU > best_mIoU:
        best_mIoU = mean_IOU
        model_savename = '{}_{:.4f}.ckpt'.format(network_backbone, best_mIoU)
        print('New best mIoU achieved, model saved as {}.'.format(model_savename))
        model.save(model_dir, model_savename)

    return valset_size, best_mIoU


def train(network_backbone, pre_trained_model=None, model_dir=None, log_dir='data/logs/deeplab/'):
    if not model_dir:
        model_dir = 'data/models/deeplab/{}_cs/'.format(network_backbone)
    n_classes = 19
    ignore_label = 255
    n_epochs = 20
    batch_size = 8
    learning_rate = 1e-5
    weight_decay = 5e-4
    batch_norm_decay = 0.99
    scales = [1]

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    trainset = read_cs_tfrecords(['train', 'train_extra']).shuffle(3000).batch(batch_size)
    train_it = trainset.make_initializable_iterator()
    train_init = train_it.initializer
    train_data = train_it.get_next()
    valset = read_cs_tfrecords('val').batch(batch_size)
    val_it = valset.make_initializable_iterator()
    val_init = val_it.initializer
    val_data = val_it.get_next()
    model = DeepLab(compute_img_means(), base_architecture=network_backbone, n_classes=n_classes, ignore_label=ignore_label, learning_rate=learning_rate, weight_decay=weight_decay, batch_norm_momentum=batch_norm_decay, pre_trained_model=pre_trained_model, log_dir=log_dir)

    trainset_size = 0
    track_trainset_size = True
    valset_size, best_mIoU = validate(model, val_init, val_data, n_classes, scales, ignore_label, batch_size)

    for i in range(n_epochs):
        print('Epoch number: {}'.format(i + 1))
        train_loss_total = 0
        num_pixels_union_total = np.zeros(n_classes)
        num_pixels_intersection_total = np.zeros(n_classes)
        p_bar = tqdm(desc='Training...', total=None if track_trainset_size else trainset_size, unit_scale=True)
        model.sess.run(train_init)
        while True:
            try:
                imgs, lbls = fetch_batch(model.sess.run(train_data), augment=True, min_scale_factor=0.25, max_scale_factor=1.0)
                logits, train_loss = model.train(imgs, lbls)
                train_loss_total += train_loss
                predictions = np.argmax(logits, axis=-1)
                num_pixels_union, num_pixels_intersection = count_label_prediction_matches(labels=lbls, predictions=predictions, num_classes=n_classes, ignore_label=ignore_label)
                num_pixels_union_total += num_pixels_union
                num_pixels_intersection_total += num_pixels_intersection
                # validation_demo(images=imgs, labels=lbls, predictions=predictions, demo_dir=os.path.join(results_dir, 'training_demo'), batch_no=p_bar.n)
                if track_trainset_size:
                    trainset_size += 1
                p_bar.update()
                pass
            except tf.errors.OutOfRangeError:
                if track_trainset_size:
                    track_trainset_size = False
                p_bar.close()
                break

        mIoU = mean_intersection_over_union(num_pixels_union=num_pixels_union_total, num_pixels_intersection=num_pixels_intersection_total)
        train_loss_ave = train_loss_total / trainset_size / batch_size
        print('Training loss: {:.4f} | mIoU: {:.4f}'.format(train_loss_ave, mIoU))
        _, best_mIoU = validate(model, val_init, val_data, n_classes, scales, ignore_label, batch_size, valset_size=valset_size, track_valset_size=False, best_mIoU=best_mIoU)

    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepLab v3 for image semantic segmentation.')

    network_backbone_default = 'resnet_101'
    pre_trained_model_default = 'data/models/pretrained/resnet_101/resnet_v2_101.ckpt'
    model_dir_default = 'data/models/deeplab/{}_cs/'.format(network_backbone_default)
    log_dir_default = 'data/logs/deeplab/'
    random_seed_default = 0

    parser.add_argument('--network_backbone', type=str, help='Network backbones: resnet_50, resnet_101, mobilenet_1.0_224. Default: resnet_101', default=network_backbone_default)
    parser.add_argument('--pre_trained_model', type=str, help='Pretrained model directory', default=pre_trained_model_default)
    parser.add_argument('--model_dir', type=str, help='Trained model saving directory', default=model_dir_default)
    parser.add_argument('--log_dir', type=str, help='TensorBoard log directory', default=log_dir_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', default=random_seed_default)

    argv = parser.parse_args()
    network_backbone = argv.network_backbone
    pre_trained_model = argv.pre_trained_model
    model_dir = argv.model_dir
    log_dir = argv.log_dir
    random_seed = argv.random_seed

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    train(network_backbone, pre_trained_model=pre_trained_model, model_dir=model_dir, log_dir=log_dir)
