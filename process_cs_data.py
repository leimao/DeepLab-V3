import os

import numpy as np
from tqdm import tqdm

import cv2
import tensorflow as tf
from utils import id2trainId

cs_dir = 'data/datasets/cityscapes/'
cs_processed_dir = os.path.join(cs_dir, 'processed/deeplab/')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_data(force=False):
    img_spec = 'leftImg8bit'
    img_dir = os.path.join(cs_dir, img_spec)
    with tqdm(desc='Processing Cityscapes data...', unit_scale=True) as p_bar:
        for split in os.listdir(img_dir):
            with tf.python_io.TFRecordWriter(os.path.join(cs_processed_dir, split + '.tfrecord')) as writer:
                img_dir_split = os.path.join(img_dir, split)
                lbl_spec = 'gtCoarse' if split == 'train_extra' else 'gtFine'
                lbl_dir_split = os.path.join(cs_dir, lbl_spec, split)
                for city_name in os.listdir(img_dir_split):
                    img_dir_city = os.path.join(img_dir_split, city_name)
                    lbl_dir_city = os.path.join(lbl_dir_split, city_name)
                    if split != 'test':
                        lbl_dir_save = os.path.join(cs_processed_dir, lbl_spec, split, city_name)
                        if not os.path.isdir(lbl_dir_save):
                            os.makedirs(lbl_dir_save)
                    for img_name in os.listdir(img_dir_city):
                        if split == 'test':
                            lbl_path = ''
                        else:
                            lbl_name = '_'.join((img_name[:-len(img_spec) - 5], lbl_spec, 'labelIds.png'))
                            lbl_path = os.path.join(lbl_dir_save, lbl_name)
                            if not os.path.isfile(lbl_path) or force:
                                lbl = id2trainId(cv2.imread(os.path.join(lbl_dir_city, lbl_name), cv2.IMREAD_UNCHANGED))
                                cv2.imwrite(lbl_path, lbl, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        feature = {'img_path': _bytes_feature(os.path.join(img_dir_city, img_name).encode()), 'lbl_path': _bytes_feature(lbl_path.encode())}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                        p_bar.update()


def compute_img_means(force=False):
    means_filepath = os.path.join(cs_processed_dir, 'img_means.npy')
    if os.path.isfile(means_filepath) and not force:
        return np.load(means_filepath)
    num_pixels = 0
    channel_sums = np.zeros(3, dtype=object)
    with tqdm(desc='Computing image channel means...', unit_scale=True) as p_bar:
        for split in {'train', 'train_extra'}:
            img_dir_split = os.path.join(cs_dir, 'leftImg8bit', split)
            for city_name in os.listdir(img_dir_split):
                img_dir_city = os.path.join(img_dir_split, city_name)
                for img_name in os.listdir(img_dir_city):
                    img = cv2.imread(os.path.join(img_dir_city, img_name))
                    channel_sums += np.sum(img, axis=(0, 1))
                    num_pixels += np.prod(img.shape[:2])
                    p_bar.update()
    img_means = (channel_sums / num_pixels).astype(np.float32)
    if not os.path.isdir(cs_processed_dir):
        os.makedirs(cs_processed_dir)
    np.save(means_filepath, img_means)
    return img_means


def _read_func(example):
    features = {'img_path': tf.FixedLenFeature((), tf.string), 'lbl_path': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example, features)
    return parsed_features['img_path'], parsed_features['lbl_path']


def read_cs_tfrecords(splits):
    if isinstance(splits, str):
        splits = [splits]
    tfrecord_paths = [os.path.join(cs_processed_dir, split + '.tfrecord') for split in splits]
    dataset = tf.data.TFRecordDataset(tfrecord_paths).map(_read_func)
    return dataset


if __name__ == '__main__':
    compute_img_means()
    process_data()
