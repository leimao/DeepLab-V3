import os

import numpy as np
from tqdm import tqdm

import cv2
import tensorflow as tf
from cs_utils import id2trainId
from utils import _bytes_feature

cs_dir = 'data/datasets/cityscapes/'
cs_processed_dir = os.path.join(cs_dir, 'processed/deeplab/')


def process_data(force=False):
    img_dir = os.path.join(cs_dir, 'leftImg8bit')
    with tqdm(desc='Processing Cityscapes data...', unit_scale=True) as p_bar:
        for split in os.listdir(img_dir):
            with tf.python_io.TFRecordWriter(os.path.join(cs_processed_dir, split + '.tfrecord')) as writer:
                img_dir_split = os.path.join(img_dir, split)
                lbl_spec = 'gtCoarse' if split == 'train_extra' else 'gtFine'
                lbl_dir_split = os.path.join(cs_dir, lbl_spec, split)
                lbl_dir_proc_split = os.path.join(cs_processed_dir, lbl_spec, split)
                for city in os.listdir(img_dir_split):
                    img_dir_city = os.path.join(img_dir_split, city)
                    lbl_dir_city = os.path.join(lbl_dir_split, city)
                    lbl_dir_proc_city = os.path.join(lbl_dir_proc_split, city)
                    if not (split in {'test', 'demoVideo'} or os.path.isdir(lbl_dir_proc_city)):
                        os.makedirs(lbl_dir_proc_city)
                    img_names = os.listdir(img_dir_city)
                    if split == 'demoVideo':
                        img_names = sorted(img_names)
                    for img_name in img_names:
                        img_path = os.path.join(img_dir_city, img_name)
                        if split in {'test', 'demoVideo'}:
                            lbl_path_proc = ''
                        else:
                            lbl_name = '_'.join((img_name[:-16], lbl_spec, 'labelIds.png'))
                            lbl_path = os.path.join(lbl_dir_city, lbl_name)
                            lbl_path_proc = os.path.join(lbl_dir_proc_city, lbl_name)
                            if not os.path.isfile(lbl_path_proc) or force:
                                lbl = id2trainId(cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED))
                                cv2.imwrite(lbl_path_proc, lbl, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        feature = {'img_path': _bytes_feature(img_path.encode()), 'lbl_path': _bytes_feature(lbl_path_proc.encode())}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                        p_bar.update()


def compute_img_means(force=False):
    means_filepath = os.path.join(cs_processed_dir, 'img_means.npy')
    if os.path.isfile(means_filepath) and not force:
        return np.load(means_filepath)
    num_pixels = 0
    channel_sums = np.zeros(3, dtype=object)
    img_dir = os.path.join(cs_dir, 'leftImg8bit')
    with tqdm(desc='Computing image channel means...', unit_scale=True) as p_bar:
        for split in {'train', 'train_extra'}:
            img_dir_split = os.path.join(img_dir, split)
            for city in os.listdir(img_dir_split):
                img_dir_city = os.path.join(img_dir_split, city)
                for img_name in os.listdir(img_dir_city):
                    img = cv2.imread(os.path.join(img_dir_city, img_name))
                    channel_sums += np.sum(img, axis=(0, 1))
                    num_pixels += np.prod(img.shape[:2])
                    p_bar.update()
    img_means = (channel_sums / num_pixels).astype(np.float32)
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
    if not os.path.isdir(cs_processed_dir):
        os.makedirs(cs_processed_dir)
    process_data()
    compute_img_means()
