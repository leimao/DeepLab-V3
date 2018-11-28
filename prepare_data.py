import os

import numpy as np
from tqdm import tqdm

import cv2
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_cs_tfrecords():
    cs_dir = 'data/datasets/cityscapes/'
    img_spec = 'leftImg8bit'
    for split in {'train', 'train_extra', 'val', 'test'}:
        with tf.python_io.TFRecordWriter(os.path.join(cs_dir, 'processed', split + '.tfrecord')) as writer:
            img_dir_split = os.path.join(cs_dir, img_spec, split)
            lbl_spec = 'gtCoarse' if split == 'train_extra' else 'gtFine'
            lbl_dir_split = os.path.join(cs_dir, lbl_spec, split)
            for city_name in os.listdir(img_dir_split):
                img_dir_city = os.path.join(img_dir_split, city_name)
                lbl_dir_city = os.path.join(lbl_dir_split, city_name)
                for img_name in os.listdir(img_dir_city):
                    img_id = img_name[:-len(img_spec) - 5]
                    img_path = os.path.join(img_dir_city, img_name).encode()
                    lbl_path = os.path.join(lbl_dir_city, '_'.join((img_id, lbl_spec, 'labelIds.png'))).encode()
                    feature = {'img': _bytes_feature(img_path), 'lbl': _bytes_feature(lbl_path)}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())


def compute_and_save_channel_means(means_filepath, next_img, force=False):
    if os.path.isfile(means_filepath) and not force:
        return
    if next_img is None:
        raise ValueError('next_img is None while calculation is needed')
    with tf.Session() as sess:
        num_pixels = 0
        channel_sums = np.zeros(3, dtype=object)
        p_bar = tqdm(desc='Computing image channel means...', unit_scale=True)
        while True:
            try:
                img_path = sess.run(next_img)
                img = cv2.imread(img_path.decode())
                channel_sums += np.sum(img, axis=(0, 1))
                num_pixels += np.prod(img.shape[:2])
                p_bar.update()
            except tf.errors.OutOfRangeError:
                p_bar.close()
                break
        channel_means = (channel_sums / num_pixels).astype(float)
        np.savez(means_filepath, channel_means=channel_means)


def _read_func(example):
    features = {'img': tf.FixedLenFeature((), tf.string), 'lbl': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example, features)
    return parsed_features['img'], parsed_features['lbl']


def read_cs_tfrecords(splits):
    if isinstance(splits, str):
        splits = [splits]
    tfrecord_paths = [os.path.join('data/datasets/cityscapes/processed/', split + '.tfrecord') for split in splits]
    dataset = tf.data.TFRecordDataset(tfrecord_paths).map(_read_func)
    return dataset


if __name__ == '__main__':
    prepare_cs_tfrecords()
    trainset = read_cs_tfrecords(['train', 'train_extra'])
    it = trainset.make_one_shot_iterator()
    next_img, _ = it.get_next()
    compute_and_save_channel_means('data/datasets/cityscapes/channel_means.npz', next_img)
