from os import path as osp

import numpy as np
from tqdm import trange

from model import DeepLab
from utils import (Dataset, Iterator, save_load_means, subtract_channel_means,
                   validation_single_demo)

if __name__ == '__main__':
    test_dataset_filename = './data/VOCdevkit/VOC2012/test_dataset.txt'
    images_dir = './data/VOCdevkit/VOC2012/JPEGImages'
    labels_dir = './data/VOCdevkit/VOC2012/SegmentationClass'
    results_dir = './results'
    model_dir = './models/deeplab/mobilenet_1.0_voc2012'
    model_filename = 'mobilenet_1.0_0.5728.ckpt'

    channel_means = save_load_means(means_filename='./channel_means.npz', image_filenames=None)

    minibatch_size = 16

    test_dataset = Dataset(dataset_filename=test_dataset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')
    test_iterator = Iterator(dataset=test_dataset, minibatch_size=minibatch_size, process_func=None, random_seed=None, scramble=False, num_jobs=1)

    deep_lab = DeepLab('mobilenet_1.0', is_training=False)
    deep_lab.load(osp.join(model_dir, model_filename))

    n_samples = 8

    for i in trange(n_samples):
        image, label = test_iterator.next_raw_data()
        image = subtract_channel_means(image=image, channel_means=channel_means)

        output = deep_lab.test(inputs=[image], target_height=image.shape[0], target_width=image.shape[1])[0]

        validation_single_demo(image, np.squeeze(label, axis=-1), np.argmax(output, axis=-1), osp.join(results_dir, 'test_demo'), str(i))
