from os import path as osp

import numpy as np

from model import DeepLab
from utils import (Dataset, Iterator, save_load_means, subtract_channel_means, single_demo, read_image)

if __name__ == '__main__':

    demo_dir = 'data/demos/deeplab/MyImg/'
    models_dir = 'data/models/deeplab/resnet_101_voc2012/'
    model_filename = 'resnet_101_0.6959.ckpt'
    image_filename='data/datasets/MyImg/JPEGImages/00.jpg'

    channel_means = save_load_means(means_filename='channel_means1.npz',image_filenames=None, recalculate=False)

    deeplab = DeepLab('resnet_101', training=False)
    deeplab.load(osp.join(models_dir, model_filename))

    image =  read_image(image_filename=image_filename)
    image_input = subtract_channel_means(image=image, channel_means=channel_means)

    output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]

    single_demo(image, np.argmax(output, axis=-1), demo_dir, "")

    deeplab.close()
