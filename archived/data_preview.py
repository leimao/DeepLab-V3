
import os.path as osp

import numpy as np

import cv2
from PIL import Image

if __name__ == '__main__':

    data_dir = 'data/datasets/VOCdevkit/VOC2012/'
    image_file = osp.join(data_dir, 'JPEGImages/2007_000039.jpg')
    label_file = osp.join(data_dir, 'SegmentationClass/2007_000039.png')
    image = cv2.imread(image_file)

    # Magic function
    # https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py#L42
    label = np.array(Image.open(label_file))
    label = np.expand_dims(label, axis=2)

    print(np.unique(label))
    print(image.shape)
    print(label.shape)
