from os import path as osp

import numpy as np

from model import DeepLab
from prepare_data import read_cs_tfrecords
from utils import fetch_batch, multiscale_test, validation_demo

if __name__ == '__main__':
    demo_dir = 'data/demos/deeplab/resnet_101_cs/'
    model_filepath = 'data/models/deeplab/resnet_101_cs/resnet_101_0.6326.ckpt'
    channel_means = np.load('data/datasets/cityscapes/channel_means.npz')['channel_means']
    batch_size = 16
    testset = read_cs_tfrecords('test').batch(batch_size)
    test_it = testset.make_one_shot_iterator()
    test_data = test_it.get_next()
    deeplab = DeepLab(n_classes=19)
    deeplab.load(model_filepath)
    n_batches = 2
    for i in range(n_batches):
        imgs = fetch_batch(deeplab.sess.run(test_data), channel_means, get_lbl=False)
        logits = multiscale_test(deeplab.test, imgs, [1])
        validation_demo(imgs, None, np.argmax(logits, axis=-1), demo_dir, str(i))
    deeplab.close()
