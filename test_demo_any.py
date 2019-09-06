'''
The script does inference (semantic segmentation) on arbitrary images.
Just drop some JPG files into demo_dir and run the script.
Results will be written into the same folder.
For better results channel_means better be recalculated I suppose. But it is kinda tricky.
'''
from os import path as osp
from glob import glob
import numpy as np

from model import DeepLab
from utils import ( save_load_means, subtract_channel_means, single_demo, read_image)

if __name__ == '__main__':

    demo_dir = 'data/demos/deeplab/resnet_101_voc2012/'
    models_dir = 'data/models/deeplab/resnet_101_voc2012/'
    model_filename = 'resnet_101_0.6959.ckpt'
    
    
    channel_means = save_load_means(means_filename='channel_means.npz',image_filenames=None, recalculate=False)

    deeplab = DeepLab('resnet_101', training=False)
    deeplab.load(osp.join(models_dir, model_filename))
    files = glob(demo_dir+'*.jpg')
    for image_filename in files:
        filename=osp.basename(image_filename).split('.')[0]
        image =  read_image(image_filename=image_filename)
        image_input = subtract_channel_means(image=image, channel_means=channel_means)
        output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]
        single_demo(image, np.argmax(output, axis=-1), demo_dir, filename)

    deeplab.close()
