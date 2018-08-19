
import cv2
import numpy as np
import imageio
from PIL import Image

def main():

    image_file = './data/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg'
    label_file = './data/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png'
    image = cv2.imread(image_file)
    # Magic function
    #https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py#L42
    label = np.asarray(Image.open(label_file))
    ss = label.copy()
    label = np.expand_dims(label, axis = 2)
    assert np.array_equal(ss, label[:,:,0])

    print(np.unique(label))


    print(image.shape)
    print(label.shape)



if __name__ == '__main__':
    
    main()