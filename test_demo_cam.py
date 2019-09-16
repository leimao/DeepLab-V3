'''
The script does inference (semantic segmentation) on videostream from camera.
Just run the script and watch output in cv2.namedWindow.
Make sure you have trained model and set an existing checkpoint filename as a model_filename
To stop the script press the "q" button.

Created on Sun Sep 15 19:53:37 2019
@author: Pinaxe
'''

from os import path as osp
import numpy as np
import cv2

from model import DeepLab
from utils import ( save_load_means, subtract_channel_means, label_to_color_image)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('reslt', cv2.WINDOW_NORMAL)
    model_filename = 'data/models/deeplab/resnet_101_voc2012/resnet_101_0.3685.ckpt'
         
    channel_means = save_load_means(means_filename='channel_means.npz',image_filenames=None, recalculate=False)

    deeplab = DeepLab('resnet_101', training=False)
    deeplab.load(model_filename)
    
    while(True):
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break 
  
        image=frame
        image_input = subtract_channel_means(image=image, channel_means=channel_means)
        output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]
        
        img=label_to_color_image(np.argmax(output, axis=-1))
        img=img.astype(np.uint8)
        cv2.imshow('reslt', img)
  
    
    cap.release()
    cv2.destroyAllWindows()   
    deeplab.close()
