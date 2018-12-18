import os

import numpy as np
import scipy.io as sio

import cv2
import tensorflow as tf
from cs_utils import trainId2color
from PIL import Image


def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte.'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''Returns a float_list from a float / double.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def colormap(lbls):
    return np.stack(trainId2color(lbls), axis=-1)


def static_vars(**kwargs):
    def decorate(func):
        for key, val in kwargs.items():
            setattr(func, key, val)
        return func
    return decorate


class RandomStateStack:
    def __init__(self):
        self.random_state = np.random.get_state()

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.random_state)


def read_label(lbl_path, is_palettised=False):
    if is_palettised:
        return np.array(Image.open(lbl_path))
    if lbl_path.endswith('.mat'):
        # http://home.bharathh.info/pubs/codes/SBD/download.html
        return sio.loadmat(lbl_path)['GTcls']['Segmentation'][0][0]
    return cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)


def resize_image_and_label(image, label, output_size):
    '''
    output_size: [height, width]
    '''

    output_size = tuple(output_size[::-1])
    image_resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, output_size, interpolation=cv2.INTER_NEAREST)
    return image_resized, label_resized


def pad_image_and_label(image, label, top, bottom, left, right, pixel_value, label_value):
    '''
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#making-borders-for-images-padding
    '''

    image_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel_value)
    label_padded = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=label_value)
    return image_padded, label_padded


def random_crop(image, label, output_size):
    assert image.shape[0] >= output_size[0] and image.shape[1] >= output_size[1], 'image size smaller than the desired output size.'

    height_start = np.random.randint(image.shape[0] - output_size[0] + 1)
    width_start = np.random.randint(image.shape[1] - output_size[1] + 1)
    height_end = height_start + output_size[0]
    width_end = width_start + output_size[1]
    image_cropped = image[height_start:height_end, width_start:width_end]
    label_cropped = label[height_start:height_end, width_start:width_end]

    return image_cropped, label_cropped


def flip_image_and_label(image, label):
    image_flipped = np.fliplr(image)
    label_flipped = np.fliplr(label)
    return image_flipped, label_flipped


def image_augmentaion(image, label, output_size, min_scale_factor, max_scale_factor, img_means, ignore_lbl):
    original_height = image.shape[0]
    original_width = image.shape[1]
    target_height = output_size[0]
    target_width = output_size[1]

    scale_factor = np.random.uniform(low=min_scale_factor, high=max_scale_factor)
    scaled_height = round(original_height * scale_factor)
    scaled_width = round(original_width * scale_factor)
    image, label = resize_image_and_label(image, label, [scaled_height, scaled_width])

    # if rescaled_size[0] < target_height:
    #     vertical_pad = round(target_height * 1.5) - rescaled_size[0]
    # else:
    #     vertical_pad = round(rescaled_size[0] * 0.5)

    vertical_pad = round(target_height * 1.5) - scaled_height
    if vertical_pad < 0:
        vertical_pad = 0
    vertical_pad_up = vertical_pad // 2
    vertical_pad_down = vertical_pad - vertical_pad_up

    # if rescaled_size[1] < target_width:
    #     horizonal_pad = round(target_width * 1.5) - rescaled_size[1]
    # else:
    #     horizonal_pad = round(rescaled_size[1] * 0.5)

    horizonal_pad = round(target_width * 1.5) - scaled_width
    if horizonal_pad < 0:
        horizonal_pad = 0
    horizonal_pad_left = horizonal_pad // 2
    horizonal_pad_right = horizonal_pad - horizonal_pad_left

    image, label = pad_image_and_label(image, label, vertical_pad_up, vertical_pad_down, horizonal_pad_left, horizonal_pad_right, img_means, ignore_lbl)
    image, label = random_crop(image, label, output_size)
    # Flip image and label
    if np.random.random() < 0.5:
        image, label = flip_image_and_label(image, label)

    return image, label


def preprocess_data(img_path, lbl_path, get_gt, augment, output_size, min_scale_factor, max_scale_factor, img_means, ignore_lbl):
    img = cv2.imread(img_path.decode()).astype(np.float32)
    if get_gt:
        lbl = read_label(lbl_path.decode())
        if augment:
            img, lbl = image_augmentaion(img, lbl, output_size, min_scale_factor, max_scale_factor, img_means, ignore_lbl)
        return img, lbl
    return img


def fetch_batch(paths, get_lbl=True, augment=False, output_size=(513, 513), min_scale_factor=0.5, max_scale_factor=2.0, img_means=0, ignore_lbl=255):
    data = zip(*[preprocess_data(*path_pair, get_lbl, augment, output_size, min_scale_factor, max_scale_factor, img_means, ignore_lbl) for path_pair in zip(*paths)])
    if get_lbl:
        imgs, lbls = data
        return np.asarray(imgs), np.asarray(lbls)
    return np.asarray(data)


'''
The following image annotition saving codes in the block are slightly modified from Google's official DeepLab repository.
https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py
'''


def bit_get(val, idx):
    '''
    Gets the bit value.
    Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.
    Returns:
    The "idx"-th bit of input val.
    '''

    return (val >> idx) & 1


def create_pascal_label_colormap():
    '''
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    Reference:
    '''

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


# @static_vars(colormap=create_pascal_label_colormap())
def label_to_color_image(label):
    '''
    Adds color defined by the dataset colormap to the label.
    Args:
    label: A 2D array with integer type, storing the segmentation label.
    dataset: The colormap used in the dataset.
    Returns:
    result: A 2D array with floating type. The element of the array is the color indexed by the corresponding element in the input label to the dataset color map.
    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color map maximum entry.
    '''
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    if np.max(label) > 255:
        raise ValueError('label value too large.')

    # return label_to_color_image.colormap[label]
    return colormap(label)


def save_annotation(label, filename, add_colormap=True):
    '''
    Saves the given label to image on disk.
    Args:
    label: The numpy array to be saved. The data will be converted to uint8 and saved as png image.
    save_dir: The directory to which the results will be saved.
    filename: The image filename.
    add_colormap: Add color map to the label or not.
    colormap_type: Colormap type for visualization.
    '''

    # Add colormap for visualizing the prediction.
    if add_colormap:
        label = label_to_color_image(label)
    cv2.imwrite(filename, label)


'''
Evaluation
'''


def validation_demo(images, labels, predictions, demo_dir, batch_no):
    has_lbls = labels is not None
    assert images.ndim == 4 and predictions.ndim == 3 and (not has_lbls or labels.ndim == 3)

    if not os.path.isdir(demo_dir):
        os.makedirs(demo_dir)

    for i in range(len(images)):
        cv2.imwrite(os.path.join(demo_dir, 'image_{}_{}.png'.format(batch_no, i)), images[i])
        if has_lbls:
            save_annotation(label=labels[i], filename=os.path.join(demo_dir, 'image_{}_{}_label.png'.format(batch_no, i)), add_colormap=True)
        save_annotation(label=predictions[i], filename=os.path.join(demo_dir, 'image_{}_{}_prediction.png'.format(batch_no, i)), add_colormap=True)


def count_label_prediction_matches(labels, predictions, num_classes, ignore_label=255):
    '''
    Pixel intersection-over-union averaged across number of classes.
    Assuming valid labels are from 0 to num_classes - 1.
    Support list shaped labels and predictions.
    '''

    num_pixels_union = np.zeros(num_classes)
    num_pixels_intersection = np.zeros(num_classes)

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    assert labels.shape == predictions.shape

    predictions[labels == ignore_label] = ignore_label
    for i in range(num_classes):
        label_class_mask = labels == i
        prediction_class_mask = predictions == i
        num_pixels_union[i] = np.sum(label_class_mask | prediction_class_mask)
        num_pixels_intersection[i] = np.sum(label_class_mask & prediction_class_mask)

    return num_pixels_union, num_pixels_intersection


def mean_intersection_over_union(num_pixels_union, num_pixels_intersection):
    valid_classes = num_pixels_union > 0
    mean_iou = np.mean(num_pixels_intersection[valid_classes] / num_pixels_union[valid_classes])

    return mean_iou


def multiscale_test(predictor, imgs, scales):
    '''
    Predict semantic segmentation labels using multiscale images.
    Inputs:
    predictor: prediction function which takes a batch of images as input and predicts the semantic segmentation labels.
    imgs: numpy array, [batch_size, height, width, channel], channel = 3.
    input_scales: list of scale factors. e.g., [0.5, 1.0, 1.5].

    Returns:
    Averaged predicted logits of multiscale inputs
    '''

    size = imgs.shape[1:3]
    multiscale_logits = []
    for scale in scales:
        size_scaled = tuple(np.round(np.asarray(size) * scale)[::-1])
        imgs_scaled = np.asarray([cv2.resize(img, size_scaled, interpolation=cv2.INTER_LINEAR) for img in imgs])
        logits = predictor(imgs_scaled, target_size=size)
        multiscale_logits.append(logits)
    logits_mean = np.mean(multiscale_logits, axis=0)
    return logits_mean


def multiscale_validate(validator, imgs, lbls, scales):
    size = imgs.shape[1:3]
    multiscale_logits = []
    multiscale_loss = []
    for scale in scales:
        size_scaled = tuple(np.round(np.asarray(size) * scale)[::-1])
        imgs_scaled = np.asarray([cv2.resize(img, size_scaled, interpolation=cv2.INTER_LINEAR) for img in imgs])
        logits, loss = validator(imgs_scaled, lbls, target_size=size)
        multiscale_logits.append(logits)
        multiscale_loss.append(loss)
    logits_mean = np.mean(multiscale_logits, axis=0)
    loss_mean = np.mean(multiscale_loss)
    return logits_mean, loss_mean
