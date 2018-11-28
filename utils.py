import os
from collections import namedtuple

import numpy as np
import scipy.io

import cv2
from PIL import Image

Label = namedtuple('Label', [
    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class
    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.
    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!
    'category',  # The name of the category that this label belongs to
    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.
    'hasInstances',  # Whether this label distinguishes between single instances or not
    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not
    'color',  # The color of this label
])

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
labels = [
    # name id trainId category catId hasInstances ignoreInEval color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

id2trainId = np.vectorize({label.id: label.trainId for label in labels}.get)
trainId2color = np.vectorize({label.trainId: label.color for label in reversed(labels)}.get)


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


# def read_label(label_filename):
#     if label_filename.endswith('.mat'):
#         # http://home.bharathh.info/pubs/codes/SBD/download.html
#         mat = scipy.io.loadmat(label_filename)
#         label = mat['GTcls']['Segmentation'][0][0]
#     else:
#         # Magic function to read VOC2012 semantic labelshttps://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py#L42
#         label = np.asarray(Image.open(label_filename))
#     return label


def resize_image_and_label(image, label, output_size):
    '''
    output_size: [height, width]
    '''

    output_size = tuple(output_size[::-1])
    image_resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, output_size, interpolation=cv2.INTER_NEAREST)
    return image_resized, label_resized


def pad_image_and_label(image, label, top, bottom, left, right, pixel_value=0, label_value=255):
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


def image_augmentaion(image, label, output_size, min_scale_factor=0.5, max_scale_factor=2.0):
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

    image, label = pad_image_and_label(image=image, label=label, top=vertical_pad_up, bottom=vertical_pad_down, left=horizonal_pad_left, right=horizonal_pad_right, pixel_value=0, label_value=255)
    image, label = random_crop(image=image, label=label, output_size=output_size)
    # Flip image and label
    if np.random.random() < 0.5:
        image, label = flip_image_and_label(image=image, label=label)
    label = np.expand_dims(label, axis=2)

    return image, label


def preprocess_data(img_path, lbl_path, channel_means, get_lbl, augment, output_size, min_scale_factor, max_scale_factor):
    img = cv2.imread(img_path.decode())
    img = img - channel_means
    if get_lbl:
        lbl = cv2.imread(lbl_path.decode())[..., 0, None]
        lbl = id2trainId(lbl)
    if augment:
        img, lbl = image_augmentaion(img, lbl, output_size, min_scale_factor=min_scale_factor, max_scale_factor=max_scale_factor)
    if get_lbl:
        return img, lbl
    return img


def fetch_batch(paths, channel_means, get_lbl=True, augment=False, output_size=(513, 513), min_scale_factor=0.5, max_scale_factor=2.0):
    data = zip(*[preprocess_data(*path_pair, channel_means, get_lbl, augment, output_size, min_scale_factor, max_scale_factor) for path_pair in zip(*paths)])
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

    image = Image.fromarray(label.astype(dtype=np.uint8))
    image.save(filename)


'''
Evaluation
'''


def validation_demo(images, labels, predictions, demo_dir, batch_no):
    has_lbls = labels is not None
    assert images.ndim == 4 and predictions.ndim == 3 and (not has_lbls or labels.ndim == 3)

    if not os.path.isdir(demo_dir):
        os.makedirs(demo_dir)

    for i in range(len(images)):
        cv2.imwrite(os.path.join(demo_dir, 'image_{}_{}.jpg'.format(batch_no, i)), images[i])
        if has_lbls:
            save_annotation(label=labels[i], filename=os.path.join(demo_dir, 'image_{}_{}_label.png'.format(batch_no, i)), add_colormap=True)
        save_annotation(label=predictions[i], filename=os.path.join(demo_dir, 'image_{}_{}_prediction.png'.format(batch_no, i)), add_colormap=True)


def count_label_prediction_matches(labels, predictions, num_classes, ignore_label):
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


# def learning_rate_policy(iteration, max_iteration, power = 0.9):
#     return (1 - iteration / max_iteration) ** power
