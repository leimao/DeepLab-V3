
import os
import time

import numpy as np
from tqdm import tqdm

import cv2
from joblib import Parallel, delayed
from PIL import Image


def image_channel_means(image_filenames):
    '''
    Calculate the means of RGB channels in image dataset.
    Support extremely large images of different sizes and arbitrarily large number of images.
    image_filenames: list of image filenames
    '''

    num_pixels = 0
    channel_sums = np.zeros(3, dtype=object)

    for image_filename in tqdm(image_filenames):
        image = cv2.imread(image_filename)
        channel_sums += np.sum(image, axis=(0, 1))
        num_pixels += np.prod(image.shape[:2])

    channel_means = (channel_sums / num_pixels).astype(float)

    return channel_means


def save_load_means(means_filename, image_filenames, recalculate=False):
    '''
    Calculate and save the means of RGB channels in image dataset if the mean file does not exist.
    Otherwise read the means directly from the mean file.
    means_filename: npz filename for image channel means
    image_filenames: list of image filenames
    recalculate: recalculate image channel means regardless the existence of mean file
    '''

    if (not os.path.isfile(means_filename)) or recalculate:
        print('Calculating pixel means for each channel of images ...')
        channel_means = image_channel_means(image_filenames=image_filenames)
        np.savez(means_filename, channel_means=channel_means)
    else:
        channel_means = np.load(means_filename)['channel_means']

    return channel_means


class RandomStateStack:
    def __init__(self):
        self.random_state = np.random.get_state()

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.random_state)


class Dataset(object):

    def __init__(self, dataset_filename, images_dir, labels_dir, image_extension='.jpg', label_extension='.png'):

        self.dataset_filename = dataset_filename
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_extension = image_extension
        self.label_extension = label_extension
        self.image_filenames, self.label_filenames = self.read_dataset()
        self.size = len(self.image_filenames)

    def read_dataset(self):

        image_filenames = []
        label_filenames = []

        with open(self.dataset_filename, 'r') as file:
            for line in file:
                filename = line.strip()
                image_filename = os.path.join(self.images_dir, filename + self.image_extension)
                label_filename = os.path.join(self.labels_dir, filename + self.label_extension)
                image_filenames.append(image_filename)
                label_filenames.append(label_filename)

        image_filenames = np.asarray(image_filenames)
        label_filenames = np.asarray(label_filenames)

        return image_filenames, label_filenames


class Iterator(object):

    def __init__(self, dataset, minibatch_size, process_func, random_seed=None, scramble=True, num_jobs=2):

        self.dataset_size = dataset.size
        self.minibatch_size = minibatch_size
        if self.minibatch_size > self.dataset_size:
            print('Warning: dataset size should be no less than minibatch size.')
            print('Set minibatch size equal to dataset size.')
            self.minibatch_size = self.dataset_size
        self.image_filenames, self.label_filenames = self.read_dataset(dataset=dataset, scramble=scramble, random_seed=random_seed)
        self.current_index = 0
        self.process_func = process_func
        self.num_jobs = num_jobs

    def read_dataset(self, dataset, scramble, random_seed):

        idx = np.arange(self.dataset_size)
        if scramble:
            with RandomStateStack():
                if random_seed is not None:
                    np.random.seed(random_seed)
                np.random.shuffle(idx)
        image_filenames = dataset.image_filenames[idx]
        label_filenames = dataset.label_filenames[idx]

        return image_filenames, label_filenames

    def reset_index(self):

        self.current_index = 0

    def shuffle_dataset(self, random_seed=None):

        self.current_index = 0
        idx = np.arange(self.dataset_size)
        with RandomStateStack():
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(idx)
        self.image_filenames = self.image_filenames[idx]
        self.label_filenames = self.label_filenames[idx]

    def next_minibatch(self):

        image_filenames_minibatch = self.image_filenames[self.current_index: self.current_index + self.minibatch_size]
        label_filenames_minibatch = self.label_filenames[self.current_index: self.current_index + self.minibatch_size]
        self.current_index += self.minibatch_size
        if self.current_index > self.dataset_size:
            self.current_index = 0

        # Multithread image processing
        # Reference: https://www.kaggle.com/inoryy/fast-image-pre-process-in-parallel

        results = Parallel(n_jobs=self.num_jobs)(delayed(self.process_func)(image_filename, label_filename)
                                                 for image_filename, label_filename in zip(image_filenames_minibatch, label_filenames_minibatch))
        images, labels = zip(*results)

        images = np.asarray(images)
        labels = np.asarray(labels)

        return images, labels


def read_image(image_filename):

    image = cv2.imread(image_filename)

    return image


def read_label(label_filename):

    # Magic function to read VOC2012 semantic labels
    # https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py#L42
    label = np.asarray(Image.open(label_filename))
    label = np.expand_dims(label, axis=2)

    return label


def subtract_channel_means(image, channel_means):

    return image - np.reshape(channel_means, (1, 1, 3))


def add_channel_means(image, channel_means):

    return image + np.reshape(channel_means, (1, 1, 3))


def flip_image(image, label):

    image_flipped = np.fliplr(image)
    label_flipped = np.fliplr(label)

    return image_flipped, label_flipped


def resize_image(image, label, output_size):
    '''
    output_size: [height, width]
    '''

    image_resized = cv2.resize(image, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, (output_size[1], output_size[0]), interpolation=cv2.INTER_NEAREST)
    label_resized = np.expand_dims(label_resized, axis=2)

    return image_resized, label_resized


def random_crop(image, label, output_size):

    assert image.shape[0] >= output_size[0] and image.shape[1] >= output_size[1], 'image size smaller than the desired output size.'

    height_start = np.random.randint(image.shape[0] - output_size[0] + 1)
    width_start = np.random.randint(image.shape[1] - output_size[1] + 1)
    height_end = height_start + output_size[0]
    width_end = width_start + output_size[1]

    image_cropped = image[height_start:height_end, width_start:width_end]
    label_cropped = label[height_start:height_end, width_start:width_end]

    return image_cropped, label_cropped


def image_augmentaion(image, label, output_size, scale_factor=1.5):

    original_height = image.shape[0]
    original_width = image.shape[1]
    target_height = output_size[0]
    target_width = output_size[1]

    image_augmented = image.copy()
    label_augmented = label.copy()

    if original_height >= int(scale_factor * target_height) and original_width >= int(scale_factor * target_width):
        image_augmented, label_augmented = random_crop(image=image_augmented, label=label_augmented, output_size=output_size)
    else:
        rescaled_size = [np.random.randint(target_height, int(scale_factor * target_height)), np.random.randint(target_width, int(scale_factor * target_width))]
        image_augmented, label_augmented = resize_image(image=image_augmented, label=label_augmented, output_size=rescaled_size)
        image_augmented, label_augmented = random_crop(image=image_augmented, label=label_augmented, output_size=output_size)

    # Flip image and label
    if np.random.random() > 0.5:
        image_augmented, label_augmented = flip_image(image=image_augmented, label=label_augmented)

    return image_augmented, label_augmented


class DataPreprocessor(object):

    def __init__(self, channel_means, output_size=[513, 513], scale_factor=1.5):

        self.channel_means = channel_means
        self.output_size = output_size
        self.scale_factor = scale_factor

    def preprocess(self, image_filename, label_filename):
        # Read data from file
        image = read_image(image_filename=image_filename)
        label = read_label(label_filename=label_filename)
        image, label = image_augmentaion(image=image, label=label, output_size=self.output_size, scale_factor=self.scale_factor)

        # Image normalization
        image = subtract_channel_means(image=image, channel_means=self.channel_means)

        return image, label



##################################################################################################################################
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

    if np.max(label) >= 256:
        raise ValueError('label value too large.')

    colormap = create_pascal_label_colormap()

    return colormap[label]


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
        colored_label = label_to_color_image(label)
    else:
        colored_label = label

    image = Image.fromarray(colored_label.astype(dtype=np.uint8))
    image.save(filename)


##################################################################################################################################
'''
Evaluation
'''


def validation_demo(images, labels, predictions, demo_dir):

    assert images.ndim == 4 and labels.ndim == 3 and predictions.ndim == 3

    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    for i in range(len(images)):

        cv2.imwrite(os.path.join(demo_dir, 'image_{}.jpg'.format(i)), images[i])
        save_annotation(label=labels[i], filename=os.path.join(demo_dir, 'image_{}_label.png'.format(i)), add_colormap=True)
        save_annotation(label=predictions[i], filename=os.path.join(demo_dir, 'image_{}_prediction.png'.format(i)), add_colormap=True)


def count_label_prediction_matches(labels, predictions, num_classes, ignore_label):
    '''
    Pixel intersection-over-union averaged across number of classes.
    Assuming valid labels are from 0 to num_classes - 1.
    '''

    assert labels.ndim == 3 and labels.shape == predictions.shape

    num_pixel_labels = np.zeros(num_classes)
    num_pixel_correct_predictions = np.zeros(num_classes)

    not_ignore_mask = np.not_equal(labels, ignore_label).astype(np.int)
    matched_pixels = (labels == predictions) * not_ignore_mask

    for i in range(num_classes):

        class_mask = (labels == i)
        num_pixel_labels[i] += np.sum(class_mask)
        num_pixel_correct_predictions[i] += np.sum(matched_pixels * class_mask)

    return num_pixel_labels, num_pixel_correct_predictions


def mean_intersection_over_union(num_pixel_labels, num_pixel_correct_predictions):

    num_classes = len(num_pixel_labels)

    num_existing_classes = 0
    iou_sum = 0

    for i in range(num_classes):

        if num_pixel_labels[i] == 0:
            continue

        num_existing_classes += 1
        iou_sum += num_pixel_correct_predictions[i] / num_pixel_labels[i]

    assert num_existing_classes != 0

    mean_iou = iou_sum / num_existing_classes

    return mean_iou


'''
def mean_intersection_over_union(labels, predictions, ignore_label):
    # Wrong implementation


    not_ignore_mask = np.not_equal(labels, ignore_label).astype(np.int)
    num_valid_labels = np.sum(not_ignore_mask)
    num_matched_labels = np.sum((labels == predictions) * not_ignore_mask)

    mIOU = num_matched_labels / num_valid_labels

    return num_matched_labels, num_valid_labels, mIOU
'''


if __name__ == '__main__':

    np.random.seed(0)

    train_dataset = Dataset(dataset_filename='./data/VOCdevkit/VOC2012/train_dataset.txt', images_dir='./data/VOCdevkit/VOC2012/JPEGImages',
                            labels_dir='./data/VOCdevkit/VOC2012/SegmentationClass', image_extension='.jpg', label_extension='.png')
    print(train_dataset.image_filenames)
    print(train_dataset.size)

    channel_means = save_load_means(means_filename='./models/channel_means.npz', image_filenames=train_dataset.image_filenames, recalculate=False)
    print(channel_means)

    voc2012_preprocessor = DataPreprocessor(channel_means=channel_means, output_size=[513, 513], scale_factor=1.5)

    # Single thread is faster :(
    train_iterator = Iterator(dataset=train_dataset, minibatch_size=16, process_func=voc2012_preprocessor.preprocess, random_seed=None, scramble=True, num_jobs=1)

    # Test iterator
    time_start = time.time()
    for i in range(10):
        print(i)
        images, labels = train_iterator.next_minibatch()
        #print(images.shape, labels.shape)
    time_end = time.time()
    time_elapsed = time_end - time_start
    print("Time Elapsed: %02d:%02d:%02d" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    '''
    images, labels = train_iterator.next_minibatch()

    print(images.dtype, labels.dtype)

    for i, (image, label) in enumerate(zip(images, labels)):
        image = add_channel_means(image = image, channel_means = channel_means)
        print(type(image), image.shape)
        print(type(label), label.shape)
        label = np.squeeze(label, axis = -1)
        print(type(label), label.shape)
        save_annotation(label = label, filename = str(i) + '.png', add_colormap = True)
        #label = Image.fromarray(image.astype(np.uint8))
        #label.save(str(i) + '.png')


        cv2.imwrite(str(i) + '.jpg', image)
    '''
