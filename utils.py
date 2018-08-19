
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

def image_channel_means(image_filenames):
    '''
    Calculate the means of RGB channels in image dataset.
    Support extremely large images of different sizes and arbitrary large number of images.
    image_filenames: list of image filenames
    '''

    num_pixels = 0
    channel_sums = [0, 0, 0]

    num_images = len(image_filenames)

    for i in tqdm(range(num_images)):

        image = cv2.imread(image_filenames[i])
        channel_sum = np.sum(image, axis = (0, 1))
        channel_sums[0] += channel_sum[0]
        channel_sums[1] += channel_sum[1]
        channel_sums[2] += channel_sum[2]
        num_pixels += image.shape[0] * image.shape[1]

    channel_means = [channel_sums[0] // num_pixels, channel_sums[1] // num_pixels, channel_sums[2] // num_pixels]
    channel_means = np.asarray(channel_means)

    return channel_means


def save_load_means(means_filename, image_filenames, recalculate = False):
    '''
    Calculate and save the means of RGB channels in image dataset if the mean file does not exist. 
    Otherwise read the means directly from the mean file.
    means_filename: npz filename for image channel means
    image_filenames: list of image filenames
    recalculate: recalculate image channel means regardless the existence of mean file
    '''

    if (not os.path.isfile(means_filename)) or recalculate == True:
        print('Calculating pixel means for each channel of images ...')
        channel_means = image_channel_means(image_filenames = image_filenames)
        np.savez(means_filename, channel_means = channel_means)
    else:
        channel_means = np.load(means_filename)['channel_means']

    return channel_means


class Dataset(object):

    def __init__(self, dataset_filename, images_dir, labels_dir, image_extension = '.jpg', label_extension = '.png'):

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

    def __init__(self, dataset, minibatch_size, process_func, random_seed = None, scramble = True, num_jobs = 2):

        self.dataset_size = dataset.size
        self.minibatch_size = minibatch_size
        if self.minibatch_size > self.dataset_size:
            print('Warning: dataset size should be greater or equal to minibatch size.')
            print('Set minibatch size equal to dataset size.')
            self.minibatch_size = self.dataset_size
        self.image_filenames, self.label_filenames = self.read_dataset(dataset = dataset, scramble = scramble, random_seed = random_seed)
        self.current_index = 0
        self.process_func = process_func
        self.num_jobs = num_jobs

    def read_dataset(self, dataset, scramble, random_seed):

        if random_seed is not None:
            np.random.seed(random_seed)
        idx = np.arange(self.dataset_size)
        if scramble == True:
            np.random.shuffle(idx)
        image_filenames = dataset.image_filenames[idx]
        label_filenames = dataset.label_filenames[idx]

        return image_filenames, label_filenames

    def reset_index(self):

        self.current_index = 0

    def shuffle_dataset(self, random_seed = None):

        self.current_index = 0
        idx = np.arange(self.dataset_size)
        if scramble == True:
            np.random.shuffle(idx)
        self.image_filenames = self.image_filenames[idx]
        self.label_filenames = self.label_filenames[idx]

    def next_minibatch(self):

        image_filenames_minibatch = self.image_filenames[self.current_index : self.current_index + self.minibatch_size]
        label_filenames_minibatch = self.label_filenames[self.current_index : self.current_index + self.minibatch_size]
        self.current_index += self.minibatch_size
        if self.current_index > self.dataset_size:
            self.current_index = 0

        # Multithread image processing
        # Reference: https://www.kaggle.com/inoryy/fast-image-pre-process-in-parallel
        results = Parallel(n_jobs = self.num_jobs)(delayed(self.process_func)((image_filename, label_filename)) for image_filename, label_filename in zip(image_filenames_minibatch, label_filenames_minibatch))
        images, labels = zip(*results)
        images = np.asarray(images)
        labels = np.asarray(labels)
        
        return results, images


def read_image(image_filename):

    image = cv2.imread(image_filename)

    return image

def read_label(label_filename):

    # Magic function to read VOC2012 semantic labels
    # https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py#L42
    label = np.asarray(Image.open(label_file))
    label = np.expand_dims(label, axis = 2)

    return label

def subtract_channel_means(image, channel_means):

    image_normalized = image - np.reshape(channel_means, (1,1,3))

    return image_normalized



class DataPrerocess(object):

    def __init__(self, height, width, channel_means):

        self.height = height
        self.width = width
        self.channel_means = channel_means

    def preprocess_data(image_filename, label_filename):
        # Read data from file
        image = read_image(image_filename = image_filename)
        label = read_label(label_filename = label_filename)
        # Image normalization
        image = subtract_channel_means(image = image, channel_means = self.channel_means)









def read_data_voc2012(image_filename, label_filename):

    image = read_image(image_filename = image_filename)
    label = read_label(label_filename = label_filename)


    return image, label


