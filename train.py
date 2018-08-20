

from utils import Dataset, Iterator, DataPrerocessor
from utils import save_load_means

from model import DeepLab

from tqdm import tqdm

import os
import numpy as np




def train(train_dataset_filename = './data/VOCdevkit/VOC2012/train_dataset.txt', valid_dataset_filename = './data/VOCdevkit/VOC2012/valid_dataset.txt', test_dataset_filename = './data/VOCdevkit/VOC2012/test_dataset.txt', images_dir = './data/VOCdevkit/VOC2012/JPEGImages', labels_dir = './data/VOCdevkit/VOC2012/SegmentationClass', pre_trained_model = './models/resnet_101/resnet_v2_101.ckpt', model_dir = './models/voc2012', log_dir = './log'):

    num_classes = 21
    num_epochs = 100
    minibatch_size = 8
    random_seed = 0
    learning_rate = 0.00001
    batch_norm_decay = 0.9997
    model_filename = 'deeplab.ckpt'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prepare datasets
    train_dataset = Dataset(dataset_filename = train_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')
    valid_dataset = Dataset(dataset_filename = valid_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')
    test_dataset = Dataset(dataset_filename = test_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')

    # Calculate image channel means
    channel_means = save_load_means(means_filename = './models/channel_means.npz', image_filenames = train_dataset.image_filenames, recalculate = False)

    voc2012_preprocessor = DataPrerocessor(channel_means = channel_means, output_size = [513, 513], scale_factor = 1.5)

    # Prepare dataset iterators
    train_iterator = Iterator(dataset = train_dataset, minibatch_size = minibatch_size, process_func = voc2012_preprocessor.preprocess, random_seed = random_seed, scramble = True, num_jobs = 1)
    valid_iterator = Iterator(dataset = valid_dataset, minibatch_size = minibatch_size, process_func = voc2012_preprocessor.preprocess, random_seed = None, scramble = False, num_jobs = 1)
    test_iterator = Iterator(dataset = test_dataset, minibatch_size = minibatch_size, process_func = voc2012_preprocessor.preprocess, random_seed = None, scramble = False, num_jobs = 1)

    model = DeepLab(is_training = True, num_classes = num_classes, image_shape = [513, 513, 3], base_architecture = 'resnet_v2_101', batch_norm_decay = batch_norm_decay, pre_trained_model = pre_trained_model, log_dir = log_dir)

    for i in range(num_epochs):

        print('Epoch number: {}'.format(i))

        print('Start validation ...')

        valid_loss_total = 0

        for j in tqdm(range(np.ceil(valid_iterator.dataset_size / minibatch_size).astype(int))):

            images, labels = valid_iterator.next_minibatch()
            num_samples = len(images)
            outputs, valid_loss = model.validate(inputs = images, labels = labels)

            valid_loss_total += valid_loss * num_samples

        valid_loss_ave = valid_loss_total / valid_iterator.dataset_size

        print('Validation loss: {:.8f}'.format(valid_loss_ave))

        print('Start training ...')
        
        train_loss_total = 0
        
        for j in tqdm(range(np.ceil(train_iterator.dataset_size / minibatch_size).astype(int))):
            images, labels = train_iterator.next_minibatch()
            num_samples = len(images)
            outputs, train_loss = model.train(inputs = images, labels = labels, learning_rate = learning_rate)

            train_loss_total += train_loss * num_samples

        train_loss_ave = train_loss_total / train_iterator.dataset_size

        print('Training loss: {:.8f}'.format(train_loss_ave))

if __name__ == '__main__':
    
    train()

