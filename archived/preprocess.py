
import os

import numpy as np


def train_val_test_split(dataset_filenames, split_ratios, train_dataset_filename, valid_dataset_filename, test_dataset_filename):
    '''
    Split dataset into train, valid, and test datasets
    dataset_filenames: a list of image filenames
    split_ratios: [train_dataset_ratio, valid_dataset_ratio, test_dataset_ratio], e.g., [0.7, 0.2, 0.1]
    train_dataset_filename: path of txt file to save the filenames of train data
    valid_dataset_filename: path of txt file to save the filenames of valid data
    test_dataset_filename: path of txt file to save the filenames of test data
    '''

    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1) < 1e-5, 'Please use all the data.'

    dataset_filenames = np.asarray(dataset_filenames)
    idx = np.arange(len(dataset_filenames))
    np.random.shuffle(idx)
    train_split_idx = int(len(dataset_filenames) * split_ratios[0])
    valid_split_idx = int(len(dataset_filenames) * (split_ratios[0] + split_ratios[1]))

    train_idx = idx[:train_split_idx]
    valid_idx = idx[train_split_idx:valid_split_idx]
    test_idx = idx[valid_split_idx:]

    train_filenames = dataset_filenames[train_idx]
    valid_filenames = dataset_filenames[valid_idx]
    test_filenames = dataset_filenames[test_idx]

    with open(train_dataset_filename, 'w') as file:
        file.write('\n'.join(train_filenames))
    with open(valid_dataset_filename, 'w') as file:
        file.write('\n'.join(valid_filenames))
    with open(test_dataset_filename, 'w') as file:
        file.write('\n'.join(test_filenames))


def voc2012_split(dataset_dir='data/datasets/VOCdevkit/VOC2012/', split_ratios=[0.7, 0.2, 0.1]):

    images_dir = os.path.join(dataset_dir, 'JPEGImages/')
    labels_dir = os.path.join(dataset_dir, 'SegmentationClass/')

    image_filenames = [filename.split('.')[0] for filename in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, filename)) and filename.endswith('.jpg')]
    label_filenames = [filename.split('.')[0] for filename in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, filename)) and filename.endswith('.png')]

    dataset_filenames = np.intersect1d(image_filenames, label_filenames)

    train_dataset_filename = os.path.join(dataset_dir, 'train.txt')
    valid_dataset_filename = os.path.join(dataset_dir, 'val.txt')
    test_dataset_filename = os.path.join(dataset_dir, 'test.txt')

    try:
        train_val_test_split(
            dataset_filenames=dataset_filenames,
            split_ratios=split_ratios,
            train_dataset_filename=train_dataset_filename,
            valid_dataset_filename=valid_dataset_filename,
            test_dataset_filename=test_dataset_filename)
    except BaseException:
        raise Exception('Dataset split failed.')

    return train_dataset_filename, valid_dataset_filename, test_dataset_filename


if __name__ == '__main__':

    voc2012_split()
