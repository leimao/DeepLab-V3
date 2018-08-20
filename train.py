

from utils import Dataset, Iterator, DataPrerocessor
from utils import save_load_means

from model import DeepLab




def train(train_dataset_filename = './data/VOCdevkit/VOC2012/train_dataset.txt', valid_dataset_filename = './data/VOCdevkit/VOC2012/valid_dataset.txt', test_dataset_filename = './data/VOCdevkit/VOC2012/test_dataset.txt', images_dir = './data/VOCdevkit/VOC2012/JPEGImages', labels_dir = './data/VOCdevkit/VOC2012/SegmentationClass', pre_trained_model = './models/resnet_101/resnet_v2_101.ckpt'):

    # Prepare datasets
    train_dataset = Dataset(dataset_filename = train_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')
    valid_dataset = Dataset(dataset_filename = valid_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')
    test_dataset = Dataset(dataset_filename = test_dataset_filename, images_dir = images_dir, labels_dir = labels_dir, image_extension = '.jpg', label_extension = '.png')

    # Calculate image channel means
    channel_means = save_load_means(means_filename = './models/channel_means.npz', image_filenames = train_dataset.image_filenames, recalculate = False)

    voc2012_preprocessor = DataPrerocessor(channel_means = channel_means, output_size = [513, 513], scale_factor = 1.5)

    # Prepare dataset iterators
    train_iterator = Iterator(dataset = train_dataset, minibatch_size = 1, process_func = voc2012_preprocessor.preprocess, random_seed = None, scramble = True, num_jobs = 1)
    valid_iterator = Iterator(dataset = valid_dataset, minibatch_size = 1, process_func = voc2012_preprocessor.preprocess, random_seed = None, scramble = False, num_jobs = 1)
    test_iterator = Iterator(dataset = train_dataset, minibatch_size = 1, process_func = voc2012_preprocessor.preprocess, random_seed = None, scramble = False, num_jobs = 1)

    model = DeepLab(is_training = True, num_classes = 21, image_shape = [513, 513, 3], base_architecture = 'resnet_v2_101', batch_norm_decay = 0.9997, pre_trained_model = pre_trained_model)

    #images, labels = train_iterator.next_minibatch()
    #print(images.dtype, labels.dtype)
    for i in range(100):
        images, labels = train_iterator.next_minibatch()
        outputs, train_loss = model.train(inputs = images, labels = labels, learning_rate = 0.001)
        print('==================================')
        print(train_loss)




if __name__ == '__main__':
    
    train()

