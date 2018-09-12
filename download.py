
'''
Download Datasets for DeepLab

Lei Mao
Department of Computer Science
University of Chicago

dukeleimao@gmail.com
'''

import os
import tarfile
import zipfile
from urllib.request import urlretrieve

from tqdm import tqdm


class TqdmUpTo(tqdm):
    '''
    Reference:
    https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    '''

    def update_to(self, b=1, bsize=1, tsize=None):

        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(filename, url, destination_dir, expected_bytes=None, force=False):

    filepath = os.path.join(destination_dir, filename)

    if force or not os.path.exists(filepath):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        print('Attempting to download: ' + filename)

        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
            urlretrieve(url, filename=filepath, reporthook=t.update_to)

        print('Download complete!')

    statinfo = os.stat(filepath)

    if expected_bytes:
        if statinfo.st_size == expected_bytes:
            print('Found and verified: ' + filename)
        else:
            raise Exception('Failed to verify: ' + filename + '. Can you get to it with a browser?')
    else:
        print('Found: ' + filename)
        print('The size of the file: ' + str(statinfo.st_size))

    return filepath


def download_voc2012(downloads_dir='./downloads', data_dir='./data'):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    filepath = maybe_download(filename=url.split('/')[-1], url=url, destination_dir=downloads_dir, expected_bytes=None, force=False)

    may_untar(tar_filepath=filepath, destination_dir=data_dir)


def may_untar(tar_filepath, destination_dir):

    print(f'Extracting tar file {os.path.split(tar_filepath)[-1]} ...')
    with tarfile.open(name=tar_filepath, mode='r') as tar:
        tar.extractall(path=destination_dir)
    print('Extraction complete!')


def maybe_unzip(zip_filepath, destination_dir):

    print(f'Extracting zip file: {os.path.split(zip_filepath)[-1]} ...')
    with zipfile.ZipFile(zip_filepath) as zf:
        zf.extractall(destination_dir)
    print('Extraction complete!')


def download_pre_trained_models(models, downloads_dir='./downloads', model_dir='./models/pretrained'):
    '''
    Download ImageNet pre-trained models
    https://github.com/tensorflow/models/tree/master/research/slim
    ResNet-50: [224, 224, 3]
    ResNet-101: [513, 513, 3]
    '''

    url_dict = {
        'resnet_50': 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
        'resnet_101': 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
        'mobilenet_1.0_224': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz'
    }

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    for model in models:
        url = url_dict[model]
        filepath = maybe_download(filename=url.split('/')[-1], url=url, destination_dir=downloads_dir, expected_bytes=None, force=False)
        may_untar(tar_filepath=filepath, destination_dir=os.path.join(model_dir, model))


if __name__ == '__main__':

    print('Downloading datasets ...')
    download_voc2012()
    print('Downloading pre-trained models ...')
    download_pre_trained_models({'resnet_50', 'resnet_101', 'mobilenet_1.0_224'})
