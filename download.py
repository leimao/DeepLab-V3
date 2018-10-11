'''
Download Datasets for DeepLab

Lei Mao
Department of Computer Science
University of Chicago
dukeleimao@gmail.com

Shengjie Lin
Toyota Technological Institute at Chicago
slin@ttic.edu
'''

import argparse
import os

from data_utils import download, extract


def download_voc2012(downloads_dir='data/downloads/', data_dir='data/datasets/', force=False):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    filepath = download(url, downloads_dir, force=force)
    extract(filepath, 'tar', data_dir, force=force)


def download_sbd(downloads_dir='data/downloads/', data_dir='data/datasets/SBD/', force=False):
    '''
    http://home.bharathh.info/pubs/codes/SBD/download.html
    '''

    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
    train_noval_url = 'http://home.bharathh.info/pubs/codes/SBD/train_noval.txt'

    filepath = download(url, downloads_dir, filenames='SBD.tgz', force=force)
    download(train_noval_url, data_dir, force=force)

    extract(filepath, 'tar', data_dir, force=force)


def download_cityscapes(downloads_dir='data/downloads/cityscapes/', data_dir='data/datasets/cityscapes/', force=False):
    '''
    Does basically the same thing as following bash commands:
    curl -c cs_cookies -d "username=StArchon&password=eUpMJjMW4mbEUjZ&submit=Login" https://www.cityscapes-dataset.com/login/
    curl -b cs_cookies -JLO https://www.cityscapes-dataset.com/file-handling/?packageID=1
    '''

    urls = ['https://www.cityscapes-dataset.com/file-handling/?packageID={}'.format(id) for id in [1, 2, 3, 4, 10, 11]]
    login_dict = {'url': 'https://www.cityscapes-dataset.com/login/', 'payload': {'username': 'StArchon', 'password': 'eUpMJjMW4mbEUjZ', 'submit': 'Login'}}

    filepaths = download(urls, downloads_dir, login_dict=login_dict, force=force)
    for filepath in filepaths:
        extract(filepath, 'zip', data_dir, force=force)


def download_pretrained_models(models, downloads_dir='data/downloads/', model_dir='data/models/pretrained/', force=False):
    '''
    Download ImageNet pre-trained models
    https://github.com/tensorflow/models/tree/master/research/slim
    ResNet-50: [224, 224, 3]
    ResNet-101: [513, 513, 3]
    '''

    urls = {
        'resnet_50': 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
        'resnet_101': 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
        'mobilenet_1.0_224': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz'
    }

    for model in models:
        print('Downloading pretrained {}'.format(model))
        filepath = download(urls[model], downloads_dir, force=force)
        extract(filepath, 'tar', os.path.join(model_dir, model), force=force)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download DeepLab semantic segmentation datasets and pretrained backbone models.')

    downloads_dir_default = 'data/downloads/'
    data_dir_default = 'data/datasets/'
    pretrained_models_dir_default = 'data/models/pretrained/'
    pretrained_models_default = ['resnet_50', 'resnet_101', 'mobilenet_1.0_224']

    parser.add_argument('--downloads_dir', type=str, help='Downloads directory', default=downloads_dir_default)
    parser.add_argument('--data_dir', type=str, help='Data directory', default=data_dir_default)
    parser.add_argument('--pretrained_models_dir', type=str, help='Pretrained models directory', default=pretrained_models_dir_default)
    parser.add_argument('--pretrained_models', type=str, nargs='+', help='Pretrained models to download: resnet_50, resnet_101, mobilenet_1.0_224', default=pretrained_models_default)
    parser.add_argument('--force', help='force downloading and extracting files that are already present', default=False, action='store_true')

    argv = parser.parse_args()

    downloads_dir = argv.downloads_dir
    data_dir = argv.data_dir
    pretrained_models_dir = argv.pretrained_models_dir
    pretrained_models = argv.pretrained_models
    force = argv.force

    print('Downloading datasets...')
    download_voc2012(downloads_dir=downloads_dir, data_dir=data_dir, force=force)
    download_sbd(downloads_dir=downloads_dir, data_dir=os.path.join(data_dir, 'SBD'), force=force)
    download_cityscapes(downloads_dir=os.path.join(downloads_dir, 'cityscapes/'), data_dir=os.path.join(data_dir, 'cityscapes'), force=force)
    print('Downloading pre-trained models...')
    download_pretrained_models(models=pretrained_models, downloads_dir=downloads_dir, model_dir=pretrained_models_dir, force=force)
