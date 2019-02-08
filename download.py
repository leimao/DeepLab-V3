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
import tarfile
import zipfile

import requests

from tqdm import tqdm
from utils import static_vars


def _download(sess, url, destination_dir, filename, expected_bytes, force):
    r = sess.get(url, stream=True)
    if not filename:
        if 'Content-Disposition' in r.headers:
            filename = r.headers['Content-Disposition'].split('filename=')[1]
            if filename[0] in {"'", '"'}:
                filename = filename[1:-1]
        else:
            filename = url.split('/')[-1]
    filepath = os.path.join(destination_dir, filename)
    if not os.path.exists(filepath) or expected_bytes and os.stat(filepath).st_size != expected_bytes or force:
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        print('Downloading: ' + filename)
        chunk_size = 8192
        total_size = int(r.headers.get('Content-Length', 0))
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename)
        with open(os.path.join(destination_dir, filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()
        print('Download complete!')
        actual_bytes = os.stat(filepath).st_size
        if expected_bytes:
            if actual_bytes == expected_bytes:
                print('Verified: ' + filename)
            else:
                print('Failed to verify: ' + filename + '. File size does not match!')
        else:
            print('Fize size: ' + str(actual_bytes))
    else:
        if expected_bytes:
            print('Found and verified: ' + filename)
        else:
            print('Found: ' + filename)
    return filepath


def download(urls, destination_dir, filenames=None, expected_bytes=None, login_dict=None, force=False):
    with requests.Session() as sess:
        if login_dict:
            sess.post(login_dict['url'], data=login_dict['payload'])
        if isinstance(urls, str):
            return _download(sess, urls, destination_dir, filenames, expected_bytes, force)
        n_urls = len(urls)
        if filenames:
            assert not isinstance(filenames, str) and len(filenames) == n_urls, 'number of filenames does not match that of urls'
        else:
            filenames = [None] * n_urls
        if expected_bytes:
            assert len(expected_bytes) == n_urls, 'number of expected_bytes does not match that of urls'
        else:
            expected_bytes = [None] * n_urls
        return [_download(sess, url, destination_dir, filename, expected_byte, force) for url, filename, expected_byte in zip(urls, filenames, expected_bytes)]


@static_vars(utils_dict={'zip': (zipfile.ZipFile, 'namelist'), 'tar': (tarfile.open, 'getnames')})
def extract(archive_filepath, archive_type, destination_dir, force=False):

    print('Extracting {} file: {}'.format(archive_type, os.path.split(archive_filepath)[-1]))
    utils = extract.utils_dict[archive_type]
    with utils[0](archive_filepath) as archive:
        for name in getattr(archive, utils[1])():
            if not os.path.exists(os.path.join(destination_dir, name)) or force:
                archive.extract(name, path=destination_dir)
    print('Extraction complete!')


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
    #download_cityscapes(downloads_dir=os.path.join(downloads_dir, 'cityscapes/'), data_dir=os.path.join(data_dir, 'cityscapes'), force=force)
    print('Downloading pre-trained models...')
    download_pretrained_models(models=pretrained_models, downloads_dir=downloads_dir, model_dir=pretrained_models_dir, force=force)
