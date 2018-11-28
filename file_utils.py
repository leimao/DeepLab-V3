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
    if not os.path.isfile(filepath) or expected_bytes and os.stat(filepath).st_size != expected_bytes or force:
        if not os.path.isdir(destination_dir):
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
def decompress(archive_filepath, archive_type, destination_dir, force=False):
    print('Decompressing {} file: {}'.format(archive_type, os.path.split(archive_filepath)[-1]))
    utils = decompress.utils_dict[archive_type]
    with utils[0](archive_filepath) as archive:
        for name in tqdm(getattr(archive, utils[1])()):
            if not os.path.exists(os.path.join(destination_dir, name)) or force:
                archive.extract(name, path=destination_dir)
    print('Decompression complete!')
