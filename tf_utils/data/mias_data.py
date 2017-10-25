import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
from PIL import Image
import os.path
import os
import sys
import re
import zipfile
import numpy as np
import h5py


def load():
    data_dir = os.getcwd() + "/data/32_patches.zip"

    with zipfile.ZipFile(data_dir, 'r', zipfile.ZIP_DEFLATED) as zf:
        image_files = [f for f in zf.namelist()]
        image_files = sorted(image_files)
        image_files = list(filter(lambda f: f.endswith('.png'), image_files))

        num_images = len(image_files)

        image_data = np.ndarray((num_images, 32, 32), dtype='float32')
        trainy = np.zeros((num_images, 1), dtype='uint8')

        for i, f in enumerate(image_files):
            image = Image.open(zf.open(f, 'r'))
            image = np.asarray(image, dtype='float32')
            # image = np.expand_dims(image, axis=2)
            image_data[i] = image
            print('%d / %d' % (i + 1, num_images), end='\r', flush=True)

    data = image_data.reshape((image_data.shape[0], 1, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    data = data.astype(np.float32)
    # data = data / 255.

    trainx = data
    trainy = trainy
    return trainx, trainy


if __name__ == '__main__':
    trainx, trainy = load()
    print(trainx.shape, trainy.shape, trainx.max(), trainx.min(), trainy.max(), trainy.min())
