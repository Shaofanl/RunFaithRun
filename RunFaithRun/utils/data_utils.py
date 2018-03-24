import os
import numpy as np
from scipy.misc import imread, imresize


def load_image_from_dir(filedir, resize=None):
    filenames = sorted(list(filter(
        lambda x: x.endswith('jpg'),
        os.listdir(filedir))))

    images = []
    images_time = []
    for filename in filenames:
        img = imread(os.path.join(filedir, filename))
        if resize is not None:
            img = imresize(img, resize)
        images.append(img)
        images_time.append(float(filename.rsplit('.', 1)[0]))
    images = np.array(images)
    images_time = np.array(images_time)
    return images, images_time
