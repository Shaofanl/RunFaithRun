import numpy as np
import tensorflow as tf
import cv2

# from RunFaithRun.learning.models.simple_cnn import simple_cnn
from RunFaithRun.learning.models.mask_cnn import mask_cnn
from RunFaithRun.utils.data_utils import load_image_from_dir


def vis(name, imgs):
    if imgs.ndim == 4:
        imgs = np.concatenate(imgs, axis=1)
    if imgs.shape[-1] == 3:
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.imshow(name, imgs)
    else:
        cv2.imshow(name, imgs)


if __name__ == '__main__':
    data = np.load('./space_data.npz')
    images, _ = load_image_from_dir(data['filedir'].flatten()[0], resize=(176, 296))

    _, height, width, channel = images.shape
    images_trans = images.transpose(0, 3, 1, 2)
    X, y = data['x'], data['y']
    pos_ind, = np.nonzero(y == 0)
    neg_ind, = np.nonzero(y == 1)
    from code_train import generator
    gen = generator(images_trans, X, y, pos_ind, neg_ind)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()

    model, mask_model = mask_cnn(
        input_shape=(height, width, channel*X.shape[1]),
        l2coef=0)
    model.load_weights('./models/model.h')

    while True:
        imgs, _ = gen.__next__()
        imgs = imgs[0]

        mask = mask_model.predict(np.expand_dims(imgs, 0))[0]
        vis('mask', mask)

        vis('rgb', imgs.transpose(2, 0, 1).\
            reshape(3, 3, imgs.shape[0], imgs.shape[1]).\
            transpose(0, 2, 3, 1))

        print('mask', mask.mean(), mask.min(), mask.max())
        print('imgs', imgs.mean(), imgs.min(), imgs.max())
        imgs = (imgs*mask).astype('uint8')
        print('imgs', imgs.mean(), imgs.min(), imgs.max())
        vis('rgb masked', imgs.transpose(2, 0, 1).\
            reshape(3, 3, imgs.shape[0], imgs.shape[1]).\
            transpose(0, 2, 3, 1))

        if cv2.waitKey(0) == ord('q'):
            break
