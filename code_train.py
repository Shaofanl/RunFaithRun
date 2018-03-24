import numpy as np

import tensorflow as tf
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

from RunFaithRun.learning.models.simple_cnn import simple_cnn
from RunFaithRun.learning.models.mask_cnn import mask_cnn
from RunFaithRun.utils.data_utils import load_image_from_dir


def generator(image, dataX, datay, pos_ind, neg_ind, batch_size=32):
    while True:
        pos_sample_ind = np.random.choice(pos_ind, size=(batch_size//2))
        neg_sample_ind = np.random.choice(neg_ind, size=(batch_size//2))
        samples = np.concatenate([dataX[pos_sample_ind],
                                  dataX[neg_sample_ind]])
        samples = image[samples.flatten()].reshape(
            samples.shape[0], image.shape[1]*samples.shape[1],
            image.shape[2], image.shape[3]).transpose(0, 2, 3, 1)
        labels = np.concatenate([datay[pos_sample_ind],
                                 datay[neg_sample_ind]])
        # [0]*(batch_size//2) + [1]*(batch_size//2)
        yield samples, labels


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.__enter__()

    data = np.load('./space_data.npz')
    images, _ = load_image_from_dir(data['filedir'].flatten()[0], resize=(176, 296))
    _, height, width, channel = images.shape
    images_trans = images.transpose(0, 3, 1, 2)
    X, y = data['x'], data['y']
    X, validX, y, validy = train_test_split(X, y, test_size=0.33)
    print(y)
    print(X.shape)
    print(y.mean())
    pos_ind, = np.nonzero(y == 0)
    neg_ind, = np.nonzero(y == 1)
    valid_pos_ind, = np.nonzero(validy == 0)
    valid_neg_ind, = np.nonzero(validy == 1)

    # model, _ = mask_cnn(
    #     input_shape=(height, width, channel*X.shape[1]), l2coef=0)
    model = simple_cnn(input_shape=(height, width, channel*X.shape[1]))
    # sgd = SGD(lr=1e-5,
    #         decay=1e-6,
    #         momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',
                  metrics=['binary_accuracy'],
                  optimizer=opt)
    model.load_weights('models/model.h')

    print(validy.mean())
    model.fit_generator(
        generator(images_trans, X, y, pos_ind, neg_ind), 50, epochs=50, verbose=1,
        validation_data=generator(images_trans, validX, validy, valid_pos_ind, valid_neg_ind),
        validation_steps=20,
    )
        # validation_data=(validX, validy))
    model.save('models/model.h')
