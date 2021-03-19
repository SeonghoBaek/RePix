# ==============================================================================
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================

import math
import numpy as np
import tensorflow as tf
import errno
import os
import copy


def get_batch(X, X_, size):
    # X, X_ must be nd-array
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def get_sequence_batch(X, seq_length, batch_size):
    # print('input dim:', len(X[0]), ', seq len:', seq_length, ', batch_size:', batch_size)
    # X must be nd-array
    a = np.random.choice(len(X)-seq_length, batch_size, replace=False)
    a = a + seq_length

    # print('index: ', a)

    seq = []

    for i in range(batch_size):
        if a[i] < seq_length - 1:
            s = np.random.normal(0.0, 0.1, [seq_length, len(X[0])])
            seq.append(s)
        else:
            s = np.arange(a[i]-seq_length, a[i])
            seq.append(X[s])

    seq = np.array(seq)

    return X[a], seq


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass


def sigmoid_normalize(value_list):
    list_max = float(max(value_list))
    alist = [i/list_max for i in value_list]
    alist = [1/(1+math.exp(-i)) for i in alist]

    return alist


def swish(logit,  name=None):
    with tf.name_scope(name):
        l = tf.multiply(logit, tf.nn.sigmoid(logit))

        return l


def generate_samples(dim, num_inlier, num_outlier, normalize=True):
    inlier = np.random.normal(0.0, 1.0, [num_inlier, dim])

    sample_inlier = []

    if normalize:
        inlier = np.transpose(inlier)

        for values in inlier:
            values = sigmoid_normalize(values)
            sample_inlier.append(values)

        inlier = np.array(sample_inlier).transpose()

    outlier = np.random.normal(1.0, 1.0, [num_outlier, dim])

    sample_outlier = []

    if normalize:
        outlier = np.transpose(outlier)

        for values in outlier:
            values = sigmoid_normalize(values)
            sample_outlier.append(values)

        outlier = np.array(sample_outlier).transpose()

    return inlier, outlier


def add_gaussian_noise(input_layer, mean, std):
    if std < 0.0:
        return input_layer

    noise = tf.random_normal(shape=input_layer.get_shape().as_list(), mean=mean, stddev=std, dtype=tf.float32)
    return tf.add(input_layer, noise)


def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def patch_compare(img1, img2, patch_size=[4, 4]):
    img1_h, img1_w = img1.shape
    img2_h, img2_w = img2.shape
    img_w = np.min([img1_w, img2_w])
    img_h = np.min([img1_h, img2_h])

    patch_h, patch_w = patch_size
    num_patch_h = img_h // patch_h
    num_patch_w = img_w // patch_w

    diff_vector = []

    img1 = img1 // 255
    img2 = img2 // 255

    for h in range(num_patch_h):
        for w in range(num_patch_w):
            patch1 = img1[h * patch_h:h * patch_h + patch_h, w * patch_w: w * patch_w + patch_w]
            patch2 = img2[h * patch_h:h * patch_h + patch_h, w * patch_w: w * patch_w + patch_w]
            d = np.abs(patch1 - patch2)
            d = np.sum(d)
            diff_vector.append(d)

    return diff_vector


class ImagePool(object):
    def __init__(self, maxsize=50, threshold=0.5):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []
        self.threshold = threshold

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image

        if np.random.rand() > self.threshold:
            idx = int(np.random.rand() * self.maxsize)
            tmp = copy.copy(self.images[idx])
            self.images[idx] = image
            return tmp
        else:
            return image


class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
