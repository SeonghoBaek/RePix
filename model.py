# ==============================================================================
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================

import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse
import time
import post_proc


def load_images(file_name_list, base_dir, use_augmentation=False, add_eps=False, rotate=-1, resize=[240, 240]):
    try:
        images = []
        non_augmented = []

        for file_name in file_name_list:
            fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            h, w, c = img.shape

            if rotate > -1:
                img = cv2.rotate(img, rotate)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if h != resize[0]:
                img = cv2.resize(img, dsize=(resize[0], resize[1]), interpolation=cv2.INTER_AREA)

            if img is not None:
                img = np.array(img)

                #n_img = (img - 127.0) / 128.0
                n_img = img / 256.0

                if add_eps is True:
                    if np.random.randint(low=0, high=10) < 5:
                        n_img = n_img + np.random.uniform(low=0, high=1/256, size=n_img.shape)

                non_augmented.append(n_img)

                if use_augmentation is True:
                    # if np.random.randint(low=0, high=10) < 5:
                    # square cut out
                    co_w = input_width // 4
                    co_h = co_w
                    padd_w = co_w // 2
                    padd_h = padd_w
                    r_x = np.random.randint(low=padd_w, high=resize[0] - padd_w)
                    r_y = np.random.randint(low=padd_h, high=resize[1] - padd_h)

                    for i in range(co_w):
                        for j in range(co_h):
                            n_img[r_x - padd_w + i][r_y - padd_h + j] = 0.0
                            #n_img[r_x - padd_w + i][r_y - padd_h + j] = np.random.normal()

                images.append(n_img)
    except cv2.error as e:
        print(e)
        return None

    return np.array(images), np.array(non_augmented)


def hr_discriminator(x1, x2, activation='relu', scope='hr_discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = unit_block_depth
        norm_func = norm

        if use_conditional_d is True:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1

        print('HR Discriminator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='norm_init')
        l = act_func(l)

        norm_func = norm

        if use_patch is True:
            print('HR Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(hr_patch_discriminator_depth):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                                 norm=norm_func, b_train=b_train, scope='patch_block_1_' + str(i))

            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm_func, b_train=b_train, use_dilation=True, scope='patch_block_2_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=False,
                                             scope='gp')

            print('HR Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                                non_linear_fn=None, bias=False)
            print('HR Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            num_iter = hr_discriminator_depth
            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_1_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample1', filter_dims=[3, 3, block_depth * 2], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_1')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 2], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_2_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample2', filter_dims=[3, 3, block_depth * 4], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_2')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 4], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_3_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')

            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.fc(feature, 1, non_linear_fn=None, scope='flat')

    return feature, logit


def lr_discriminator(x1, x2, activation='relu', scope='lr_discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = unit_block_depth * 8

        norm_func = norm

        if use_conditional_d is True:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1

        print('LR Discriminator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='norm_init')
        l = act_func(l)

        norm_func = norm

        if use_patch is True:
            print('LR Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(lr_patch_discriminator_depth):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                                 norm=norm_func, b_train=b_train, scope='patch_block_1_' + str(i))

            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm_func, b_train=b_train, use_dilation=True, scope='patch_block_2_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=False,
                                             scope='gp')

            print('LR Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                                non_linear_fn=None, bias=False)
            print('LR Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            num_iter = lr_discriminator_depth

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_1_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample1', filter_dims=[3, 3, block_depth * 2], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_1')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 2], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_2_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample2', filter_dims=[3, 3, block_depth * 4], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_2')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 4], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_3_' + str(i))

            print('Discriminator Block : ' + str(l.get_shape().as_list()))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')

            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.fc(feature, 1, non_linear_fn=None, scope='flat')

    return feature, logit


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'hinge':
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ls':
        return tf.reduce_mean((real - fake) ** 2)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        # loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l1')
    a_n = get_residual_loss(anchor, negative, 'l1')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return tf.reduce_mean(a_p / a_n)


def get_gradient_loss(img1, img2):
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    #v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    #v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    # loss = tf.abs(tf.subtract(v_a, v_b))
    loss = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b))) + tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss


def hr_translator(x, lr_feature, activation='relu', scope='hr_translator', norm='layer', upsample='espcn', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = unit_block_depth  # Number of channel at start
        bottleneck_num_itr = hr_bottleneck_depth  # Num of bottleneck blocks of layers

        downsample_num_itr = int(np.log2(lr_ratio))  # Num of downsampling
        upsample_num_itr = downsample_num_itr  # Num of upsampling

        refine_num_itr = hr_refinement_depth

        print('HR Translator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        # Init Stage. Coordinated convolution: Embed explicit positional information
        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                              non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        shorcut_layers = []

        # Downsample stage.
        for i in range(downsample_num_itr):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

            if use_unet_hr is True:
                shorcut_layers.append(l)

            print('HR Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        l = tf.concat([l, lr_feature], axis=-1)
        l = layers.conv(l, scope='concat', filter_dims=[3, 3, block_depth],
                        stride_dims=[1, 1], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='concat_norm')
        l = act_func(l)

        print('HR_Concat layer: ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_num_itr):
            print('HR Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, use_dilation=False,
                                             scope='bt_block_' + str(i))

        # Upsample stage
        for i in range(upsample_num_itr):
            if upsample == 'espcn':
                if use_unet_hr is True:
                    l = tf.concat([l, shorcut_layers[upsample_num_itr-1-i]], axis=-1)
                # ESPCN upsample
                block_depth = block_depth // 2
                l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
                l = act_func(l)
                l = tf.nn.depth_to_space(l, 2)

                print('HR_Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            elif upsample == 'resize':
                # Image resize
                block_depth = block_depth // 2

                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                # l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                # l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))
                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

                for j in range(refine_num_itr):
                    l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                     act_func=act_func, norm=norm, b_train=b_train,
                                                     scope='block_' + str(i) + '_' + str(j))
            else:
                # Deconvolution
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                  filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print('Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        if use_refinement is True:
            # Refinement
            block_depth = block_depth // 2
            for i in range(refine_num_itr):
                l = layers.conv(l, scope='refine_' + str(i), filter_dims=[3, 3, block_depth],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='refine_norm_' + str(i))
                l = act_func(l)

                print('HR Refinement ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                        dilation=[1, 2, 2, 1], non_linear_fn=tf.nn.sigmoid,
                        bias=False)

    print('HR Translator Output: ' + str(l.get_shape().as_list()))
    return l


def lr_translator(x, activation='relu', scope='lr_translator', norm='layer', upsample='espcn', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        bottleneck_num_itr = lr_bottleneck_depth  # Num of bottleneck blocks of layers
        downsample_num_itr = lr_bottleneck_num_resize # Num of downsampling
        upsample_num_itr = lr_bottleneck_num_resize # Num of upsampling

        refine_num_itr = lr_refinement_depth

        print('LR Translator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        # Init Stage. Coordinated convolution: Embed explicit positional information
        block_depth = unit_block_depth * 2
        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        shorcut_layers = []

        # Downsample stage.
        for i in range(downsample_num_itr):
            if use_unet is True:
                shorcut_layers.append(l)

            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)
            print('LR Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_num_itr):
            print('LR Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, use_dilation=False, scope='bt_block_' + str(i))

        # Upsample stage
        for i in range(upsample_num_itr):
            if upsample == 'espcn':
                # ESPCN upsample
                block_depth = block_depth // 2
                l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
                l = act_func(l)
                l = tf.nn.depth_to_space(l, 2)

                print('LR Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
                if use_unet is True:
                    l = tf.concat([l, shorcut_layers[upsample_num_itr - 1 - i]], axis=-1)
                    print('Concat Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            elif upsample == 'resize':
                # Image resize
                block_depth = block_depth // 2

                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                # l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                # l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))
                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

                for j in range(refine_num_itr):
                    l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                     act_func=act_func, norm=norm, b_train=b_train,
                                                     scope='block_' + str(i) + '_' + str(j))
            else:
                # Deconvolution
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                  filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print('Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        if use_refinement is True:
            # Refinement
            for i in range(refine_num_itr):
                l = layers.conv(l, scope='refine_' + str(i), filter_dims=[3, 3, block_depth],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='refine_norm_' + str(i))
                l = act_func(l)

                print('LR Refinement ' + str(i) + ': ' + str(l.get_shape().as_list()))

        if use_attention is True:
            block_depth = block_depth // 4
            l = layers.conv(l, scope='squeeze', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=act_func, bias=False)
            l = layers.self_attention(l, channels=block_depth, act_func=act_func)

        lr_feature = l

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                        dilation=[1, 2, 2, 1], non_linear_fn=tf.nn.sigmoid,
                        bias=False)

    print('LR Translator Output: ' + str(l.get_shape().as_list()))
    return l, lr_feature


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    train_start_time = time.time()

    HR_G_scope = 'hr_translator'
    LR_G_scope = 'lr_translator'
    HR_DY_scope = 'hr_discriminator'
    LR_DY_scope = 'lr_discriminator'

    with tf.device('/device:CPU:0'):
        HR_X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR_X_IN = tf.placeholder(tf.float32, [batch_size, lr_input_height, lr_input_width, num_channel])
        HR_Y_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR_Y_IN = tf.placeholder(tf.float32, [batch_size, lr_input_height, lr_input_width, num_channel])

        HR_X_POOL = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR_X_POOL = tf.placeholder(tf.float32, [batch_size, lr_input_height, lr_input_width, num_channel])
        HR_Y_POOL = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR_Y_POOL = tf.placeholder(tf.float32, [batch_size, lr_input_height, lr_input_width, num_channel])

        LR = tf.placeholder(tf.float32, None)  # Learning Rate
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Low Resolution
    fake_lr_Y, fake_lr_feature = lr_translator(LR_X_IN, activation='relu', norm='instance', b_train=b_train, scope=LR_G_scope)
    # High Resolution
    fake_hr_Y = hr_translator(HR_X_IN, fake_lr_feature, activation='relu', norm='instance', b_train=b_train, scope=HR_G_scope)

    if use_identity_loss is True:
        # Low Resolution
        id_lr_Y, id_lr_Y_feature = lr_translator(LR_Y_IN, activation='relu', norm='instance', b_train=b_train, scope=LR_G_scope)
        # High Resolution
        id_hr_Y = hr_translator(HR_Y_IN, id_lr_Y_feature, activation='relu', norm='instance', b_train=b_train, scope=HR_G_scope)

    # High Resolution
    pool_hr_Y_feature, pool_hr_Y_logit = hr_discriminator(HR_Y_POOL, HR_X_POOL, activation='relu', norm='instance', b_train=b_train,
                                                          scope=HR_DY_scope, use_patch=use_patch_discriminator)
    real_hr_Y_feature, real_hr_Y_logit = hr_discriminator(HR_Y_IN, HR_X_IN, activation='relu', norm='instance', b_train=b_train,
                                                          scope=HR_DY_scope, use_patch=use_patch_discriminator)
    fake_hr_Y_feature, fake_hr_Y_logit = hr_discriminator(fake_hr_Y, HR_X_IN, activation='relu', norm='instance', b_train=b_train,
                                                          scope=HR_DY_scope, use_patch=use_patch_discriminator)
    # Low Resolution
    pool_lr_Y_feature, pool_lr_Y_logit = lr_discriminator(LR_Y_POOL, LR_X_POOL, activation='relu', norm='instance', b_train=b_train,
                                                          scope=LR_DY_scope, use_patch=use_patch_discriminator)
    real_lr_Y_feature, real_lr_Y_logit = lr_discriminator(LR_Y_IN, LR_X_IN, activation='relu', norm='instance', b_train=b_train,
                                                          scope=LR_DY_scope, use_patch=use_patch_discriminator)
    fake_lr_Y_feature, fake_lr_Y_logit = lr_discriminator(fake_lr_Y, LR_X_IN, activation='relu', norm='instance', b_train=b_train,
                                                          scope=LR_DY_scope, use_patch=use_patch_discriminator)

    # High Resolution Loss
    reconstruction_loss_hr_Y = get_residual_loss(HR_Y_IN, fake_hr_Y, type='l1')
    cyclic_loss_hr = alpha_hr * reconstruction_loss_hr_Y

    if use_gradient_loss is True:
        gradient_loss_hr = get_gradient_loss(HR_Y_IN, fake_hr_Y)

    # LS GAN
    trans_loss_X2Y_hr = get_feature_matching_loss(real_hr_Y_feature, fake_hr_Y_feature, 'l2') + \
                        get_residual_loss(fake_hr_Y_logit, tf.ones_like(fake_hr_Y_logit), 'l2')

    disc_loss_hr_Y = get_discriminator_loss(real_hr_Y_logit, tf.ones_like(real_hr_Y_logit), type='ls') + \
                     get_discriminator_loss(pool_hr_Y_logit, tf.zeros_like(pool_hr_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_hr_Y = alpha_hr * (get_residual_loss(HR_Y_IN, id_hr_Y, type='l1'))
        total_trans_loss_hr = trans_loss_X2Y_hr + cyclic_loss_hr + identity_loss_hr_Y
    else:
        total_trans_loss_hr = trans_loss_X2Y_hr + cyclic_loss_hr

    if use_gradient_loss is True:
        total_trans_loss_hr = total_trans_loss_hr + gradient_loss_hr

    total_disc_loss_hr = disc_loss_hr_Y

    # Low Resolution Loss
    reconstruction_loss_lr_Y = get_residual_loss(LR_Y_IN, fake_lr_Y, type='l1')
    cyclic_loss_lr = alpha_lr * reconstruction_loss_lr_Y

    if use_gradient_loss is True:
        gradient_loss_lr = get_gradient_loss(LR_Y_IN, fake_lr_Y)

    # LS GAN
    trans_loss_X2Y_lr = get_feature_matching_loss(real_lr_Y_feature, fake_lr_Y_feature, 'l2') + \
                        get_residual_loss(fake_lr_Y_logit, tf.ones_like(fake_lr_Y_logit), 'l2')

    disc_loss_lr_Y = get_discriminator_loss(real_lr_Y_logit, tf.ones_like(real_lr_Y_logit), type='ls') + \
                     get_discriminator_loss(pool_lr_Y_logit, tf.zeros_like(pool_lr_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_lr_Y = alpha_lr * (get_residual_loss(LR_Y_IN, id_lr_Y, type='l1'))
        total_trans_loss_lr = trans_loss_X2Y_lr + cyclic_loss_lr + identity_loss_lr_Y
    else:
        total_trans_loss_lr = trans_loss_X2Y_lr + cyclic_loss_lr

    if use_gradient_loss is True:
        total_trans_loss_lr = total_trans_loss_lr + gradient_loss_lr

    total_disc_loss_lr = disc_loss_lr_Y

    # High Resolution Network Variables and optimizer
    disc_Y_vars_hr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_DY_scope)
    disc_vars_hr = disc_Y_vars_hr

    disc_l2_regularizer_hr = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars_hr if 'bias' not in v.name])
    total_disc_loss_hr = total_disc_loss_hr + weight_decay * disc_l2_regularizer_hr

    trans_X2Y_vars_hr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_G_scope)
    trans_vars_hr = trans_X2Y_vars_hr
    trans_l2_regularizer_hr = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars_hr if 'bias' not in v.name])
    total_trans_loss_hr = total_trans_loss_hr + weight_decay * trans_l2_regularizer_hr

    # Low Resolution Network Variables and optimizer
    disc_Y_vars_lr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LR_DY_scope)
    disc_vars_lr = disc_Y_vars_lr

    disc_l2_regularizer_lr = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars_lr if 'bias' not in v.name])
    total_disc_loss_lr = total_disc_loss_lr + weight_decay * disc_l2_regularizer_lr

    trans_X2Y_vars_lr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LR_G_scope)
    trans_vars_lr = trans_X2Y_vars_lr
    trans_l2_regularizer_lr = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars_lr if 'bias' not in v.name])
    total_trans_loss_lr = total_trans_loss_lr + weight_decay * trans_l2_regularizer_lr

    disc_optimizer_hr = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_disc_loss_hr, var_list=disc_vars_hr)
    trans_optimizer_hr = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_trans_loss_hr, var_list=trans_vars_hr+trans_vars_lr)
    disc_optimizer_lr = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_disc_loss_lr, var_list=disc_vars_lr)
    trans_optimizer_lr = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_trans_loss_lr, var_list=trans_vars_lr)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print(util.COLORS.OKGREEN + 'Model Restored' + util.COLORS.ENDC)
        except:
            print(util.COLORS.WARNING + 'Start New Training. Wait ...' + util.COLORS.ENDC)

        trX_dir = os.path.join(train_data, 'X').replace("\\", "/")
        trY_dir = os.path.join(train_data, 'Y').replace("\\", "/")
        trX = os.listdir(trX_dir)
        trY = os.listdir(trY_dir)
        total_input_size = min(len(trX), len(trY)) // batch_size

        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        if file_batch_size == 0:
            file_batch_size = 1

        num_critic = 1

        # Discriminator draw inputs from image pool with threshold probability
        image_pool_hr = util.ImagePool(maxsize=30, threshold=0.5)
        image_pool_lr = util.ImagePool(maxsize=30, threshold=0.5)
        learning_rate = 2e-4

        for e in range(num_epoch):
            trY = shuffle(trY)
            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                rot = np.random.randint(-1, 3)
                imgs_Y_hr, _ = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False, add_eps=True,
                                     rotate=rot, resize=[input_width, input_height])

                if imgs_Y_hr is None:
                    continue

                imgs_Y_lr, _ = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False, add_eps=True,
                                           rotate=rot, resize=[lr_input_width, lr_input_height])

                if imgs_Y_hr is None:
                    continue

                avg_bright = np.average(imgs_Y_hr)
                imgs_Y_hr = np.where(imgs_Y_hr >= avg_bright, 1.0, 0)

                if len(imgs_Y_hr[0].shape) != 3:
                    imgs_Y_hr = np.expand_dims(imgs_Y_hr, axis=-1)

                avg_bright = np.average(imgs_Y_lr)
                imgs_Y_lr = np.where(imgs_Y_lr >= avg_bright, 1.0, 0)

                if len(imgs_Y_lr[0].shape) != 3:
                    imgs_Y_lr = np.expand_dims(imgs_Y_lr, axis=-1)

                trX = []
                for f_name in trY[start:end]:
                    i_name = f_name.replace('gt', 'input')
                    trX.append(i_name)

                imgs_X_hr, NonAugmented_imgs_X_hr = load_images(trX, base_dir=trX_dir, use_augmentation=True, add_eps=True,
                                     rotate=rot, resize=[input_width, input_height])

                if imgs_X_hr is None:
                    continue

                if len(imgs_X_hr[0].shape) != 3:
                    imgs_X_hr = np.expand_dims(imgs_X_hr, axis=-1)

                if len(NonAugmented_imgs_X_hr[0].shape) != 3:
                    NonAugmented_imgs_X_hr = np.expand_dims(NonAugmented_imgs_X_hr, axis=-1)

                imgs_X_lr, NonAugmented_imgs_X_lr = load_images(trX, base_dir=trX_dir, use_augmentation=False, add_eps=True,
                                                          rotate=rot, resize=[lr_input_width, lr_input_height])

                if imgs_X_lr is None:
                    continue

                if len(imgs_X_lr[0].shape) != 3:
                    imgs_X_lr = np.expand_dims(imgs_X_lr, axis=-1)

                if len(NonAugmented_imgs_X_lr[0].shape) != 3:
                    NonAugmented_imgs_X_lr = np.expand_dims(NonAugmented_imgs_X_lr, axis=-1)

                trans_lr = sess.run([fake_lr_Y], feed_dict={LR_X_IN: imgs_X_lr, b_train: True})
                trans_hr = sess.run([fake_hr_Y], feed_dict={HR_X_IN: imgs_X_hr, LR_X_IN: imgs_X_lr, b_train: True})

                threshold = fg_threshold
                trans_lr[0] = np.where(trans_lr[0] > threshold, 1.0, 0)
                pool_X2Y_lr = image_pool_lr([trans_lr[0], NonAugmented_imgs_X_lr])

                if e <= lr_pretrain_epoch:
                    trans_hr[0] = np.random.uniform(low=0, high=1.0, size=trans_hr[0].shape)
                else:
                    trans_hr[0] = np.where(trans_hr[0] > threshold, 1.0, 0)

                pool_X2Y_hr = image_pool_hr([trans_hr[0], NonAugmented_imgs_X_hr])

                cur_steps = (e * total_input_size) + itr + 1
                total_steps = (total_input_size * num_epoch)

                # Cosine learning rate decay
                learning_rate = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, d_loss_lr = sess.run([disc_optimizer_lr, total_disc_loss_lr],
                                        feed_dict={LR_Y_IN: imgs_Y_lr, LR_X_IN: NonAugmented_imgs_X_lr,
                                        LR_Y_POOL: pool_X2Y_lr[0], LR_X_POOL: pool_X2Y_lr[1], b_train: True, LR: learning_rate})

                _, d_loss_hr = sess.run([disc_optimizer_hr, total_disc_loss_hr],
                                        feed_dict={HR_Y_IN: imgs_Y_hr, HR_X_IN: NonAugmented_imgs_X_hr,
                                                   HR_Y_POOL: pool_X2Y_hr[0], HR_X_POOL: pool_X2Y_hr[1], b_train: True,
                                                   LR: learning_rate})
                if itr % num_critic == 0:
                    _, t_loss_lr, x2y_loss_lr = sess.run([trans_optimizer_lr, total_trans_loss_lr, trans_loss_X2Y_lr],
                                                         feed_dict={LR_Y_IN: imgs_Y_lr, LR_X_IN: imgs_X_lr,
                                                         b_train: True, LR: learning_rate})
                    print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKGREEN + 'd_loss_lr: ' + str(d_loss_lr) + util.COLORS.ENDC +
                          ', ' + util.COLORS.WARNING + 't_loss_lr: ' + str(t_loss_lr) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKBLUE + 'g_loss_lr: ' + str(x2y_loss_lr) + util.COLORS.ENDC)

                    if e > lr_pretrain_epoch:
                        _, t_loss_hr, x2y_loss_hr = sess.run([trans_optimizer_hr, total_trans_loss_hr, trans_loss_X2Y_hr],
                                                             feed_dict={HR_Y_IN: imgs_Y_hr, HR_X_IN: imgs_X_hr,
                                                             LR_Y_IN: imgs_Y_lr, LR_X_IN: imgs_X_lr,
                                                             b_train: True, LR: learning_rate})
                        print(util.COLORS.HEADER + '     : ' + str(e) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKGREEN + 'd_loss_hr: ' + str(d_loss_hr) + util.COLORS.ENDC +
                              ', ' + util.COLORS.WARNING + 't_loss_hr: ' + str(t_loss_hr) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKBLUE + 'g_loss_hr: ' + str(x2y_loss_hr) + util.COLORS.ENDC)

                        decoded_images_X2Y = trans_hr[0]
                        final_image = (decoded_images_X2Y[0] * 255.0)
                        cv2.imwrite(out_dir + '/' + trX[0], final_image)

                itr += 1

                if itr % 10 == 0:
                    print('Elapsed Time at  ' + str(cur_steps) + '/' + str(total_steps) + ' steps, ' + str(time.time() - train_start_time) + ' sec')

                if itr % 100 == 0:
                    try:
                        print(util.COLORS.WARNING + 'Saving model...' + util.COLORS.ENDC)
                        saver.save(sess, model_path)
                        print(util.COLORS.OKGREEN + 'Saved.' + util.COLORS.ENDC)
                    except:
                        print(util.COLORS.FAIL + 'Save failed' + util.COLORS.ENDC)

        print('Training Time: ' + str(time.time() - train_start_time))


def test(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    HR_G_scope = 'hr_translator'
    LR_G_scope = 'lr_translator'

    with tf.device('/device:CPU:0'):
        HR_X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR_X_IN = tf.placeholder(tf.float32, [batch_size, lr_input_height, lr_input_width, num_channel])

        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Low Resolution
    fake_lr_Y, fake_lr_feature = lr_translator(LR_X_IN, activation='relu', norm='instance', b_train=b_train,
                                               scope=LR_G_scope)
    # High Resolution
    fake_hr_Y = hr_translator(HR_X_IN, fake_lr_feature, activation='relu', norm='instance', b_train=b_train,
                              scope=HR_G_scope)

    trans_X2Y_vars_lr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LR_G_scope)
    trans_X2Y_vars_hr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_G_scope)

    trans_vars = trans_X2Y_vars_lr + trans_X2Y_vars_hr

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver(var_list=trans_vars)
            saver.restore(sess, model_path)
            print(util.COLORS.OKGREEN + 'Model Restored' + util.COLORS.ENDC)
        except:
            print(util.COLORS.FAIL + 'Fail to restore' + util.COLORS.ENDC)
            return

        trX_dir = test_data
        trX = os.listdir(trX_dir)
        score_list = []
        file_name_list = []

        for f_name in trX:
            imgs_HR_X, _ = load_images([f_name], base_dir=trX_dir, use_augmentation=False, resize=[input_width, input_height])
            imgs_LR_X, _ = load_images([f_name], base_dir=trX_dir, use_augmentation=False, resize=[lr_input_width, lr_input_height])

            if len(imgs_HR_X[0].shape) != 3:
                imgs_HR_X = np.expand_dims(imgs_HR_X, axis=3)
            if len(imgs_LR_X[0].shape) != 3:
                imgs_LR_X = np.expand_dims(imgs_LR_X, axis=3)

            #pixel_average = np.max(imgs_HR_X) * 0.5
            #imgs_HR_X = np.where(imgs_HR_X > pixel_average, 1.0, 0)
            #imgs_LR_X = np.where(imgs_LR_X > pixel_average, 1.0, 0)

            trans_X2Y = sess.run([fake_hr_Y], feed_dict={HR_X_IN: imgs_HR_X, LR_X_IN: imgs_LR_X, b_train: False})
            decoded_images_X2Y = np.squeeze(trans_X2Y)

            decoded_images_X2Y = (decoded_images_X2Y * 255.0)
            decoded_images_X2Y = np.where(decoded_images_X2Y > (255 * fg_threshold), 255, 0)

            sample_file_path = os.path.join(out_dir, f_name).replace("\\", "/")
            cv2.imwrite(sample_file_path, decoded_images_X2Y)

            if use_postprocess is True:
                configs = dict()
                configs['MAX_DISTANCE'] = postproc_smoothness
                configs['MAX_ITERATION'] = postproc_iteration

                result_contours = post_proc.refine_image(sample_file_path, configs)
                result_image_contours = np.asarray(result_contours)

                # Get result image
                mask_result_img = cv2.imread(sample_file_path)
                mask_result_img = np.zeros(mask_result_img.shape)
                cv2.drawContours(mask_result_img, result_image_contours, -1, (255, 255, 255), -1)

                # Save result image
                cv2.imwrite(sample_file_path, mask_result_img)

            decoded_images_X2Y = np.array(decoded_images_X2Y, np.int32)
            decoded_images_X2Y = ((decoded_images_X2Y + intensity) / 255) * 255

            fullname = os.path.join(trX_dir, f_name).replace("\\", "/")
            img = cv2.imread(fullname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            composed_img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
            composed_img = 1 + (composed_img + decoded_images_X2Y)
            composed_img[composed_img > 255] = 255
            composed_img = np.array(composed_img, np.int32)
            composed_img = ((composed_img + intensity) / 255) * 255

            np.set_printoptions(threshold=np.inf)
            di = util.patch_compare(decoded_images_X2Y,  composed_img, patch_size=[32, 32])
            di = np.array(di)
            score = np.sqrt(np.sum(np.square(di)))
            score_list.append(score)
            file_name_list.append(f_name)

        score_list = np.array(score_list)
        score_index = np.argsort(score_list)
        score_list = np.sort(score_list)

    for i in range(len(score_list)):
        score = score_list[i]
        print('Hardness: ' + file_name_list[score_index[i]] + ', ' + str(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')
    parser.add_argument('--out', type=str, help='output directory', default='imgs')
    parser.add_argument('--intensity', type=int, help='intensity', default=63)
    parser.add_argument('--alpha', type=int, help='reconstruction weight', default=50)
    parser.add_argument('--resize', type=int, help='training image resize to', default=256)
    parser.add_argument('--use_patch', help='Use patch discriminator', default=False, action='store_true')
    parser.add_argument('--use_postprocess', help='Use post process at test', default=False, action='store_true')
    parser.add_argument('--use_attention', help='Use attention block in translator', default=False, action='store_true')
    parser.add_argument('--smoothness', type=int, help='max line segment length to flatten', default=8)
    parser.add_argument('--iteration', type=int, help='num iterations of flattening', default=2)
    parser.add_argument('--epoch', type=int, help='num epoch', default=40)
    parser.add_argument('--pretrain_epoch', type=int, help='num epoch', default=10)
    parser.add_argument('--lr_ratio', type=int, help='low resolution ratio', default=8)
    parser.add_argument('--network_depth', type=int, help='low resolution translator depth', default=8)

    args = parser.parse_args()

    print(args)
    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path
    intensity = args.intensity
    alpha = args.alpha
    alpha_hr = alpha
    alpha_lr = alpha_hr

    out_dir = args.out
    use_postprocess = args.use_postprocess
    postproc_smoothness = args.smoothness
    postproc_iteration = args.iteration
    use_attention = args.use_attention
    use_unet = True
    use_unet_hr = True
    use_refinement = False
    # Input image will be resized.
    input_width = args.resize
    input_height = args.resize
    num_channel = 1
    unit_block_depth = 16  # Unit channel depth. Most layers would use N x unit_block_depth

    hr_bottleneck_depth = 8
    hr_refinement_depth = 1
    hr_discriminator_depth = 12
    hr_patch_discriminator_depth = 8

    lr_pretrain_epoch = args.pretrain_epoch
    lr_bottleneck_num_resize = 2  # Num of Downsampling, Upsampling
    lr_bottleneck_depth = args.network_depth  # Number of translator bottle neck layers or blocks
    lr_ratio = args.lr_ratio
    lr_input_width = input_width // lr_ratio
    lr_input_height = input_height // lr_ratio
    lr_refinement_depth = 2
    lr_discriminator_depth = 12
    lr_patch_discriminator_depth = 4

    batch_size = 1
    representation_dim = 128  # Discriminator last feature size.
    num_epoch = args.epoch
    use_identity_loss = True
    use_gradient_loss = True
    use_conditional_d = True
    use_patch_discriminator = args.use_patch
    weight_decay = 1e-4
    fg_threshold = 0.95

    if args.mode == 'train':
        train(model_path)
    else:
        test(model_path)
