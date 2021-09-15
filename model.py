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


def load_images(file_name_list, base_dir, crop=None, use_augmentation=False, add_eps=False, rotate=-1, resize=[240, 240]):
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

            if crop is not None:
                patch_width = w // 2
                patch_height = h // 2
                patch_x = crop[0] * patch_width
                patch_y = crop[1] * patch_height
                patch = img[patch_x:patch_x + patch_width, patch_y:patch_y + patch_height]
                img = patch
                h = h // 2
                w = w // 2

            if h != resize[0]:
                img = cv2.resize(img, dsize=(resize[0], resize[1]), interpolation=cv2.INTER_AREA)

            if img is not None:
                img = np.array(img)
                img = img * 1.0
                n_img = (img - 127.5) / 127.5
                #n_img = img / 256.0
                non_augmented.append(n_img)

                if add_eps is True:
                    if np.random.randint(low=0, high=10) < 5:
                        n_img = n_img + np.random.uniform(low=0, high=1/256, size=n_img.shape)

                if use_augmentation is True:
                    # if np.random.randint(low=0, high=10) < 5:
                    # square cut out
                    co_w = input_width // 32
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


def discriminator(x1, x2, activation='relu', scope='discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = unit_block_depth * 2

        norm_func = norm

        if use_conditional_d is True:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1

        print(scope + ' Input: ' + str(x.get_shape().as_list()))

        l = layers.conv(x, scope='init', filter_dims=[7, 7, block_depth], stride_dims=[1, 1],
                        non_linear_fn=act_func, dilation=[1, 1, 1, 1])

        if use_patch is True:
            print(scope + ' Discriminator Patch Block : ' + str(l.get_shape().as_list()))
            downsample_num_itr = 3

            for i in range(downsample_num_itr):
                block_depth = block_depth * 2
                l = layers.conv(l, scope='disc_dn_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='disc_dn_norm_' + str(i))
                l = act_func(l)

            last_layer = l
            feature = l

            print(scope + ' Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[3, 3, 1], stride_dims=[1, 1],
                                non_linear_fn=None)
            print(scope + ' Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            num_iter = discriminator_depth

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_1_' + str(i))

            print(scope + ' Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample1', filter_dims=[3, 3, block_depth * 2], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_1')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 2], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_2_' + str(i))

            print(scope + ' Discriminator Block : ' + str(l.get_shape().as_list()))

            l = layers.conv(l, scope='disc_dn_sample2', filter_dims=[3, 3, block_depth * 4], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='disc_norm_2')

            for i in range(num_iter//3):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth * 4], num_layers=2, act_func=act_func,
                                              norm=norm_func, b_train=b_train, scope='disc_block_3_' + str(i))

            print(scope + ' Discriminator Block : ' + str(l.get_shape().as_list()))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim,
                                             scope='gp')

            print(scope + ' Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

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
        return gamma * tf.reduce_mean((real - fake) ** 2)


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

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l1')
    a_n = get_residual_loss(anchor, negative, 'l1')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return tf.reduce_mean(a_p / a_n)


def get_gradient_loss(img1, img2):
    # Laplacian second derivation
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    d2x_ax, d2y_ax = tf.image.image_gradients(dx_a)
    d2x_bx, d2y_bx = tf.image.image_gradients(dx_b)
    d2x_ay, d2y_ay = tf.image.image_gradients(dy_a)
    d2x_by, d2y_by = tf.image.image_gradients(dy_b)

    # loss1 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ax, d2x_bx))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ax, d2y_bx)))
    # loss2 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ay, d2x_by))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ay, d2y_by)))

    loss1 = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b)))
    loss2 = tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss1+loss2


def hr_refinement(x, activation='relu', scope='hr_refinement', norm='instance', upsample='resize', b_train=False):
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

        print(scope + ' Refinement Input: ' + str(x.get_shape().as_list()))

        l = x
        downsample_num_itr = 3
        upsample_num_itr = 3

        # Downsample stage.
        for i in range(downsample_num_itr):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)
            print(scope + ' Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(hr_bottleneck_depth):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, use_dilation=False,
                                             scope='bt_block_' + str(i))
        # Upsample stage
        for i in range(upsample_num_itr):
            # ESPCN upsample
            if upsample == 'espcn':
                block_depth = block_depth // 2
                l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
                l = act_func(l)
                l = tf.nn.depth_to_space(l, 2)

            elif upsample == 'resize':
                # Image resize
                block_depth = block_depth // 2
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                #l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                #l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))

                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)

            print(scope + ' Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[1, 1, num_channel], stride_dims=[1, 1],
                        non_linear_fn=None)
    print(scope + ' Refinement Output: ' + str(l.get_shape().as_list()))

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

        print('SCOPE: ' + scope)
        bottleneck_num_itr = lr_bottleneck_depth # Num of bottleneck blocks of layers
        downsample_num_itr = lr_bottleneck_num_resize # Num of downsampling
        upsample_num_itr = lr_bottleneck_num_resize # Num of upsampling

        print(scope + ' Translator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        # Init Stage. Coordinated convolution: Embed explicit positional information
        block_depth = 2 * unit_block_depth # * np.log2(lr_ratio)
        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=act_func)

        # Downsample stage.
        for i in range(downsample_num_itr):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)
            print(scope + ' Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_num_itr):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
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

                print(scope + ' Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            elif upsample == 'resize':
                # Image resize
                block_depth = block_depth // 2
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                #l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                #l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))

                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                print(scope + ' Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            else:
                # Deconvolution
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                  filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print(scope + ' Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        if use_attention is True:
            block_depth = block_depth // 4
            l = layers.conv(l, scope='squeeze', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=act_func)
            l = layers.self_attention(l, channels=block_depth, act_func=act_func)

        tr_feature = l

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[1, 1, num_channel], stride_dims=[1, 1],
                        non_linear_fn=None)

    print(scope + ' Translator Output: ' + str(l.get_shape().as_list()))
    return l, tr_feature


def hr_translator(x, tr_feature, activation='relu', scope='lr_translator', norm='layer', upsample='espcn', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid
        print('SCOPE: ' + scope)
        block_depth = unit_block_depth  # Number of channel at start
        bottleneck_num_itr = hr_bottleneck_depth  # Num of bottleneck blocks of layers

        downsample_num_itr = int(np.log2(lr_ratio))  # Num of downsampling
        upsample_num_itr = downsample_num_itr  # Num of upsampling

        print(scope + ' Translator ' + scope + ' Input: ' + str(x.get_shape().as_list()))

        # Init Stage. Coordinated convolution: Embed explicit positional information
        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                              non_linear_fn=act_func)

        # Downsample stage.
        for i in range(downsample_num_itr):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

            print(scope + ' Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        if hr_use_concat is True:
            l = tf.concat([l, tr_feature], axis=-1)
            l = layers.conv(l, scope='concat_front', filter_dims=[3, 3, block_depth],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='concat_norm_front')
            l = act_func(l)
            print(scope + ' concat layer: ' + str(l.get_shape().as_list()))
        else:
            l = tf.add(l, tr_feature)

        # Bottleneck stage
        for i in range(bottleneck_num_itr):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, use_dilation=False,
                                             scope='bt_block_' + str(i))

        # Upsample stage
        for i in range(upsample_num_itr):
            if upsample == 'espcn':
                # ESPCN upsample
                l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                                stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
                l = act_func(l)
                l = tf.nn.depth_to_space(l, 2)

                block_depth = block_depth // 2
                print(scope + ' ESPCN Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            elif upsample == 'resize':
                # Image resize
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                #l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                # l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))

                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                block_depth = block_depth // 2

                print(scope + ' Resize Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
            else:
                # Deconvolution
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                  filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print(scope + ' Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        lr_feature = l
        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[1, 1, num_channel], stride_dims=[1, 1],
                        non_linear_fn=None)
    print('HR Translator Output: ' + str(l.get_shape().as_list()))
    return l, lr_feature


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    train_start_time = time.time()

    HR_G_scope = 'hr_translator'
    LR_G_scope = 'lr_translator'
    TR_G_scope = 'tr_translator'
    HR_DY_scope = 'hr_discriminator'
    LR_DY_scope = 'lr_discriminator'
    TR_DY_scope = 'tr_discriminator'
    HR_REF_scope = 'hr_refinement'
    HR_REF_DY_scope = 'hr_ref_discriminator'

    with tf.device('/device:CPU:0'):
        HR_X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        LR_X_IN = tf.placeholder(tf.float32, [None, lr_input_height, lr_input_width, num_channel])
        TR_X_IN = tf.placeholder(tf.float32, [None, tr_input_height, tr_input_width, num_channel])
        HR_Y_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        LR_Y_IN = tf.placeholder(tf.float32, [None, lr_input_height, lr_input_width, num_channel])
        TR_Y_IN = tf.placeholder(tf.float32, [None, tr_input_height, tr_input_width, num_channel])
        REF_X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        REF_Y_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        ALPHA = tf.placeholder(tf.float32, None)
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    translator_norm = 'instance'
    translator_act = 'relu'
    disc_norm = 'instance'
    disc_act = 'lrelu'
    ref_act = 'relu'
    ref_norm = 'instance'
    upsample_type = 'resize' # espcn(default), resize, deconv

    fake_tr_Y, fake_tr_feature = lr_translator(TR_X_IN, upsample=upsample_type, activation=translator_act,
                                               norm=translator_norm, b_train=b_train, scope=TR_G_scope)
    fake_lr_Y, fake_lr_feature = hr_translator(LR_X_IN, fake_tr_feature, upsample=upsample_type, activation=translator_act,
                                               norm=translator_norm, b_train=b_train, scope=LR_G_scope)
    fake_hr_Y, _ = hr_translator(HR_X_IN, fake_lr_feature, upsample=upsample_type, activation=translator_act,
                                 norm=translator_norm, b_train=b_train, scope=HR_G_scope)
    real_hr_Y_feature, real_hr_Y_logit = discriminator(HR_Y_IN, HR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=HR_DY_scope, use_patch=use_patch_discriminator)
    fake_hr_Y_feature, fake_hr_Y_logit = discriminator(fake_hr_Y, HR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=HR_DY_scope, use_patch=use_patch_discriminator)
    real_lr_Y_feature, real_lr_Y_logit = discriminator(LR_Y_IN, LR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=LR_DY_scope, use_patch=use_patch_discriminator)
    fake_lr_Y_feature, fake_lr_Y_logit = discriminator(fake_lr_Y, LR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=LR_DY_scope, use_patch=use_patch_discriminator)
    real_tr_Y_feature, real_tr_Y_logit = discriminator(TR_Y_IN, TR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=TR_DY_scope, use_patch=use_patch_discriminator)
    fake_tr_Y_feature, fake_tr_Y_logit = discriminator(fake_tr_Y, TR_X_IN, activation=disc_act, norm=disc_norm,
                                                       b_train=b_train, scope=TR_DY_scope, use_patch=use_patch_discriminator)
    # Refinement
    refined_hr_X = hr_refinement(REF_X_IN, activation=ref_act, scope=HR_REF_scope, norm=ref_norm, b_train=b_train)
    augmented_REF_Y_IN = util.random_augment_brightness_constrast(REF_Y_IN)
    refined_hr_Y = hr_refinement(augmented_REF_Y_IN, activation=ref_act, scope=HR_REF_scope, norm=ref_norm, b_train=b_train)
    refinement_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_REF_scope)
    refinement_l2_regularizer_hr = tf.add_n([tf.nn.l2_loss(v) for v in refinement_vars if 'bias' not in v.name])
    refinement_identity_loss = get_residual_loss(refined_hr_Y, REF_Y_IN, type='l1')
    refinement_loss = get_residual_loss(refined_hr_X, REF_Y_IN, type='l1') \
                      + weight_decay * refinement_l2_regularizer_hr + refinement_identity_loss

    if use_identity_loss is True:
        id_tr_Y, id_tr_Y_feature = lr_translator(TR_Y_IN, activation=translator_act, norm=translator_norm, b_train=b_train,
                                                 upsample=upsample_type, scope=TR_G_scope)
        id_lr_Y, id_lr_Y_feature = hr_translator(LR_Y_IN, id_tr_Y_feature, activation=translator_act, norm=translator_norm,
                                                 b_train=b_train, upsample=upsample_type, scope=LR_G_scope)
        id_hr_Y, _ = hr_translator(HR_Y_IN, id_lr_Y_feature, activation=translator_act, norm=translator_norm, b_train=b_train,
                                   upsample=upsample_type, scope=HR_G_scope)

    # High Resolution Loss
    reconstruction_loss_hr_Y = get_residual_loss(HR_Y_IN, fake_hr_Y, type='l1')
    cyclic_loss_hr = ALPHA * reconstruction_loss_hr_Y

    if use_gradient_loss is True:
        gradient_loss_hr = alpha_grad * get_gradient_loss(HR_Y_IN, fake_hr_Y)
        cyclic_loss_hr = cyclic_loss_hr + gradient_loss_hr

    # LS GAN
    trans_loss_X2Y_hr = get_discriminator_loss(fake_hr_Y_logit, tf.ones_like(fake_hr_Y_logit), 'ls')

    smooth_hr_Y_real = tf.ones_like(real_hr_Y_logit) - tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)

    disc_loss_hr_Y = get_discriminator_loss(real_hr_Y_logit, smooth_hr_Y_real, type='ls') + \
                     get_discriminator_loss(fake_hr_Y_logit, tf.zeros_like(fake_hr_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_hr_Y = ALPHA * (get_residual_loss(HR_Y_IN, id_hr_Y, type='l1'))
        total_trans_loss_hr = trans_loss_X2Y_hr + cyclic_loss_hr + identity_loss_hr_Y
    else:
        total_trans_loss_hr = trans_loss_X2Y_hr + cyclic_loss_hr

    total_disc_loss_hr = disc_loss_hr_Y

    # Low Resolution Loss
    reconstruction_loss_lr_Y = get_residual_loss(LR_Y_IN, fake_lr_Y, type='l1')
    cyclic_loss_lr = ALPHA * reconstruction_loss_lr_Y

    if use_gradient_loss is True:
        gradient_loss_lr = alpha_grad * get_gradient_loss(LR_Y_IN, fake_lr_Y)
        cyclic_loss_lr = cyclic_loss_lr + gradient_loss_lr

    # LS GAN
    trans_loss_X2Y_lr = get_discriminator_loss(fake_lr_Y_logit, tf.ones_like(fake_lr_Y_logit), 'ls')

    smooth_lr_Y_real = tf.ones_like(real_lr_Y_logit) - tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)

    disc_loss_lr_Y = get_discriminator_loss(real_lr_Y_logit, smooth_lr_Y_real, type='ls') + \
                     get_discriminator_loss(fake_lr_Y_logit, tf.zeros_like(fake_lr_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_lr_Y = ALPHA * (get_residual_loss(LR_Y_IN, id_lr_Y, type='l1'))
        total_trans_loss_lr = trans_loss_X2Y_lr + cyclic_loss_lr + identity_loss_lr_Y
    else:
        total_trans_loss_lr = trans_loss_X2Y_lr + cyclic_loss_lr

    total_disc_loss_lr = disc_loss_lr_Y

    # Tiny Resolution Loss
    reconstruction_loss_tr_Y = get_residual_loss(TR_Y_IN, fake_tr_Y, type='l1')
    cyclic_loss_tr = ALPHA * reconstruction_loss_tr_Y

    # LS GAN
    trans_loss_X2Y_tr = get_discriminator_loss(fake_tr_Y_logit, tf.ones_like(fake_tr_Y_logit), 'ls')
    smooth_tr_Y_real = tf.ones_like(real_tr_Y_logit) - tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)

    disc_loss_tr_Y = get_discriminator_loss(real_tr_Y_logit, smooth_tr_Y_real, type='ls') + \
                     get_discriminator_loss(fake_tr_Y_logit, tf.zeros_like(fake_tr_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_tr_Y = ALPHA * (get_residual_loss(TR_Y_IN, id_tr_Y, type='l1'))
        total_trans_loss_tr = trans_loss_X2Y_tr + cyclic_loss_tr + identity_loss_tr_Y
    else:
        total_trans_loss_tr = trans_loss_X2Y_tr + cyclic_loss_tr

    total_disc_loss_tr = disc_loss_tr_Y

    # High Resolution Network Variables and optimizer
    disc_Y_vars_hr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_DY_scope)
    disc_vars_hr = disc_Y_vars_hr

    if use_d_weight_decay is True:
        disc_l2_regularizer_hr = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars_hr if 'bias' not in v.name])
        total_disc_loss_hr = total_disc_loss_hr + weight_decay * disc_l2_regularizer_hr

    trans_X2Y_vars_hr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=HR_G_scope)
    trans_vars_hr = trans_X2Y_vars_hr

    if use_g_weight_decay is True:
        trans_l2_regularizer_hr = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars_hr if 'bias' not in v.name])
        total_trans_loss_hr = total_trans_loss_hr + weight_decay * trans_l2_regularizer_hr

    # Low Resolution Network Variables and optimizer
    disc_Y_vars_lr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LR_DY_scope)
    disc_vars_lr = disc_Y_vars_lr

    if use_d_weight_decay is True:
        disc_l2_regularizer_lr = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars_lr if 'bias' not in v.name])
        total_disc_loss_lr = total_disc_loss_lr + weight_decay * disc_l2_regularizer_lr

    trans_X2Y_vars_lr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LR_G_scope)
    trans_vars_lr = trans_X2Y_vars_lr

    if use_g_weight_decay is True:
        trans_l2_regularizer_lr = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars_lr if 'bias' not in v.name])
        total_trans_loss_lr = total_trans_loss_lr + weight_decay * trans_l2_regularizer_lr

    disc_Y_vars_tr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TR_DY_scope)
    disc_vars_tr = disc_Y_vars_tr

    if use_d_weight_decay is True:
        disc_l2_regularizer_tr = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars_tr if 'bias' not in v.name])
        total_disc_loss_tr = total_disc_loss_tr + weight_decay * disc_l2_regularizer_tr

    trans_X2Y_vars_tr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TR_G_scope)
    trans_vars_tr = trans_X2Y_vars_tr

    if use_g_weight_decay is True:
        trans_l2_regularizer_tr = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars_tr if 'bias' not in v.name])
        total_trans_loss_tr = total_trans_loss_tr + weight_decay * trans_l2_regularizer_tr

    total_joint_loss = total_trans_loss_lr + total_trans_loss_hr + total_trans_loss_tr
    total_joint_trans_vars = trans_vars_hr + trans_vars_lr + trans_vars_tr

    learning_rate = 1e-3
    disc_optimizer_hr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_disc_loss_hr, var_list=disc_vars_hr)
    trans_optimizer_hr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_joint_loss, var_list=total_joint_trans_vars)
    disc_optimizer_lr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_disc_loss_lr, var_list=disc_vars_lr)
    trans_optimizer_lr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_trans_loss_lr + total_trans_loss_tr,
                                                                             var_list=trans_vars_lr + trans_vars_tr)
    disc_optimizer_tr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_disc_loss_tr, var_list=disc_vars_tr)
    trans_optimizer_tr = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_trans_loss_tr, var_list=trans_vars_tr)
    refinement_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(refinement_loss, var_list=refinement_vars)

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
        total_input_size = min(len(trX), len(trY))

        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        if file_batch_size == 0:
            file_batch_size = 1

        num_critic = 1

        for e in range(num_epoch):
            trY = shuffle(trY)
            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs_Y_hr, _ = load_images(trY[start:end], base_dir=trY_dir, resize=[input_width, input_height])

                if imgs_Y_hr is None:
                    continue

                imgs_Y_lr, _ = load_images(trY[start:end], base_dir=trY_dir, resize=[lr_input_width, lr_input_height])
                imgs_Y_tr, _ = load_images(trY[start:end], base_dir=trY_dir, resize=[tr_input_width, tr_input_height])

                avg_bright = np.average(imgs_Y_hr)
                imgs_Y_hr = np.where(imgs_Y_hr >= avg_bright, 1.0, -1.0)

                if len(imgs_Y_hr[0].shape) != 3:
                    imgs_Y_hr = np.expand_dims(imgs_Y_hr, axis=-1)

                avg_bright = np.average(imgs_Y_lr)
                imgs_Y_lr = np.where(imgs_Y_lr >= avg_bright, 1.0, -1.0)

                if len(imgs_Y_lr[0].shape) != 3:
                    imgs_Y_lr = np.expand_dims(imgs_Y_lr, axis=-1)

                avg_bright = np.average(imgs_Y_tr)
                imgs_Y_tr = np.where(imgs_Y_tr >= avg_bright, 1.0, -1.0)

                if len(imgs_Y_tr[0].shape) != 3:
                    imgs_Y_tr = np.expand_dims(imgs_Y_tr, axis=-1)

                trX = []
                for f_name in trY[start:end]:
                    i_name = f_name.replace('gt', 'input')
                    trX.append(i_name)

                imgs_X_hr, _ = load_images(trX, base_dir=trX_dir, resize=[input_width, input_height])

                if imgs_X_hr is None:
                    continue

                if len(imgs_X_hr[0].shape) != 3:
                    imgs_X_hr = np.expand_dims(imgs_X_hr, axis=-1)

                imgs_X_lr, _ = load_images(trX, base_dir=trX_dir, resize=[lr_input_width, lr_input_height])

                if len(imgs_X_lr[0].shape) != 3:
                    imgs_X_lr = np.expand_dims(imgs_X_lr, axis=-1)

                imgs_X_tr, _ = load_images(trX, base_dir=trX_dir, resize=[tr_input_width, tr_input_height])

                if len(imgs_X_tr[0].shape) != 3:
                    imgs_X_tr = np.expand_dims(imgs_X_tr, axis=-1)

                cur_steps = (e * total_input_size) + itr + 1.0
                total_steps = (total_input_size * num_epoch * 1.0)

                # Cosine learning rate decay
                #lr = learning_rate * np.cos((np.pi * 7.0 / 16.0) * (cur_steps / total_steps))

                if use_adaptive_alpha is True:
                    trans_alpha = 1.0 + alpha * (1.0 - (e * 1.0 / num_epoch))
                else:
                    trans_alpha = alpha

                _, d_loss_tr = sess.run([disc_optimizer_tr, total_disc_loss_tr],
                                        feed_dict={TR_Y_IN: imgs_Y_tr, TR_X_IN: imgs_X_tr,
                                                   b_train: True})

                if e >= lr_pretrain_epoch:
                    _, d_loss_hr = sess.run([disc_optimizer_hr, total_disc_loss_hr],
                                            feed_dict={HR_Y_IN: imgs_Y_hr, HR_X_IN: imgs_X_hr, LR_X_IN: imgs_X_lr, TR_X_IN: imgs_X_tr,
                                            b_train: True})

                if e >= tr_pretrain_epoch:
                    _, d_loss_lr = sess.run([disc_optimizer_lr, total_disc_loss_lr],
                                            feed_dict={LR_Y_IN: imgs_Y_lr, LR_X_IN: imgs_X_lr, TR_X_IN: imgs_X_tr,
                                                       b_train: True})

                if itr % num_critic == 0:
                    if e >= lr_pretrain_epoch:
                        _, t_loss_hr, x2y_loss_hr, t_loss_lr = sess.run([trans_optimizer_hr,
                                                                         total_trans_loss_hr, trans_loss_X2Y_hr, total_trans_loss_lr],
                                                                        feed_dict={HR_Y_IN: imgs_Y_hr,
                                                                                   HR_X_IN: imgs_X_hr,
                                                                                   LR_Y_IN: imgs_Y_lr,
                                                                                   LR_X_IN: imgs_X_lr,
                                                                                   TR_Y_IN: imgs_Y_tr,
                                                                                   TR_X_IN: imgs_X_tr,
                                                                                   b_train: True, ALPHA: trans_alpha})

                        trans_hr = sess.run([fake_hr_Y], feed_dict={HR_X_IN: imgs_X_hr, LR_X_IN: imgs_X_lr, TR_X_IN: imgs_X_tr, b_train: True})
                        _, refine_loss, refined_x = sess.run([refinement_optimizer, refinement_loss, refined_hr_X],
                                                             feed_dict={REF_X_IN: trans_hr[0], REF_Y_IN: imgs_Y_hr,
                                                                        b_train: True, ALPHA: trans_alpha})
                        decoded_images_X2Y = refined_x

                        print(util.COLORS.HEADER + ' epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKGREEN + 'd_loss_hr: ' + str(d_loss_hr) + util.COLORS.ENDC +
                              ', ' + util.COLORS.WARNING + 't_loss_hr: ' + str(t_loss_hr) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKBLUE + 'g_loss_hr: ' + str(x2y_loss_hr) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKGREEN + 'ref_loss: ' + str(refine_loss) + util.COLORS.ENDC)
                    elif e >= tr_pretrain_epoch:
                        _, t_loss_lr, x2y_loss_lr = sess.run(
                            [trans_optimizer_lr, total_trans_loss_lr, trans_loss_X2Y_lr],
                            feed_dict={LR_Y_IN: imgs_Y_lr,
                                       LR_X_IN: imgs_X_lr,
                                       TR_Y_IN: imgs_Y_tr,
                                       TR_X_IN: imgs_X_tr,
                                       REF_Y_IN: imgs_Y_hr,
                                       b_train: True, ALPHA: alpha})
                        print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKGREEN + 'd_loss_lr: ' + str(d_loss_lr) + util.COLORS.ENDC +
                              ', ' + util.COLORS.WARNING + 't_loss_lr: ' + str(t_loss_lr) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKBLUE + 'g_loss_lr: ' + str(x2y_loss_lr) +
                              util.COLORS.ENDC)

                        trans_lr = sess.run([fake_lr_Y], feed_dict={LR_X_IN: imgs_X_lr, TR_X_IN: imgs_X_tr, b_train: True})
                        decoded_images_X2Y = trans_lr[0]
                    else:
                        _, t_loss_tr, x2y_loss_tr = sess.run(
                            [trans_optimizer_tr, total_trans_loss_tr, trans_loss_X2Y_tr],
                            feed_dict={TR_Y_IN: imgs_Y_tr,
                                       TR_X_IN: imgs_X_tr,
                                       REF_Y_IN: imgs_Y_hr,
                                       b_train: True, ALPHA: alpha})
                        print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKGREEN + 'd_loss_tr: ' + str(d_loss_tr) + util.COLORS.ENDC +
                              ', ' + util.COLORS.WARNING + 't_loss_tr: ' + str(t_loss_tr) + util.COLORS.ENDC + ', ' +
                              util.COLORS.OKBLUE + 'g_loss_tr: ' + str(x2y_loss_tr) +
                              util.COLORS.ENDC)

                        trans_tr = sess.run([fake_tr_Y], feed_dict={TR_X_IN: imgs_X_tr, b_train: True})
                        decoded_images_X2Y = trans_tr[0]

                    for num_outputs in range(batch_size):
                        final_image = decoded_images_X2Y[num_outputs] * 127.5 + 127.5
                        cv2.imwrite(out_dir + '/' + trX[num_outputs], final_image)

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
    HR_REF_scope = 'hr_refinement'
    TR_G_scope = 'tr_translator'
    translator_norm = 'instance'
    translator_act = 'relu'
    ref_act = 'relu'
    ref_norm = 'instance'
    upsample_type = 'resize'

    show_hardness_score = False

    with tf.device('/device:CPU:0'):
        HR_X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
        LR_X_IN = tf.placeholder(tf.float32, [None, lr_input_height, lr_input_width, num_channel])
        TR_X_IN = tf.placeholder(tf.float32, [None, tr_input_height, tr_input_width, num_channel])

        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    _, fake_tr_feature = lr_translator(TR_X_IN, activation=translator_act, norm=translator_norm, upsample=upsample_type,
                                               b_train=b_train, scope=TR_G_scope)
    # Low Resolution
    _, fake_lr_feature = hr_translator(LR_X_IN, fake_tr_feature, activation=translator_act, norm=translator_norm, b_train=b_train, upsample=upsample_type,
                                               scope=LR_G_scope)

    # High Resolution
    fake_hr_Y, _ = hr_translator(HR_X_IN, fake_lr_feature, activation=translator_act, norm=translator_norm, b_train=b_train, upsample=upsample_type,
                              scope=HR_G_scope)
    refined_hr_X = hr_refinement(fake_hr_Y, activation=ref_act, scope=HR_REF_scope, norm=ref_norm, b_train=b_train)

    trans_X2Y_vars_tr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TR_G_scope)
    trans_X2Y_vars_lr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=LR_G_scope)
    trans_X2Y_vars_hr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=HR_G_scope)
    refine_X2Y_vars_hr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=HR_REF_scope)

    trans_vars = trans_X2Y_vars_tr + trans_X2Y_vars_lr + trans_X2Y_vars_hr + refine_X2Y_vars_hr

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
        total_input_size = len(trX)

        if show_hardness_score is True:
            score_list = []
            file_name_list = []

        test_batch = zip(range(0, total_input_size, batch_size),
                             range(batch_size, total_input_size + 1, batch_size))

        for start, end in test_batch:
            _, imgs_HR_X = load_images(trX[start:end], base_dir=trX_dir, use_augmentation=False, resize=[input_width, input_height])
            _, imgs_LR_X = load_images(trX[start:end], base_dir=trX_dir, use_augmentation=False, resize=[lr_input_width, lr_input_height])
            _, imgs_TR_X = load_images(trX[start:end], base_dir=trX_dir, use_augmentation=False, resize=[tr_input_width, tr_input_height])

            if len(imgs_HR_X[0].shape) != 3:
                imgs_HR_X = np.expand_dims(imgs_HR_X, axis=3)
            if len(imgs_LR_X[0].shape) != 3:
                imgs_LR_X = np.expand_dims(imgs_LR_X, axis=3)
            if len(imgs_TR_X[0].shape) != 3:
                imgs_TR_X = np.expand_dims(imgs_TR_X, axis=3)

            trans_X2Y = sess.run([refined_hr_X], feed_dict={HR_X_IN: imgs_HR_X, LR_X_IN: imgs_LR_X, TR_X_IN: imgs_TR_X, b_train: True})
            decoded_images_X2Y = np.squeeze(trans_X2Y)
            threshold = np.average(decoded_images_X2Y)
            decoded_images_X2Y = np.where(decoded_images_X2Y > threshold, 1.0, -1.0)
            decoded_images_X2Y = decoded_images_X2Y * 127.5 + 127.5

            for num_outputs in range(batch_size):
                sample_file_path = os.path.join(out_dir, trX[start + num_outputs]).replace("\\", "/")
                cv2.imwrite(sample_file_path, decoded_images_X2Y[num_outputs])

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

                if show_hardness_score is True:
                    target_images_X2Y = np.array(decoded_images_X2Y[num_outputs], np.int32)
                    target_images_X2Y = ((target_images_X2Y + intensity) / 255) * 255

                    fullname = os.path.join(trX_dir, trX[start + num_outputs]).replace("\\", "/")
                    img = cv2.imread(fullname)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    composed_img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
                    composed_img = 1 + (composed_img + target_images_X2Y)
                    composed_img[composed_img > 255] = 255
                    composed_img = np.array(composed_img, np.int32)
                    composed_img = ((composed_img + intensity) / 255) * 255

                    np.set_printoptions(threshold=np.inf)
                    di = util.patch_compare(target_images_X2Y,  composed_img, patch_size=[32, 32])
                    di = np.array(di)
                    score = np.sqrt(np.sum(np.square(di)))
                    score_list.append(score)
                    file_name_list.append(trX[start + num_outputs])

        if show_hardness_score is True:
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
    parser.add_argument('--pretrain_epoch', type=int, help='num epoch', default=0)
    parser.add_argument('--lr_ratio', type=int, help='low resolution ratio', default=8)
    parser.add_argument('--network_depth', type=int, help='low resolution translator depth', default=8)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=4)

    args = parser.parse_args()

    print(args)
    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path
    intensity = args.intensity
    alpha = args.alpha

    out_dir = args.out
    use_postprocess = args.use_postprocess
    postproc_smoothness = args.smoothness
    postproc_iteration = args.iteration
    use_attention = args.use_attention

    # Input image will be resized.
    input_width = args.resize
    input_height = args.resize
    num_channel = 1
    unit_block_depth = 24  # Unit channel depth. Most layers would use N x unit_block_depth

    hr_bottleneck_depth = 8
    discriminator_depth = 12
    hr_use_concat = False

    tr_pretrain_epoch = args.pretrain_epoch
    lr_pretrain_epoch = 2 * tr_pretrain_epoch
    lr_bottleneck_depth = args.network_depth  # Number of translator bottle neck layers or blocks
    lr_ratio = args.lr_ratio
    lr_bottleneck_num_resize = 1
    lr_input_width = input_width // lr_ratio
    lr_input_height = input_height // lr_ratio
    tr_input_width = lr_input_width // lr_ratio
    tr_input_height = lr_input_height // lr_ratio

    batch_size = args.batch_size  # Instance normalization
    representation_dim = 512  # Discriminator last feature size.
    num_epoch = args.epoch
    use_identity_loss = True
    use_gradient_loss = False
    alpha_grad = 0.3
    use_conditional_d = True
    use_patch_discriminator = args.use_patch
    weight_decay = 1e-4
    fg_threshold = 0.95
    use_adaptive_alpha = False
    use_g_weight_decay = True
    use_d_weight_decay = True

    if args.mode == 'train':
        train(model_path)
    else:
        test(model_path)
