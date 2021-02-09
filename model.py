# ==============================================================================
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
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


def load_images(file_name_list, base_dir, use_augmentation=False, add_eps=False, rotate=-1):
    try:
        images = []

        for file_name in file_name_list:
            fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            if rotate > -1:
                img = cv2.rotate(img, rotate)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)

            if img is not None:
                img = np.array(img)

                n_img = (img - 128.0) / 128.0

                if add_eps is True:
                    n_img = n_img + np.random.uniform(low=-1/128, high=1/128, size=n_img.shape)

                if use_augmentation is True:
                    if np.random.randint(low=0, high=10) < 5:
                        # square cut out
                        co_w = input_width // 10
                        co_h = co_w
                        padd_w = co_w // 2
                        padd_h = padd_w
                        r_x = np.random.randint(low=padd_w, high=input_width - padd_w)
                        r_y = np.random.randint(low=padd_h, high=input_height - padd_h)

                        for i in range(co_w):
                            for j in range(co_h):
                                n_img[r_x - padd_w + i][r_y - padd_h + j] = 0.0

                    #n_img = cv2.flip(img, 1)
                    #n_img = (n_img - 128.0) / 128.0
                images.append(n_img)
    except cv2.error as e:
        print(e)
        return None

    return np.array(images)


def discriminator(x, activation='relu', scope='discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = dense_block_depth
        bottleneck_width = 8
        if use_patch is True:
            bottleneck_width = 16
        #num_iter = input_width // bottleneck_width
        #num_iter = int(np.sqrt(num_iter))
        num_iter = 2

        print('Discriminator Input: ' + str(x.get_shape().as_list()))
        #l = layers.conv(x, scope='conv_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
        #                non_linear_fn=None, bias=False)
        l = layers.coord_conv(x, scope='coord_init', filter_dims=[1, 1, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Discriminator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            for j in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='res_block_' + str(i) + '_' + str(j))
            #block_depth = block_depth * 2
            #l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
            #                non_linear_fn=None)
            #l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            #l = act_func(l)

        if use_patch is True:
            print('Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='patch_block_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')
            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            #logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[3, 3, 1], stride_dims=[3, 3],
            #                    non_linear_fn=None, bias=False)
            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                                non_linear_fn=None, bias=False)
            print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
            #print('Discriminator Attention Block : ' + str(l.get_shape().as_list()))
            #l = layers.self_attention(l, block_depth, act_func=act_func)
            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='at_block_' + str(i))

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
        # loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(target, value)))))
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

    v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    # loss = tf.abs(tf.subtract(v_a, v_b))
    loss = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b))) + tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss


def translator(x, activation='relu', scope='translator', norm='layer', use_upsample=False, b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if activation == 'swish':
                act_func = util.swish
            elif activation == 'relu':
                act_func = tf.nn.relu
            elif activation == 'lrelu':
                act_func = tf.nn.leaky_relu
            else:
                act_func = tf.nn.sigmoid

            block_depth = dense_block_depth  # Number of channel at start
            bottleneck_num_itr = 15  # Num of bottleneck blocks of layers
            bottleneck_wh_ratio = 4  # Shrink to 1/8 of input image size for botteneck layers

            downsample_num_itr = int(np.log2(bottleneck_wh_ratio))
            upsample_num_itr = downsample_num_itr
            refine_num_itr = downsample_num_itr

            print('Translator Input: ' + str(x.get_shape().as_list()))

            l = layers.coord_conv(x, scope='coor_init', filter_dims=[1, 1, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=None, bias=False)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
            l = act_func(l)

            # Downsample
            downsample_layers = []

            for i in range(downsample_num_itr):
                print('Translator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

                block_depth = block_depth * 2
                l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
                l = act_func(l)

            # Bottleneck
            for i in range(bottleneck_num_itr):
                print('Bottleneck Block : ' + str(l.get_shape().as_list()))
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=True, scope='bt_block_' + str(i))

            # Upsample
            for i in range(upsample_num_itr):
                block_depth = block_depth // 2

                if use_upsample is True:
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
                        l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                      act_func=act_func, norm=norm, b_train=b_train,
                                                      scope='block_' + str(i) + '_' + str(j))
                else:
                    l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i),
                                      filter_dims=[3, 3, block_depth],
                                      stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                    print('Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                    l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                    l = act_func(l)

            # Refinement
            for i in range(refine_num_itr):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, use_dilation=True,
                                                      act_func=act_func, norm=norm, b_train=b_train,
                                                      scope='refine_' + str(i))
                print('Refinement ' + str(i) + ': ' + str(l.get_shape().as_list()))

            # Transform to input channels
            l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                            dilation=[1, 2, 2, 1], non_linear_fn=tf.nn.tanh,
                            bias=False)

        return l


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    train_start_time = time.time()

    G_scope = 'translator_X_to_Y'
    DY_scope = 'discriminator_Y'

    with tf.device('/device:CPU:0'):
        X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_FAKE_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        LR = tf.placeholder(tf.float32, None)
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    #with tf.device('/device:GPU:1'):
    fake_Y = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                        use_upsample=False)

    if use_identity_loss is True:
        id_Y = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope,
                          use_upsample=False)

    #with tf.device('/device:GPU:0'):
    _, Y_FAKE_IN_logit = discriminator(Y_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                       scope=DY_scope, use_patch=True)
    #augmented_Y_IN = tf.add(Y_IN, tf.random_uniform(shape=Y_IN.get_shape(), minval=0.0, maxval=1/128))
    _, real_Y_logit = discriminator(Y_IN, activation='swish', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=True)
    _, fake_Y_logit = discriminator(fake_Y, activation='swish', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=True)

    reconstruction_loss_Y = get_residual_loss(Y_IN, fake_Y, type='l1')
    alpha = 50.0
    cyclic_loss = alpha * reconstruction_loss_Y

    # LS GAN
    beta = 1.0
    trans_loss_X2Y = beta * tf.reduce_mean((fake_Y_logit - tf.ones_like(fake_Y_logit)) ** 2)
    disc_loss_Y = get_discriminator_loss(real_Y_logit, tf.ones_like(real_Y_logit), type='ls') + \
                  get_discriminator_loss(Y_FAKE_IN_logit, tf.zeros_like(Y_FAKE_IN_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_Y = alpha * (get_residual_loss(Y_IN, id_Y, type='l1'))
        total_trans_loss = trans_loss_X2Y + cyclic_loss + identity_loss_Y
    else:
        total_trans_loss = trans_loss_X2Y + cyclic_loss

    total_disc_loss = disc_loss_Y

    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)
    disc_vars = disc_Y_vars

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    trans_vars = trans_X2Y_vars

    disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_disc_loss, var_list=disc_vars)
    trans_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_trans_loss, var_list=trans_vars)

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

        image_pool = util.ImagePool(maxsize=30)
        learning_rate = 2e-4
        lr_decay_step = 30

        for e in range(num_epoch):
            trY = shuffle(trY)
            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                rot = np.random.randint(-1, 3)
                imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False, add_eps=False, rotate=rot)

                if imgs_Y is None:
                    break

                if len(imgs_Y[0].shape) != 3:
                    imgs_Y = np.expand_dims(imgs_Y, axis=-1)

                trX = []
                for f_name in trY[start:end]:
                    i_name = f_name.replace('gt', 'input')
                    trX.append(i_name)

                imgs_X = load_images(trX, base_dir=trX_dir, use_augmentation=True, rotate=rot)

                if imgs_X is None:
                    break

                if len(imgs_X[0].shape) != 3:
                    imgs_X = np.expand_dims(imgs_X, axis=-1)

                trans_X2Y = sess.run([fake_Y], feed_dict={X_IN: imgs_X, b_train: True})
                pool_X2Y = image_pool(trans_X2Y[0])

                cur_steps = (e * total_input_size) + itr + 1
                total_steps = (total_input_size * num_epoch)

                learning_rate = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, d_loss = sess.run([disc_optimizer, total_disc_loss],
                                     feed_dict={Y_IN: imgs_Y,
                                                Y_FAKE_IN: pool_X2Y, b_train: True, LR: learning_rate})

                if itr % num_critic == 0:
                    imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False, add_eps=False, rotate=rot)

                    if imgs_Y is None:
                        break

                    if len(imgs_Y[0].shape) != 3:
                        imgs_Y = np.expand_dims(imgs_Y, axis=-1)

                    _, t_loss, x2y_loss = sess.run([trans_optimizer, total_trans_loss, trans_loss_X2Y],
                                         feed_dict={Y_IN: imgs_Y, X_IN: imgs_X, b_train: True, LR: learning_rate})

                    print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKGREEN + 'd_loss: ' + str(d_loss) + util.COLORS.ENDC +
                          ', ' + util.COLORS.WARNING + 't_loss: ' + str(t_loss) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKBLUE + 'g_loss: ' + str(x2y_loss) + util.COLORS.ENDC)

                    decoded_images_X2Y = np.squeeze(trans_X2Y)
                    #decoded_images_X2Y[decoded_images_X2Y > -0.99] = 1.0
                    #decoded_images_X2Y[decoded_images_X2Y <= -0.99] = -1.0
                    final_images = (decoded_images_X2Y * 128.0) + 128.0
                    #final_images = cv2.resize(final_images, dsize=(input_width // 2, input_height // 2), interpolation=cv2.INTER_NEAREST)
                    #final_images = cv2.resize(final_images, dsize=(input_width * 2, input_height * 2), interpolation=cv2.INTER_NEAREST)
                    #final_images = cv2.resize(final_images, dsize=(input_width, input_height), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(out_dir + '/' + trX[0], final_images[0])

                itr += 1

                if itr % 10 == 0:
                    print('Elapsed Time at ' + str(cur_steps) + '/' + str(total_steps) + ' steps, ' + str(time.time() - train_start_time) + ' sec')

                if itr % 200 == 0:
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

    G_scope = 'translator_X_to_Y'
    DY_scope = 'discriminator_Y'

    with tf.device('/device:CPU:0'):
        X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        #Y_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    fake_Y = translator(X_IN, activation='relu', norm='instance', b_train=b_train, scope=G_scope,
                        use_upsample=False)

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)

    trans_vars = trans_X2Y_vars + disc_Y_vars

    _, d_logit = discriminator(fake_Y, activation='swish', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=True)

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
            imgs_X = load_images([f_name], base_dir=trX_dir, use_augmentation=False)
            if len(imgs_X[0].shape) != 3:
                imgs_X = np.expand_dims(imgs_X, axis=3)

            trans_X2Y, logit = sess.run([fake_Y, d_logit], feed_dict={X_IN: imgs_X, b_train: False})
            decoded_images_X2Y = np.squeeze(trans_X2Y)
            decoded_images_X2Y = (decoded_images_X2Y * 128.0) + 128.0
            #decoded_images_X2Y = cv2.cvtColor(decoded_images_X2Y, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_dir + '/' + f_name, decoded_images_X2Y)
            decoded_images_X2Y = np.array(decoded_images_X2Y, np.int32)
            decoded_images_X2Y = ((decoded_images_X2Y + intensity) / 255) * 255

            fullname = os.path.join(trX_dir, f_name).replace("\\", "/")
            img = cv2.imread(fullname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            composed_img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)
            composed_img = 1 + (composed_img + decoded_images_X2Y)
            composed_img[composed_img > 255] = 255
            composed_img = np.array(composed_img, np.int32)
            composed_img = ((composed_img + intensity) / 255) * 255
            #cv2.imwrite('imgs/c_' + f_name, composed_img)
            #cv2.imwrite('imgs/o_' + f_name, (np.squeeze(imgs_X) * 128.0) + 128.0)

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

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path
    intensity = args.intensity
    out_dir = args.out

    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    batch_size = 2
    representation_dim = 128

    img_width = 256
    img_height = 256
    input_width = 256
    input_height = 256
    num_channel = 1

    test_size = 100
    num_epoch = 300
    use_identity_loss = True

    if args.mode == 'train':
        train(model_path)
    else:
        test_data = args.test_data
        test(model_path)
