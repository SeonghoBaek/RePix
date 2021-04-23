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

        for file_name in file_name_list:
            fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            if rotate > -1:
                img = cv2.rotate(img, rotate)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, dsize=(resize[0], resize[1]), interpolation=cv2.INTER_LINEAR)

            if img is not None:
                img = np.array(img)

                n_img = (img - 128.0) / 128.0

                if add_eps is True:
                    n_img = n_img + np.random.uniform(low=-1/128, high=1/128, size=n_img.shape)

                if use_augmentation is True:
                    if np.random.randint(low=0, high=10) < 5:
                        # square cut out
                        co_w = input_width // 16
                        co_h = co_w
                        padd_w = co_w // 2
                        padd_h = padd_w
                        r_x = np.random.randint(low=padd_w, high=resize[0] - padd_w)
                        r_y = np.random.randint(low=padd_h, high=resize[1] - padd_h)

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

        block_depth = unit_block_depth

        if use_patch is False:
            block_depth = block_depth // 2

        num_iter = bottleneck_num_layer
        norm_func = norm

        if use_conditional_d is True:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1

        print('Discriminator Input: ' + str(x.get_shape().as_list()))

        l = layers.coord_conv(x, scope='coord_init', filter_dims=[1, 1, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='norm_init')
        l = act_func(l)

        norm_func = norm

        if use_patch is True:
            print('Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(4):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                                 norm=norm_func, b_train=b_train, scope='patch_block_1_' + str(i))

            for i in range(2):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                                 norm=norm_func, b_train=b_train, use_dilation=True, scope='patch_block_2_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')

            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            l = layers.conv(l, scope='patch_dn_sample', filter_dims=[4, 4, block_depth // 4], stride_dims=[4, 4],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='patch_norm')
            last_layer = act_func(l)

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                                non_linear_fn=None, bias=False)
            print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:
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


def translator(x, activation='relu', scope='translator', norm='layer', upsample='espcn', b_train=False):
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

            block_depth = unit_block_depth  # Number of channel at start
            bottleneck_num_itr = bottleneck_num_layer  # Num of bottleneck blocks of layers
            bottleneck_wh_ratio = int(input_width // bottleneck_input_wh)

            resize_ratio = int(input_width // bottleneck_input_wh)
            downsample_num_itr = int(np.log2(bottleneck_wh_ratio))
            upsample_num_itr = int(np.log2(resize_ratio))
            refine_num_itr = 1

            print('Translator Input: ' + str(x.get_shape().as_list()))

            # Init Stage. Coordinated convolution: Embed explicit positional information
            l = layers.coord_conv(x, scope='coord_init', filter_dims=[1, 1, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=None, bias=False)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
            l = act_func(l)

            # Downsample stage.
            for i in range(downsample_num_itr):
                block_depth = int(unit_block_depth * np.exp2(i))
                l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
                l = act_func(l)
                print('Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            # Bottleneck stage
            for i in range(bottleneck_num_itr):
                print('Bottleneck Block : ' + str(l.get_shape().as_list()))
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
                    print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
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

            # Refinement
            for i in range(refine_num_itr):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, use_dilation=True,
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

        X_POOL = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Y_POOL = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

        LR = tf.placeholder(tf.float32, None)
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    fake_Y = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope)

    if use_identity_loss is True:
        id_Y = translator(Y_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope)

    pool_Y_feature, pool_Y_logit = discriminator(Y_POOL, X_POOL, activation='relu', norm='instance', b_train=b_train,
                                       scope=DY_scope, use_patch=use_patch_discriminator)
    real_Y_feature, real_Y_logit = discriminator(Y_IN, X_IN, activation='relu', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=use_patch_discriminator)
    fake_Y_feature, fake_Y_logit = discriminator(fake_Y, X_IN, activation='relu', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=use_patch_discriminator)

    reconstruction_loss_Y = get_residual_loss(Y_IN, fake_Y, type='l1')
    cyclic_loss = alpha * reconstruction_loss_Y

    if use_gradient_loss is True:
        gradient_loss = get_gradient_loss(Y_IN, fake_Y)

    # LS GAN
    trans_loss_X2Y = get_feature_matching_loss(real_Y_feature, fake_Y_feature, 'l2') + \
                     get_residual_loss(fake_Y_logit, tf.ones_like(fake_Y_logit), 'l2')

    disc_loss_Y = get_discriminator_loss(real_Y_logit, tf.ones_like(real_Y_logit), type='ls') + \
                  get_discriminator_loss(pool_Y_logit, tf.zeros_like(pool_Y_logit), type='ls')

    if use_identity_loss is True:
        identity_loss_Y = alpha * (get_residual_loss(Y_IN, id_Y, type='l1'))
        total_trans_loss = trans_loss_X2Y + cyclic_loss + identity_loss_Y
    else:
        total_trans_loss = trans_loss_X2Y + cyclic_loss

    if use_gradient_loss is True:
        total_trans_loss = total_trans_loss + gradient_loss

    total_disc_loss = disc_loss_Y

    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)
    disc_vars = disc_Y_vars

    disc_l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars if 'bias' not in v.name])
    total_disc_loss = total_disc_loss + weight_decay * disc_l2_regularizer

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    trans_vars = trans_X2Y_vars
    trans_l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trans_vars if 'bias' not in v.name])
    total_trans_loss = total_trans_loss + weight_decay * trans_l2_regularizer

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

        num_critic = 2

        # Discriminator draw inputs from image pool with threshold probability
        image_pool = util.ImagePool(maxsize=30, threshold=0.5)
        learning_rate = 2e-4

        for e in range(num_epoch):
            trY = shuffle(trY)
            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                rot = np.random.randint(-1, 3)
                imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False, add_eps=False,
                                     rotate=rot, resize=[input_width, input_height])

                if imgs_Y is None:
                    continue

                avg_bright = np.average(imgs_Y)
                imgs_Y = np.where(imgs_Y >= avg_bright, 1.0, -1.0)

                if len(imgs_Y[0].shape) != 3:
                    imgs_Y = np.expand_dims(imgs_Y, axis=-1)

                trX = []
                for f_name in trY[start:end]:
                    i_name = f_name.replace('gt', 'input')
                    trX.append(i_name)

                imgs_X = load_images(trX, base_dir=trX_dir, use_augmentation=True,
                                     rotate=rot, resize=[input_width, input_height])

                if imgs_X is None:
                    continue

                if len(imgs_X[0].shape) != 3:
                    imgs_X = np.expand_dims(imgs_X, axis=-1)

                trans_X2Y = sess.run([fake_Y], feed_dict={X_IN: imgs_X, b_train: True})

                trans_X2Y[0] = np.where(trans_X2Y[0] > 0.9, 1.0, -1.0)
                pool_X2Y = image_pool([trans_X2Y[0], imgs_X])

                cur_steps = (e * total_input_size) + itr + 1
                total_steps = (total_input_size * num_epoch)

                # Cosine learning rate decay
                learning_rate = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, d_loss = sess.run([disc_optimizer, total_disc_loss],
                                     feed_dict={Y_IN: imgs_Y, X_IN: imgs_X,
                                                Y_POOL: pool_X2Y[0], X_POOL: pool_X2Y[1], b_train: True, LR: learning_rate})

                if itr % num_critic == 0:
                    if use_identity_loss is True:
                        _, t_loss, x2y_loss = sess.run([trans_optimizer, total_trans_loss, trans_loss_X2Y],
                                             feed_dict={Y_IN: imgs_Y, X_IN: imgs_X,
                                                     b_train: True, LR: learning_rate})
                    else:
                        _, t_loss, x2y_loss = sess.run([trans_optimizer, total_trans_loss, trans_loss_X2Y],
                                                       feed_dict={Y_IN: imgs_Y, X_IN: imgs_X,
                                                                  b_train: True, LR: learning_rate})

                    print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKGREEN + 'd_loss: ' + str(d_loss) + util.COLORS.ENDC +
                          ', ' + util.COLORS.WARNING + 't_loss: ' + str(t_loss) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKBLUE + 'g_loss: ' + str(x2y_loss) + util.COLORS.ENDC)

                    decoded_images_X2Y = trans_X2Y[0]
                    final_image = (decoded_images_X2Y[0] * 128.0) + 128.0
                    cv2.imwrite(out_dir + '/' + trX[0], final_image)

                    # Test
                    '''
                    image_path = out_dir + '/' + trX[0]
                    configs = dict()
                    configs['MAX_DISTANCE'] = 4
                    configs['MAX_ITERATION'] = 2
                    contours = post_proc.refine_image(image_path, configs)
                    result_image_contours = np.asarray(contours)

                    # Get result image
                    mask_result_img = cv2.imread(image_path)
                    mask_result_img = np.zeros(mask_result_img.shape)
                    cv2.drawContours(mask_result_img, result_image_contours, -1, (255, 255, 255), -1)

                    # Save result image
                    cv2.imwrite(image_path, mask_result_img)
                    '''

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

    G_scope = 'translator_X_to_Y'
    DY_scope = 'discriminator_Y'

    with tf.device('/device:CPU:0'):
        X_IN = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    fake_Y = translator(X_IN, activation='swish', norm='instance', b_train=b_train, scope=G_scope)

    trans_X2Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_scope)
    disc_Y_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DY_scope)

    trans_vars = trans_X2Y_vars + disc_Y_vars

    _, d_logit = discriminator(fake_Y, activation='relu', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=use_patch_discriminator)

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
            imgs_X = load_images([f_name], base_dir=trX_dir, use_augmentation=False, resize=[input_width, input_height])
            if len(imgs_X[0].shape) != 3:
                imgs_X = np.expand_dims(imgs_X, axis=3)

            trans_X2Y, logit = sess.run([fake_Y, d_logit], feed_dict={X_IN: imgs_X, b_train: False})
            decoded_images_X2Y = np.squeeze(trans_X2Y)
            avg_bright = np.average(decoded_images_X2Y)
            decoded_images_X2Y = np.where(decoded_images_X2Y > avg_bright, 1.0, -1.0)
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
    parser.add_argument('--resize', type=int, help='traing image resize to', default=256)
    parser.add_argument('--use_patch', type=bool, help='Use patch discriminator', default=True)

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path
    intensity = args.intensity
    alpha = args.alpha
    out_dir = args.out

    # Input image will be resized.
    input_width = args.resize
    input_height = args.resize
    num_channel = 1
    bottleneck_num_layer = 32  # Number of translator bottle neck layers or blocks
    bottleneck_input_wh = 64  # Fixed. input_width // 2  # Translator bottle neck layer input size
    unit_block_depth = 128  # Unit channel depth. Most layers would use N x unit_block_depth

    batch_size = 1
    representation_dim = 128  # Discriminator last feature size.
    test_size = 100
    num_epoch = 300
    use_identity_loss = True
    use_gradient_loss = True
    use_conditional_d = True
    use_patch_discriminator = args.use_patch
    weight_decay = 1e-4

    if args.mode == 'train':
        train(model_path)
    else:
        test_data = args.test_data
        test(model_path)
