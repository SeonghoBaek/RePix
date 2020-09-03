# ==============================================================================
# Author: Seongho Baek
# ==============================================================================

import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse


def load_images(file_name_list, base_dir, use_augmentation=False):
    images = []

    for file_name in file_name_list:
        fullname = os.path.join(base_dir, file_name).replace("\\", "/")
        img = cv2.imread(fullname)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)

        if img is not None:
            img = np.array(img)

            n_img = (img - 128.0) / 128.0
            images.append(n_img)

            if use_augmentation is True:
                n_img = cv2.flip(img, 1)
                n_img = (n_img - 128.0) / 128.0
                images.append(n_img)

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
        num_iter = 3

        print('Discriminator Input: ' + str(x.get_shape().as_list()))
        l = layers.conv(x, scope='conv_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Discriminator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            for j in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='res_block_' + str(i) + '_' + str(j))
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        if use_patch is True:
            print('Discriminator Patch Block : ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='patch_block_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')
            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[3, 3, 1], stride_dims=[1, 1],
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

        bottleneck_width = 64
        bottleneck_itr = 9
        num_iter = input_width // bottleneck_width
        num_iter = int(np.sqrt(num_iter))

        print('Translator Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        l = layers.conv(x, scope='conv_init', filter_dims=[7, 7, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        for i in range(num_iter):
            print('Translator Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        for i in range(bottleneck_itr):
            print('Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='bt_block_' + str(i))

        for i in range(num_iter):
            block_depth = block_depth // 2

            if use_upsample is True:
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                # l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                # l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))
                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                                non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)
                print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

                for j in range(2):
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

        l = layers.conv(l, scope='last', filter_dims=[7, 7, num_channel], stride_dims=[1, 1], non_linear_fn=tf.nn.tanh,
                        bias=False)

        print('Translator Final: ' + str(l.get_shape().as_list()))

    return l


def train(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

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
    fake_Y = translator(X_IN, activation='relu', norm='instance', b_train=b_train, scope=G_scope,
                        use_upsample=False)

    if use_identity_loss is True:
        id_Y = translator(Y_IN, activation='relu', norm='instance', b_train=b_train, scope=G_scope,
                          use_upsample=False)

    #with tf.device('/device:GPU:0'):
    _, Y_FAKE_IN_logit = discriminator(Y_FAKE_IN, activation='swish', norm='instance', b_train=b_train,
                                       scope=DY_scope, use_patch=True)
    _, real_Y_logit = discriminator(Y_IN, activation='swish', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=True)
    _, fake_Y_logit = discriminator(fake_Y, activation='swish', norm='instance', b_train=b_train,
                                    scope=DY_scope, use_patch=True)

    reconstruction_loss_Y = get_residual_loss(Y_IN, fake_Y, type='l1')
    alpha = 50.0
    cyclic_loss = alpha * reconstruction_loss_Y

    # LS GAN
    trans_loss_X2Y = tf.reduce_mean((fake_Y_logit - tf.ones_like(fake_Y_logit)) ** 2)
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

    #with tf.device('/device:GPU:0'):
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(total_disc_loss, var_list=disc_vars)

    #with tf.device('/device:GPU:1'):
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
        total_input_size = min(len(trX), len(trY))

        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        if file_batch_size == 0:
            file_batch_size = 1

        num_critic = 1

        image_pool = util.ImagePool(maxsize=100)
        learning_rate = 2e-4
        lr_decay_step = 100

        for e in range(num_epoch):
            trY = shuffle(trY)
            training_batch = zip(range(0, total_input_size, file_batch_size),
                                 range(file_batch_size, total_input_size + 1, file_batch_size))
            itr = 0
            if e > lr_decay_step:
                learning_rate = learning_rate * (num_epoch - e)/(num_epoch - lr_decay_step)

            for start, end in training_batch:
                imgs_Y = load_images(trY[start:end], base_dir=trY_dir, use_augmentation=False)
                if len(imgs_Y[0].shape) != 3:
                    imgs_Y = np.expand_dims(imgs_Y, axis=3)

                trX = []
                for f_name in trY[start:end]:
                    i_name = f_name.replace('gt', 'input')
                    trX.append(i_name)

                imgs_X = load_images(trX, base_dir=trX_dir, use_augmentation=False)
                if len(imgs_X[0].shape) != 3:
                    imgs_X = np.expand_dims(imgs_X, axis=3)

                trans_X2Y = sess.run([fake_Y], feed_dict={X_IN: imgs_X, b_train: True})
                pool_X2Y = image_pool(trans_X2Y[0])

                _, d_loss = sess.run([disc_optimizer, total_disc_loss],
                                     feed_dict={Y_IN: imgs_Y,
                                                Y_FAKE_IN: pool_X2Y, b_train: True, LR: learning_rate})

                if itr % num_critic == 0:
                    _, t_loss, x2y_loss = sess.run([trans_optimizer, total_trans_loss, trans_loss_X2Y],
                                         feed_dict={Y_IN: imgs_Y, X_IN: imgs_X, b_train: True, LR: learning_rate})

                    print(util.COLORS.HEADER + 'epoch: ' + str(e) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKGREEN + 'd_loss: ' + str(d_loss) + util.COLORS.ENDC +
                          ', ' + util.COLORS.WARNING + 't_loss: ' + str(t_loss) + util.COLORS.ENDC + ', ' +
                          util.COLORS.OKBLUE + 'x2y: ' + str(x2y_loss) + util.COLORS.ENDC)
                    decoded_images_X2Y = np.squeeze(trans_X2Y)
                    cv2.imwrite('imgs/X2Y_' + trX[0], (decoded_images_X2Y * 128.0) + 128.0)
                itr += 1

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

        for f_name in trX:
            imgs_X = load_images([f_name], base_dir=trX_dir, use_augmentation=False)
            if len(imgs_X[0].shape) != 3:
                imgs_X = np.expand_dims(imgs_X, axis=3)

            trans_X2Y, logit = sess.run([fake_Y, d_logit], feed_dict={X_IN: imgs_X, b_train: False})
            decoded_images_X2Y = np.squeeze(trans_X2Y)
            decoded_images_X2Y = (decoded_images_X2Y * 128.0) + 128.0
            decoded_images_X2Y = cv2.cvtColor(decoded_images_X2Y, cv2.COLOR_BGR2GRAY)
            decoded_images_X2Y = np.array(decoded_images_X2Y, np.int32)
            decoded_images_X2Y = ((decoded_images_X2Y + 127) / 255) * 255
            cv2.imwrite('imgs/t_' + f_name, decoded_images_X2Y)

            fullname = os.path.join(trX_dir, f_name).replace("\\", "/")
            img = cv2.imread(fullname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            composed_img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_CUBIC)
            composed_img = 1 + (composed_img + decoded_images_X2Y)
            composed_img[composed_img > 255] = 255
            composed_img = np.array(composed_img, np.int32)
            composed_img = ((composed_img + 63) / 255) * 255
            cv2.imwrite('imgs/c_' + f_name, composed_img)

            cv2.imwrite('imgs/o_' + f_name, (np.squeeze(imgs_X) * 128.0) + 128.0)

            np.set_printoptions(threshold=np.inf)
            di = util.patch_compare(decoded_images_X2Y,  composed_img, patch_size=[32, 32])
            di = np.array(di)
            score = np.sqrt(np.sum(np.square(di)))
            print('Hardness: ' + f_name + ', ' + str(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path

    dense_block_depth = 32

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32
    batch_size = 1
    representation_dim = 128

    img_width = 256
    img_height = 256
    input_width = 256
    input_height = 256
    num_channel = 3

    test_size = 100
    num_epoch = 300
    use_identity_loss = True

    if args.mode == 'train':
        train(model_path)
    else:
        test_data = args.test_data
        test(model_path)
