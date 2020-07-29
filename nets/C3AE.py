#coding=utf-8
import cv2
import os
import re
import feather
import base64
import math
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.activations import softmax, sigmoid
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, load_model
from keras.engine.input_layer import Input
from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D, Concatenate
from keras.layers.core import Dense, Dropout
from keras.layers import Lambda, Multiply, multiply, Reshape, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from sklearn.model_selection import train_test_split
from keras.backend import argmax, pool2d
from keras import backend as K
from keras import regularizers
from utils import focal_loss, ThresCallback, config_gpu


def BRA(input):
    bn = BatchNormalization()(input)
    activation = ReLU()(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)

def BN_ReLU(input, name):
    return ReLU()(BatchNormalization()(input))

def Concat(ins):
    return K.concatenate(ins, axis=-1)

def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation=sigmoid)(ReLU()(fc1))
    return multiply([scale, input])


def KeepTopN(input, keep=2):
    values, indices = tf.nn.top_k(input, k=keep, sorted=False)
    thres = tf.reduce_min(values)
    mask = tf.greater_equal(input, thres)
    return tf.nn.softmax(tf.where(mask, input, tf.zeros_like(input)))

def KeepMaxAdjactent(input, keep=2, axis=-1):
    dims = int(input.get_shape()[axis])
    combine_val = tf.gather(input, [x for x in range(1, dims)], axis=axis) + tf.gather(input, [x for x in range(0, dims-1)], axis=axis)
    indices = tf.argmax(combine_val, axis=axis)
    window_ind = tf.stack([indices, indices+1], axis=-1)
    print(window_ind)
    val = tf.gather_nd(input, window_ind)
    print(val)
    # 下面的tensor记得要改改, https://stackoverflow.com/questions/37001686/using-sparsetensor-as-a-trainable-variable
    # validate_indices 一定要设置成False否则过不去
    window_mat = tf.sparse_tensor_to_dense(tf.SparseTensor(window_ind, val, dense_shape=tf.shape(input, out_type=tf.int64)), validate_indices=False)
    print(window_mat)
    return tf.nn.softmax(window_mat)

def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def build_shared_plain_network(height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    input_image = Input(shape=(height, width, channel))

    if using_white_norm:
        wn = Lambda(white_norm, name="white_norm")(input_image)
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(wn)  # output 62*62*32
    else:
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(input_image)  # output 62*62*32
    block1 = BRA(conv1)
    block1 = SE_BLOCK(block1, using_SE)

    conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2")(block1)  # param 9248 = 32 * 32 * 3 * 3 + 32
    block2 = BRA(conv2)
    block2 = SE_BLOCK(block2, using_SE)  # put the se_net after BRA which achived better!!!!

    conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3")(block2)  # 9248
    block3 = BRA(conv3)
    block3 = SE_BLOCK(block3, using_SE)

    conv4 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv4")(block3)  # 9248
    block4 = BN_ReLU(conv4, name="BN_ReLu")  # 128
    block4 = SE_BLOCK(block4, using_SE)

    conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)  # 1024 + 32
    conv5 = SE_BLOCK(conv5, using_SE)  # r=16效果不如conv5

    flat_conv = Reshape((-1,))(conv5)
    # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
    # fc or pooling or any ohter operation
    #shape = map(int, conv5.get_shape()[1:])
    #shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

    pmodel = Model(input=input_image, output=[flat_conv])
    return pmodel

def build_net(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    print(base_model.summary())
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    bulk_feat = Dense(CATES, use_bias=True, activity_regularizer=regularizers.l1(0), activation=softmax, name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    #gender = Dense(2, activation=softmax, activity_regularizer=regularizers.l2(0), name="gender")(cfeat)

    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in range(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(input=[x1, x2, x3], output=[age, bulk_feat])

def preprocessing(dataframes, batch_size=50, category=12, interval=10, is_training=True, dropout=0.):
    # category: bin + 2 due to two side
    # interval: age interval
    from utils import age_data_generator
    return age_data_generator(dataframes, category=category, interval=interval, batch_size=batch_size, is_training=is_training, dropout=dropout)


def train(params):
    from utils import reload_data
    sample_rate, seed, batch_size, category, interval = 0.7, 2019, params.batch_size, params.category + 2, int(math.ceil(100. / params.category))
    lr = params.learning_rate
    data_dir, file_ptn = params.dataset, params.source
    dataframes = reload_data(data_dir, file_ptn)
    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
    train_gen = preprocessing(trainset, dropout=params.dropout, category=category, interval=interval)
    validation_gen = preprocessing(testset, is_training=False, category=category, interval=interval)
    print(trainset.groupby(["age"])["age"].agg("count"))

    print(testset.groupby(["age"]).agg(["count"]))
    age_dist = [trainset["age"][(trainset.age >= x -10) & (trainset.age <= x)].count() for x in range(10, 101, 10)]
    age_dist = [age_dist[0]] + age_dist + [age_dist[-1]]
    print(age_dist)

    if params.pretrain_path and os.path.exists(params.pretrain_path):
        models = load_model(params.pretrain_path, custom_objects={"pool2d": pool2d, "ReLU": ReLU, "BatchNormalization": BatchNormalization, "tf": tf, "focal_loss_fixed": focal_loss(age_dist)})
    else:
        models = build_net(category, using_SE=params.se_net, using_white_norm=params.white_norm)
    adam = Adam(lr=lr)
    #cate_weight = K.variable(params.weight_factor)

    models.compile(
        optimizer=adam,
        loss=["mae", focal_loss(age_dist)],  # "kullback_leibler_divergence"
        #metrics={"age": "mae", "W1": "mae", "gender": "acc"},
        metrics={"age": "mae", "W1": "mae"},
        loss_weights=[1, params.weight_factor]
        #loss_weights=[1, params.weight_factor, 5]
    )
    W2 = models.get_layer("age")

    print(models.summary())
    #thres_callback = ThresCallback(cate_weight, models.get_layer("age_mean_absolute_error"), 10, 10)

    def get_weights(epoch, loggs):
        print(epoch, K.get_value(models.optimizer.lr), W2.get_weights())

    callbacks = [
        ModelCheckpoint(params.save_path, monitor='val_age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        #TensorBoard(log_dir=params.log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
    ]
    history = models.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=250, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size * 3)


def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='C3AE retry')
    parser.add_argument(
        '-s', '--save_path', default="./model/c3ae_model_v2.h5", type=str,
        help='the best model to save')
    parser.add_argument(
        '-l', '--log_dir', default="./logs", type=str,
        help='the tensorboard log to save')
    parser.add_argument(
        '-r', '--r_factor', default=2, type=int,
        help='the r factor of SE')

    parser.add_argument(
        '--source', default="wiki", type=str,
        choices=['wiki', 'imdb', 'wiki|imdb'],
        help='"wiki|imdb" or regrex pattern of feather')

    parser.add_argument(
        '--dataset', default="./dataset/data/", type=str,
        help='the path of dataset to load')

    parser.add_argument(
        '-p', '--pretrain_path', dest="pretrain_path", default="", type=str,
        help='the pretrain path')

    parser.add_argument(
        '-b', '--batch_size', default=50, type=int,
        help='batch size degfault=50')

    parser.add_argument(
        '-w', '--weight_factor', default=10, type=int,
        help='age feature weight=10')


    parser.add_argument(
        '-c', '--category', default=10, type=int,
        help='category nums degfault=10, n+2')

    parser.add_argument(
        '-gpu', dest="gpu", action='store_true',
        help='config of GPU')

    parser.add_argument(
        '-se', "--se-net", dest="se_net", action='store_true',
        help='use SE-NET')

    parser.add_argument(
        '-white', '--white-norm', dest="white_norm", action='store_true',
        help='use white norm')

    parser.add_argument(
        '-d', '--dropout', default="0.2", type=float,
        help='dropout rate of erasing')

    parser.add_argument(
        '-lr', '--learning-rate', default="0.002", type=float,
        help='learning rate')

    params = parser.parse_args()
    if params.gpu:
        config_gpu()
    return params

if __name__ == "__main__":
    params = init_parse()
    train(params)
