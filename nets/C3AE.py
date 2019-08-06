#coding=utf-8
import cv2
import os
import re
import feather
import base64
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import keras.backend as K
from keras.activations import softmax, sigmoid
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, load_model
from keras.engine.input_layer import Input
from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply
from keras.layers.core import Dense, Dropout 
from keras.layers import Lambda, Multiply, multiply
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.backend import argmax, pool2d
from keras import backend as K
from keras import regularizers


def BRA(input):
    return pool2d(ReLU()(BatchNormalization()(input)), pool_size=(2, 2), strides=(2, 2), pool_mode="avg", padding="valid")  # 31*31*32

def BN_ReLU(input):
    return ReLU()(BatchNormalization()(input))

def Concat(ins):
    return K.concatenate(ins, axis=-1)

def SE_BLOCK(input, r_factor=2):
    channel_nums = input.get_shape()[-1].value
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation=sigmoid)(ReLU()(fc1))
    return multiply([scale, input])


def KeepTopN(input, keep=3):
    values, indices = tf.nn.top_k(input, k=keep, sorted=False)
    thres = tf.reduce_min(values)
    mask = tf.greater_equal(input, thres)
    return tf.nn.softmax(tf.where(mask, input, tf.zeros_like(input)))

def KeepMaxAdjactent(input, keep=2, axis=-1):
    dims = int(input.get_shape()[axis])
    combine_val = tf.gather(input, [x for x in xrange(1, dims)], axis=axis) + tf.gather(input, [x for x in xrange(0, dims-1)], axis=axis)
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

def build_complaim_network(height=64, width=64, channel=3):
    input_image = Input(shape=(height, width, channel))

    conv1 = Conv2D(32, (3, 3), padding="valid", strides=1)(input_image)  # output 62*62*32
    block1 = Lambda(BRA, name="BRA")(conv1)

    conv2 = Conv2D(32, (3, 3), padding="valid", strides=1)(block1)
    block2 = Lambda(BRA, name="BRA2")(block1)

    conv3 = Conv2D(32, (3, 3), padding="valid", strides=1)(block2)
    block3 = Lambda(BRA, name="BRA3")(conv3)
    #block3 = SE_BLOCK(bra3)

    conv4 = Conv2D(32, (3, 3), padding="valid", strides=1)(block3)
    block4 = Lambda(BN_ReLU, name="BN_ReLu")(conv4)

    #conv5 = Conv2D(32, (1, 1), padding="valid", strides=1)(block4)  # 4 * 4 *32
    conv5 = SE_BLOCK(block4)  # r=16效果不如conv5

    # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
    # fc or pooling or any ohter operation
    shape = map(int, conv5.get_shape()[1:])
    shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

    pmodel = Model(input=input_image, output=[shrinking_op])
    return pmodel

def build_net(CATES=12, height=64, width=64, channel=3):
    base_model = build_complaim_network()
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Lambda(Concat)([y1, y2, y3])
    #cfeat = Lambda(BN_ReLU, name="BN_ReLu2")(cfeat)
    bulk_feat = Dense(CATES, activation=softmax, kernel_regularizer=regularizers.l1(0), name="W1")(cfeat)
    #bulk_feat = Lambda(KeepTopN, name="bulk")(bulk_feat)
    age = Dense(1, name="age", kernel_regularizer=regularizers.l1(0))(bulk_feat)
    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in xrange(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(input=[x1, x2, x3], output=[age, bulk_feat])

def preprocessing(dataframes, batch_size=50, category=12, interval=10, is_training=True, dropout=0.):
    # category: bin + 2 due to two side
    # interval: age interval 
    from utils import generate_data_generator
    return generate_data_generator(dataframes, category=category, interval=interval, batch_size=batch_size, is_training=is_training, dropout=dropout)

def config_gpu():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

def train(params):
    from utils import reload_data
    sample_rate, seed, batch_size, category, interval = 0.8, 2019, 50, 12, 10
    data_dir, file_ptn = params.dataset, params.source
    dataframes = reload_data(data_dir, file_ptn)
    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
    train_gen = preprocessing(trainset, dropout=params.dropout)
    validation_gen = preprocessing(testset, is_training=False)

    if params.pretrain_path and os.path.exists(params.pretrain_path): 
        models = load_model(params.pretrain_path, custom_objects={"pool2d": pool2d, "ReLU": ReLU, "BatchNormalization": BatchNormalization, "tf": tf})
    else:
        models = build_net()
    models.compile(
        optimizer="Adam",
        loss=["mean_absolute_error", "kullback_leibler_divergence"],
        metrics={"age": "mae", "W1": "mae"},
        loss_weights=[1, 10]
    )
    callbacks = [
        ModelCheckpoint(params.save_path, monitor='val_age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        TensorBoard(log_dir=params.log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
        EarlyStopping(monitor='val_age_mean_absolute_error', patience=20, verbose=0, mode='min')
    ]
    history = models.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=600, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size * 3)


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
        '-gpu', dest="gpu", action='store_true',
        help='config of GPU')

    parser.add_argument(
        '-d', '--dropout', default="0.2", type=float,
        help='dropout rate of erasing')


    params = parser.parse_args()
    if params.gpu:
        config_gpu()
    return params

if __name__ == "__main__":
    params = init_parse()
    train(params)
