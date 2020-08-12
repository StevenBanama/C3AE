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
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D, Concatenate, Layer, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Lambda, Multiply, multiply, Reshape, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.backend import argmax, pool2d
from tensorflow.keras import regularizers
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import unit_norm
from utils import focal_loss, ThresCallback, CosineAnnealingScheduler, MishActivation, MishActivation6, config_cpu, config_gpu, model_refresh_without_nan



Activation = MishActivation6  #ReLU
#Activation = ReLU

class GeM(Layer):

    def __init__(self, init_p=3., dynamic_p=False, **kwargs):
        # https://arxiv.org/pdf/1711.02512.pdf
        # add this behind relu
        if init_p <= 0:
             raise Exception("fatal p")
        super(GeM, self).__init__(**kwargs)
        self.init_p = init_p
        self.epsilon = 1e-8
        self.dynamic_p = dynamic_p

    def build(self, input_shape):
        super(GeM, self).build(input_shape)
        if self.dynamic_p:
            self.init_p = tf.Variable(self.init_p, dtype=tf.float32)

    def call(self, inputs):
        pool = tf.nn.avg_pool(tf.pow(tf.math.maximum(inputs, self.epsilon), self.init_p), inputs.shape[1:3], strides=(1, 1), padding="VALID")
        pool = tf.pow(pool, 1. / self.init_p)
        return pool

def BRA(input):
    bn = BatchNormalization()(input)
    activation = Activation()(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)

def BN_ReLU(input, name):
    return Activation()(BatchNormalization()(input))

def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GeM(dynamic_p=True)(input)  # GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation=sigmoid)(Activation()(fc1))
    return multiply([scale, input])

def SE_BLOCK1(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor / 2)(ga_pooling)

    gm_pooling = GlobalMaxPooling2D()(input)
    fc2 = Dense(channel_nums//r_factor / 2)(gm_pooling)

    fc = Concatenate()([fc1, fc2]) 

    scale = Dense(channel_nums, activation=sigmoid)(Activation()(fc))
    return multiply([scale, input])

def SE_BLOCK2(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)

    gm_pooling = GlobalMaxPooling2D()(input)
    fc2 = Dense(channel_nums//r_factor)(gm_pooling)
   
    fc = Concatenate()([Activation()(fc1), Activation()(fc2)]) 

    scale = Dense(channel_nums, activation=sigmoid)(fc)
    return multiply([scale, input])


def SE_BLOCK_SAM(input, using_SE=True, r_factor=2):
    # SAM
    if not using_SE:
        return input

    a_pooling = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(input)
    m_pooling = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(input)
    fc = Concatenate()([a_pooling, m_pooling]) 
    weight = Conv2D(1, (3, 3), padding="same", strides=1, use_bias=False, activation=sigmoid)(fc)
    return multiply([weight, input])
      

def SE_BLOCK_YOLO(input, using_SE=True, r_factor=2):
    # SAM yolo-v4
    if not using_SE:
        return input
    weight = Conv2D(1, (5, 5), padding="same", strides=1, use_bias=False, activation=sigmoid)(input)
    return multiply([weight, input])

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

    flat_conv = Flatten()(conv5)
    # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
    # fc or pooling or any ohter operation
    #shape = map(int, conv5.get_shape()[1:])
    #shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

    pmodel = Model(inputs=[input_image], outputs=[flat_conv])
    return pmodel

def build_net(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    print("cates->", CATES, cfeat.shape)
    bulk_feat = Dense(CATES, use_bias=True, activity_regularizer=regularizers.l1(0.), activation=softmax, name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    gender = Dense(2, activation=softmax, activity_regularizer=regularizers.l2(0.), name="gender")(cfeat)

    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in xrange(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat, gender])

def build_net3(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    print(base_model.summary())
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)
    cfeat = Concatenate(axis=-1)([l2_normalize(y1, axis=-1), l2_normalize(y2, axis=-1), l2_normalize(y3, axis=-1)])
    cfeat = BatchNormalization()(cfeat)
    cfeat = Dropout(0.5)(cfeat)
    cfeat = Dense(512, use_bias=False)(cfeat)
    cfeat = BatchNormalization()(cfeat)
    cfeat = l2_normalize(cfeat, axis=-1)
     
    print("cates->", CATES, cfeat.shape)
    bulk_feat = Dense(CATES, use_bias=False, kernel_constraint=unit_norm(axis=0), activation=softmax, name="W1")(16 * cfeat)

    age = Dense(1, use_bias=False, name="age")(bulk_feat)

    gender = Dense(2, use_bias=False, kernel_constraint=unit_norm(axis=0), activation=softmax, name="gender")(cfeat)
    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat, gender])


def preprocessing(dataframes, batch_size=64, category=12, interval=10, is_training=True, dropout=0.):
    # category: bin + 2 due to two side
    # interval: age interval
    from utils import generate_data_generator
    return generate_data_generator(dataframes, category=category, interval=interval, batch_size=batch_size, is_training=is_training, dropout=dropout)


def train(params):
    from utils import reload_data
    if params.fp16:
        os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

    sample_rate, seed, batch_size, category, interval = 0.8, 2019, params.batch_size, params.category + 2, int(math.ceil(100. / params.category))
    lr = params.learning_rate
    data_dir, file_ptn = params.dataset, params.source
    dataframes = reload_data(data_dir, file_ptn)
    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
    print(testset.gender)
    train_gen = preprocessing(trainset, dropout=params.dropout, category=category, interval=interval)
    validation_gen = preprocessing(testset, dropout=0, is_training=False, category=category, interval=interval)
    print(trainset.groupby(["age"])["age"].agg("count"))

    print(testset.groupby(["age"]).agg(["count"]))
    age_dist = [trainset["age"][(trainset.age >= x -10) & (trainset.age <= x)].count() for x in range(10, 10 * params.category + 1, 10)]
    age_dist = [age_dist[0]] + age_dist + [age_dist[-1]]
    gender_dist = [trainset["gender"][trainset.gender == 0].count(), trainset["gender"][trainset.gender == 1].count()]
    print(age_dist, gender_dist)

    models = build_net3(category, using_SE=params.se_net, using_white_norm=params.white_norm)
    if params.pretrain_path:
        #models = load_model(params.pretrain_path, custom_objects={"pool2d": pool2d, "ReLU": ReLU, "BatchNormalization": BatchNormalization, "tf": tf, "focal_loss_fixed": focal_loss(age_dist)})
        ret = models.load_weights(params.pretrain_path)
        model_refresh_without_nan(models)

    #optim = SGD(lr=lr, momentum=0.9)
    if params.freeze:
        converter = tf.lite.TFLiteConverter.from_keras_model(models)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
        tflite_model = converter.convert()
        open("profile.tflite", "wb").write(tflite_model) 
        return


    epoch_nums = 160
    optim = Adam(lr=lr)
    if params.fp16:
        optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim)

    print("-----outputs-----", models.outputs)

    models.compile(
        optimizer=optim,
        loss=["mae", focal_loss(age_dist), "categorical_crossentropy"],  # "kullback_leibler_divergence"
        #loss=["mae", focal_loss(age_dist), focal_loss(gender_dist)],  # "kullback_leibler_divergence"
        metrics={"age": "mae", "gender": "acc", "W1": "mae"},
        loss_weights=[1, 10, 30],
        #loss_weights=[1, 10, 100],
        experimental_run_tf_function=False
    )
    W2 = models.get_layer("age")

    callbacks = [
        ModelCheckpoint(params.save_path, monitor='val_age_mae', verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
        ModelCheckpoint(params.save_path, monitor='val_gender_acc', verbose=1, save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir=params.log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
        #ReduceLROnPlateau(monitor='val_age_mae', factor=0.1, patience=10, min_lr=0.00001),
        CosineAnnealingScheduler(epoch_nums, lr, lr / 100)
    ]
    if not params.test:
        history = models.fit(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=epoch_nums, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size, workers=1)
    else:
        models.evaluate(validation_gen, steps=len(testset) / batch_size)


def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='C3AE retry')
    parser.add_argument(
        '-s', '--save_path', default="./model/c3ae_model_v2_$message$_{epoch}_{val_age_mae:02f}-{val_gender_acc:.3f}", type=str,
        help='the best model to save')
    parser.add_argument(
        '-l', '--log_dir', default="./logs", type=str,
        help='the tensorboard log to save')
    parser.add_argument(
        '-r', '--r_factor', default=2, type=int,
        help='the r factor of SE')

    parser.add_argument(
        '--source', default="wiki", type=str,
        choices=['asia', 'wiki', 'imdb', 'wiki|imdb', "utk", "utk|asia", "afad", "afad|utk|asia"],
        help='"wiki|imdb" or regrex pattern of feather')

    parser.add_argument(
        '--dataset', default="./dataset/data/", type=str,
        help='the path of dataset to load')

    parser.add_argument(
        '-m', "--message", default="", type=str,
        help='message')

    parser.add_argument(
        '-p', '--pretrain_path', dest="pretrain_path", default="", type=str,
        help='the pretrain path')

    parser.add_argument(
        '-b', '--batch_size', default=128, type=int,
        help='batch size degfault=64')

    parser.add_argument(
        '-a', '--activation', default="relu", type=str,
        help='relu|leakrelu|mish')


    parser.add_argument(
        '-c', '--category', default=10, type=int,
        help='category nums degfault=10, n+2')

    parser.add_argument(
        '-gpu', dest="gpu", action='store_true',
        help='config of GPU')

    parser.add_argument(
        '-fz', dest="freeze", action='store_true',
        help='freeze model')

    parser.add_argument(
        '-test', dest="test", action='store_true',
        help='test')

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

    parser.add_argument(
        '-fp16', dest="fp16", action='store_true',
        help='mix precision training')


    params = parser.parse_args()
    params.save_path = params.save_path.replace("$message$", params.message)
    print("!!-----", params.save_path)
    config_gpu() if params.gpu else config_cpu()
    return params

if __name__ == "__main__":
    params = init_parse()
    train(params)
