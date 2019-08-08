#coding=utf-8
import cv2
import os
import re
import feather
import base64
import numpy as np
import pandas as pd
import random
import keras.backend as K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, load_model
from keras.engine.input_layer import Input
from keras.layers.core import Dense, Dropout
from keras.layers import Lambda, BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.backend import argmax, pool2d
from keras import backend as K
from utils import reload_data


def l1_smooth(x):
    return K.cast(K.sum(x * x), dtype="float32")

def smooth_l1_ce_loss(y_true, y_pred):
    # we use smoth l1 for adding constance relation of age
    ture_arg_max = K.cast(K.argmax(y_true, axis = -1), "float32")
    pred_arg_max = K.cast(K.argmax(y_pred, axis = -1), "float32")
    diff = K.abs(ture_arg_max - pred_arg_max)
    ls = K.switch(diff < 3, (0.3 * diff * diff), (diff - 0.3)) * categorical_crossentropy(y_true, y_pred)
    return K.mean(ls)

def smooth_l1_fl_loss(y_true, y_pred):
    # we use smoth l1 for adding constance relation of age focal loss version
    ture_arg_max = K.cast(K.argmax(y_true, axis = -1), "float32")
    pred_arg_max = K.cast(K.argmax(y_pred, axis = -1), "float32")
    factor_alpha = (1 - K.sum(y_true * y_pred, axis=-1))
    factor_alpha = factor_alpha * factor_alpha
    factor_offset = K.abs(ture_arg_max - 25) // 5 + 1
    diff = K.abs(ture_arg_max - pred_arg_max)
    ls = K.switch(diff < 3, (0.3 * diff * diff), (diff - 0.3)) * factor_offset * factor_alpha * categorical_crossentropy(y_true, y_pred)
    return K.mean(ls)

def metrix_age_diff(y_true, y_pred):
    ture_arg_max = K.cast(K.argmax(y_true, axis = -1), "float32")
    pred_arg_max = K.cast(K.argmax(y_pred, axis = -1), "float32")
    diff = K.abs(ture_arg_max - pred_arg_max)
    return K.mean(diff)

def kmse(y_true, y_pred):
    ture_arg_max = K.cast(K.argmax(y_true, axis = -1), "float32")
    pred_arg_max = K.cast(K.argmax(y_pred, axis = -1), "float32")
    diff = K.pow(ture_arg_max - pred_arg_max, 2)
    return K.mean(diff)

def restore_model(model_name=".model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", dropout=0.5, compile=True):
    height, width, channel = 96, 96, 3
    if os.path.exists(model_name):
         pmodel = load_model(model_name, custom_objects={"smooth_l1_fl_loss": smooth_l1_fl_loss, "metrix_age_diff": metrix_age_diff, "kmse": kmse}, compile=compile)
    else:
        input_image = Input(shape=(height, width, channel))
        mobilev2_model = MobileNetV2(input_shape=(height, width, channel), alpha=0.35)
        #for y in mobilev2_model.layers:
        #    y.trainable = False

        m2v = mobilev2_model(input_image)
        c = Dropout(rate=1 - dropout, seed=1000)(m2v)
        gender = Dense(2, activation="softmax", name="gender")(c)
        age = Dense(100, activation='softmax', name="age")(c)
        age_identy = Lambda(lambda x: x, name="age_identy")(age)
        predict_age = K.argmax(age_identy, axis = -1)
        pmodel = Model(input=input_image, output=[gender, age, age_identy])
    return pmodel

def image_process(row, seed=100, contrast=(0.5, 2.5), bright=(-50, 50)):
    (idx, row) = row[0], row[1]
    img = np.fromstring(row["image"], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    width, height = 96, 96
    img = cv2.resize(img, (width, height))
    img = cv2.convertScaleAbs(img, alpha=random.uniform(*contrast), beta=random.uniform(*bright))
    matRotate = cv2.getRotationMatrix2D((width, height), random.randint(-15, 15), 1) # mat rotate 1 center 2 angle 3 缩放系数
    img = cv2.warpAffine(img, matRotate, (width, height))
    if random.randint(0, 1):  # 翻转
       img = cv2.flip(img, 1)
    return img

def preprocessing(dataframes, batch_size=32):
    return generate_data_generator(dataframes) 

def generate_data_generator(dataframe, batch_size=32, is_training=True):
    from sklearn.preprocessing import OneHotEncoder
    from keras.utils import to_categorical
 
    while True:
        candis = dataframe.sample(batch_size, replace=False)
        imgs = np.array(map(lambda x: image_process(x), candis.iterrows()))
        out2 = [to_categorical(candis.gender, num_classes=2), to_categorical(candis.age, num_classes=100), to_categorical(candis.age, num_classes=100)]
        yield imgs, out2

def init_config():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config)) 

def main():
    init_config()
    sample_rate, seed = 0.8, 2019
    data_dir, file_ptn = "./dataset/data", "imdb|wiki"
    dataframes = reload_data(data_dir, file_ptn)
    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
    train_gen = preprocessing(trainset)
    validation_gen = preprocessing(testset)
    model_name = "./model/mobilev2.h5"
    pmodel = restore_model(model_name)
    pmodel.compile(optimizer="Adam",
        loss=["categorical_crossentropy", smooth_l1_fl_loss, "categorical_crossentropy"],
        metrics={"gender": "acc", "age": metrix_age_diff, "age_identy": kmse},
        loss_weights=[50, 0.5, 0]
    )
    batch_size = 32
    callbacks = [ModelCheckpoint(model_name, monitor='val_gender_acc', verbose=1, save_best_only=False, mode='max')]
    history = pmodel.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=200, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size * 3)


def test(img_dir="./img/", out_dir=None):
    model = restore_model("model/mobilev2.h5", 0)
    for root, dirname, filenames in os.walk(img_dir):
        for filename in filenames:
            img = cv2.imread(os.path.join(root, filename))
            img = cv2.resize(img, (96, 96))
            result = model.predict(np.array([img]))
            print(filename, np.argmax(result[1]), np.argmax(result[0]))

if __name__ == "__main__":
    main()
