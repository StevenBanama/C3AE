#coding=utf-8
import os
import re
import cv2
import feather
import pandas as pd
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer

#COLUMS = ["age", "gender", "image", "width", "height", "source", "md5"]
COLUMS = ["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"]

class MishActivation(Layer):

    def __init__(self, **kwargs):
        super(MishActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MishActivation, self).build(input_shape)

    def call(self, x):
        return x * tf.tanh(tf.math.log(1 + tf.exp(x)))

class MishActivation6(Layer):

    def __init__(self, **kwargs):
        super(MishActivation6, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MishActivation6, self).build(input_shape)

    def call(self, x):
        return tf.clip_by_value(x * tf.tanh(tf.math.log(1 + tf.exp(x))), -6., 6.)

def reload_data(path_dir, file_ptn):
    dataset = pd.DataFrame(columns=COLUMS)
    for rdir, dlist, fnames in os.walk(path_dir):
        fnames = filter(lambda x: x.endswith(".feather"), fnames)
        fnames = list(filter(lambda x: re.search(file_ptn, x), fnames))
        if fnames:
            file_paths = map(lambda name: os.path.join(rdir, name), fnames)
            frames = map(lambda path: feather.read_dataframe(path), file_paths)
            dataset = pd.concat(list(frames), ignore_index=True)
    dataframe = dataset
    dataframe = dataframe[(dataframe["age"] > 0) & (dataframe["age"] < 101)]
    dataframe = dataframe.dropna()
    print("---!!!!", path_dir, file_ptn, dataframe.shape)
    #dataframe = dataframe[(dataframe["yaw"] >-30) & (dataframe["yaw"] < 30) & (dataframe["roll"] >-20) & (dataframe["roll"] < 20) & (dataframe["pitch"] >-20) & (dataframe["pitch"] < 20) ]
    #return process_unbalance(dataframe)
    return dataframe

def process_unbalance(dataframe, max_nums=500, random_seed=2019):
    sample = []
    for x in range(100):
        age_set = dataframe[dataframe.age == x]
        cur_age_num = len(age_set)
        if cur_age_num > max_nums:
            age_set = age_set.sample(max_nums, random_state=random_seed, replace=False)
        sample.append(age_set)
    return pd.concat(sample, ignore_index=True)


def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, category)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < category:
            embed[idx+1] = right_prob
        return embed
    return np.array(age_split(age_label))

def random_erasing(img, drop_out=0.3, aspect=(0.5, 2), area=(0.06, 0.10)):
    # https://arxiv.org/pdf/1708.04896.pdf
    if 1 - random.random() > drop_out:
        return img
    img = img.copy()
    height, width = img.shape[:-1]
    aspect_ratio = np.random.uniform(*aspect)
    area_ratio = np.random.uniform(*area)
    img_area = height * width * area_ratio
    dwidth, dheight = np.sqrt(img_area * aspect_ratio), np.sqrt(img_area * 1 / aspect_ratio) 
    xmin = random.randint(0, height)
    ymin = random.randint(0, width)
    xmax, ymax = min(height, int(xmin + dheight)), min(width, int(ymin + dwidth))
    img[xmin:xmax,ymin:ymax,:] = np.random.random_integers(0, 256, (xmax-xmin, ymax-ymin, 3))
    return img

def get_normal_image(row, seed=100, shape=(96, 96), is_training=True):
    (idx, row) = row[0], row[1]
    img = np.fromstring(row["image"], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = image_enforcing(img, random.randint(0, 8))
    bbox = np.loads(row.org_box)
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3],:]
    img = cv2.resize(img, shape)
    return img
    

class ThresCallback(Callback):
    def __init__(self, candi, watch_dog, thres, val):
        self.candi =candi
        self.watch_dog = watch_dog
        self.thres = thres

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.watch_dog) <= thres:
             K.set_value(self.candi, val)

def model_refresh_without_nan(models):
    import numpy as np                                                                                                                 
    valid_weights = []
    for l in models.get_weights():                                                                                                     
        if np.isnan(l).any():
            valid_weights.append(np.nan_to_num(l))
            print("!!!!!", l)
        else:
            valid_weights.append(l) 
    models.set_weights(valid_weights)


def model_refresh_without_nan(models):
    '''
        https://github.com/tensorflow/tensorflow/issues/38698
    '''
    import numpy as np                                                                                                                 
    valid_weights = []                                                                                                                 
    for l in models.get_weights():                                                                                                     
        if np.isnan(l).any():
            print("!!!!!", l)
            valid_weights.append(np.nan_to_num(l))                                                                                     
        else:                                                                                                                          
            valid_weights.append(l)
    models.set_weights(valid_weights)

def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    # copy from https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-6, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ (total_num / ff if ff != 0 else 0.0) for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        print(classes_w_t2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)
        return fianal_loss
    return focal_loss_fixed

def image_transform(row, seed=100, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15), dropout=0., shape=(64, 64), is_training=True):
    (idx, row) = row[0], row[1]
    img = np.fromstring(row["image"], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #cv2.imwrite("%s_%s__.jpg"%(row.age, row.gender), img)

    if is_training:
        img = random_erasing(img, dropout)

    cascad_imgs, padding = [], 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    height, width = img.shape[:2]
    for bbox in np.loads(row.trible_box, encoding="bytes"):
        h_min, w_min = bbox[0]
        h_max, w_max = bbox[1]
        # cv2.rectangle(img, (h_min, w_min), (h_max, w_max), (0,0,255), 2)
        # cascad_imgs.append(cv2.resize(new_bd_img[max(w_min+padding, 0):min(w_max+padding, width), max(h_min+padding, 0): min(h_max+padding, height)], shape))
        cascad_imgs.append(cv2.resize(new_bd_img[w_min+padding:w_max+padding, h_min+padding: h_max+padding,:], shape))
    ## if you want check data, and then you can remove these marks
    #if idx > 10000:
    #    cv2.imwrite("%s_%s_%s.jpg"%(row.age, row.gender, idx), cascad_imgs[2])
    if is_training:
       flag = random.randint(0, 3)
       contrast = random.uniform(0.5, 2.5)
       bright = random.uniform(-50, 50)
       rotation = random.randint(-15, 15)
       
       cascad_imgs = [image_enforcing(x, flag, contrast, bright, rotation) for x in cascad_imgs]
    return cascad_imgs

class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def image_enforcing(img, flag, contrast, bright, rotation):
    if flag & 1:  # trans hue
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=bright)
    elif flag & 2:  # rotation
        height, width = img.shape[:-1]
        matRotate = cv2.getRotationMatrix2D((height // 2, width // 2), rotation, 1) # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img, matRotate, (height, width))
    elif flag & 4:  # flp 翻转
        img = cv2.flip(img, 1)
    return img


def padding_image(img, padding=0.1):
    height, width = img.shape[:-1]
    p_left, p_top = int(height * padding), int(width * padding)
    img = img[p_top:width-p_left, p_top:height-p_top,:]
    return img

def smooth_label(labels, cls_num, on_value=0.99, epsilon=1e-8):
    from keras.utils.np_utils import to_categorical
    if not (0 <= on_value <= 1 and cls_num > 1):
        raise Exception("fatal params on smooth")
    onehot = to_categorical(labels, 2)
    return np.where(onehot > 0, on_value, (1 - on_value) / (cls_num - 1 + epsilon))

def config_cpu():
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #tf.get_logger().setLevel('ERROR')
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except Exception as ee:
        print(ee)

def config_gpu():
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
       try:
           # Currently, memory growth needs to be the same across GPUs
           for gpu in gpus:
               tf.config.experimental.set_memory_growth(gpu, True)
       except RuntimeError as e:
           # Memory growth must be set before GPUs have been initialized
           print(e)


def generate_data_generator(dataframe, batch_size=32, category=12, interval=10, is_training=True, dropout=0.):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe)
    from keras.utils.np_utils import to_categorical
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            candis = dataframe.ix[list(idxs[start:start+batch_size])]
            imgs = np.array([image_transform(x, is_training=is_training, dropout=dropout) for x in candis.iterrows()])
            gender = smooth_label(candis.gender.to_numpy(), 2, 0.99)
            out2 = [candis.age.to_numpy(), np.array([two_point(x, category, interval) for x in candis.age.to_numpy()]), gender]
            yield [imgs[:,0], imgs[:,1], imgs[:,2]], out2
            start += batch_size

def age_data_generator(dataframe, batch_size=32, category=12, interval=10, is_training=True, dropout=0.):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe)
    from keras.utils.np_utils import to_categorical
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            candis = dataframe.ix[list(idxs[start:start+batch_size])]
            imgs = np.array([image_transform(x, is_training=is_training, dropout=dropout) for x in candis.iterrows()])
            out2 = [candis.age.to_numpy(), np.array([two_point(x, category, interval) for x in candis.age.to_numpy()])]
            yield [imgs[:,0], imgs[:,1], imgs[:,2]], out2
            start += batch_size

if __name__ == "__main__":
    #cv2.imwrite("%s.jpg"%idx, ig)
    print(two_point(32, 12, 10))
