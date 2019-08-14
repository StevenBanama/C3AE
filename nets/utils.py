#coding=utf-8
import os
import re
import cv2
import feather
import pandas as pd
import numpy as np
import random
import tensorflow as tf

#COLUMS = ["age", "gender", "image", "width", "height", "source", "md5"]
COLUMS = ["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"]

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
    dataframe = dataframe[(dataframe["yaw"] >-30) & (dataframe["yaw"] < 30) & (dataframe["roll"] >-20) & (dataframe["roll"] < 20) & (dataframe["pitch"] >-20) & (dataframe["pitch"] < 20) ]

    return process_unbalance(dataframe)

def process_unbalance(dataframe, max_nums=500, random_seed=2019):
    sample = []
    for x in xrange(100):
        age_set = dataframe[dataframe.age == x]
        cur_age_num = len(age_set)
        if cur_age_num > max_nums:
            age_set = age_set.sample(max_nums, random_state=random_seed, replace=False)
        sample.append(age_set)
    return pd.concat(sample, ignore_index=True)


def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in xrange(0, category)]
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

def image_transform(row, seed=100, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15), dropout=0., shape=(64, 64), is_training=True): 
    (idx, row) = row[0], row[1]
    img = np.fromstring(row["image"], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #cv2.imwrite("%s_%s__.jpg"%(row.age, row.gender), img)

    if is_training:
        img = random_erasing(img, dropout)

    cascad_imgs, padding = [], 0
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    height, width = img.shape[:2]
    for bbox in np.loads(row.trible_box):
        h_min, w_min = bbox[0]
        h_max, w_max = bbox[1]
        #cv2.rectangle(img, (h_min, w_min), (h_max, w_max), (0,0,255), 2)
        cascad_imgs.append(cv2.resize(new_bd_img[max(w_min+padding, 0):min(w_max+padding, width), max(h_min+padding, 0): min(h_max+padding, height)], shape))
    ## if you want check data, and then you can remove these marks
    #if idx > 10000:
    #    cv2.imwrite("%s_%s_%s.jpg"%(row.age, row.gender, idx), cascad_imgs[2])
    if is_training:
       flag = random.randint(0, 3)
       cascad_imgs = map(lambda x: image_enforcing(x, flag, contrast, bright, rotation), cascad_imgs)
    return cascad_imgs

def image_enforcing(img, flag=0, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15)):
    if flag == 1:  # trans hue
        #img = cv2.convertScaleAbs(img, alpha=random.uniform(*contrast), beta=random.uniform(*bright))
        pass
    elif flag == 2:  # rotation
        #height, width = img.shape[:-1]
        #matRotate = cv2.getRotationMatrix2D((height, width), random.randint(-15, 15), 1) # mat rotate 1 center 2 angle 3 缩放系数
        #img = cv2.warpAffine(img, matRotate, (height, width))
        pass
    elif flag == 3:  # flp 翻转
        img = cv2.flip(img, 1)
    return img


def padding_image(img, padding=0.1):
    height, width = img.shape[:-1]
    p_left, p_top = int(height * padding), int(width * padding)
    img = img[p_top:width-p_left, p_top:height-p_top,:]
    return img

def generate_data_generator(dataframe, batch_size=32, category=12, interval=10, is_training=True, dropout=0.):
    while True:
        candis = dataframe.sample(batch_size, replace=False)
        imgs = np.array(map(lambda x: image_transform(x, is_training=is_training, dropout=dropout), candis.iterrows()))
        out2 = [candis.age.to_numpy(), np.array(map(lambda x: two_point(x, category, interval), candis.age.to_numpy()))]
        yield [imgs[:,0], imgs[:,1], imgs[:,2]], out2


if __name__ == "__main__":
    #cv2.imwrite("%s.jpg"%idx, ig)
    print(two_point(32, 12, 10))
