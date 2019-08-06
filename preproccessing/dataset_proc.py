#coding=utf-8
import sys

import argparse
import os
import time
from datetime import datetime
from scipy.io import loadmat
import cv2
import json
import mxnet as mx
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pyutils.funlib.tools import gen_md5
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from serilize import BaseProc
from pose import get_rotation_angle

'''
    所有数据固化传输以feather(), feather文件不能超过2g需要做分片
'''
COLUMS = ["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"]

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def gen_boundbox(box, landmark):
    ymin, xmin, ymax, xmax = map(int, [box[1], box[0], box[3], box[2]])
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark[2], landmark[2+5])
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # 包含五官最小的框
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # 外
        [(nose_x - top2nose, nose_y - top2nose), (nose_x+top2nose, nose_y + top2nose)],  # 中间框
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # 内层框
    ])

def gen_face(image, image_path=""):
    ret = MTCNN_DETECT.detect_face(image) 
    if not ret:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = ret
    if len(bounds) > 1:
        raise Exception("more than one face %s"%image_path)
    return ret


MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)

class ProfileProc(BaseProc):

    def __init__(self, name, data_dir, output_dir, overwrite=False, tf_dir="../data/", sample_rate=0.1):
        BaseProc.__init__(self, name, data_dir, output_dir, COLUMS, overwrite, tf_dir, sample_rate)

    def _trans2tf_record(self, dataframe, trunck_num, sub_dir="train"):
        logging.info("not implemented %s"%sub_dir)

class WikiProc(ProfileProc):

    def __init__(self, data_dir, output_dir, mat_file="wiki.mat", *args, **kwargs):
        ProfileProc.__init__(self, "wiki", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

    def _process(self, nums=-1):
        mat_path = os.path.join(self.data_dir, self.mat_file)
        logging.info(mat_path)
        meta = loadmat(mat_path)
        full_path = meta[self.name][0, 0]["full_path"][0][:nums]
        dob = meta[self.name][0, 0]["dob"][0][:nums]  # Matlab serial date number
        gender = meta[self.name][0, 0]["gender"][0][:nums]
        photo_taken = meta[self.name][0, 0]["photo_taken"][0][:nums]  # year
        face_score = meta[self.name][0, 0]["face_score"][0][:nums]
        second_face_score = meta[self.name][0, 0]["second_face_score"][0][:nums]

        age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        org_box, trible_box, raw_images, landmarks, rolls, yaws, pitches = self.crop_and_trans_images(self.data_dir, full_path, second_face_score, age, gender)
        data = {"gender": gender, "age": age, "score": face_score,
               "image": raw_images, "org_box": org_box, "trible_box": trible_box,
               "landmarks": landmarks, "roll": rolls, "yaw": yaws, "pitch": pitches}
        dataset1 = pd.DataFrame(data)
        dataset1 = dataset1[(dataset1.score > 0.75) & (dataset1.age > 0) & (dataset1.age < 100)]
        dataset1 = dataset1[COLUMS]
        dataset1 = dataset1[dataset1.landmarks != np.array([]).dumps()]
        dataset1 = dataset1.dropna(axis=0)
        self.dataframe = dataset1

    def crop_and_trans_images(self, dirpath, full_path, second_face_score, age, gender):
        # imdb 数据存在多张人脸，所以对于多人脸的数据直接清除掉
        org_boxes, trible_boxes, raw_images, landmarks, rolls, yaws, pitches = [], [], [], [], [], [], []

        for idx, file_name in enumerate(full_path):
            image_path = os.path.join(dirpath, file_name[0])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            try:
                bounds, lmarks = gen_face(image)
                crops = MTCNN_DETECT.extract_image_chips(image, lmarks, padding=0.4)
                if len(crops) == 0:
                    raise Exception("no crops~~ %s"%image_path)
                bounds, lmarks = gen_face(crops[0])
                org_box, first_lmarks = bounds[0], lmarks[0]
                trible_box = gen_boundbox(org_box, first_lmarks)
                pitch, yaw, roll = get_rotation_angle(image, first_lmarks)
                image = crops[0]   # aliagn and replace
            except Exception as ee:
                logging.info("exception as ee: %s"%ee)
                trible_box = np.array([])
                org_box, first_lmarks = np.array([]), np.array([])
                pitch, yaw, roll = 90, 90, 90
            status, buf = cv2.imencode(".jpg", image)
            raw_images.append(buf.tostring())
            org_boxes.append(org_box.dumps())  # xmin, ymin, xmax, ymax
            landmarks.append(first_lmarks.dumps())  # y1..y5, x1..x5
            trible_boxes.append(trible_box.dumps())
            pitches.append(pitch)
            yaws.append(yaw)
            rolls.append(roll)

        return org_boxes, trible_boxes, raw_images, landmarks, rolls, yaws, pitches

    def rectify_data(self):
        logging.info(self.dataframe.groupby(["age"]).agg(["count"]))
        sample = []
        max_nums = 500.0
        for x in xrange(100):
            age_set = self.dataframe[self.dataframe.age == x]
            cur_age_num = len(age_set)
            if cur_age_num > max_nums:
                age_set = age_set.sample(frac=max_nums / cur_age_num, random_state=2007)
            sample.append(age_set)
        self.dataframe = pd.concat(sample, ignore_index=True)
        self.dataframe.age = self.dataframe.age 
        logging.info(self.dataframe.groupby(["age", "gender"]).agg(["count"]))


class ImdbProc(WikiProc):
    def __init__(self, data_dir, output_dir, mat_file="imdb.mat", *args, **kwargs):
        ProfileProc.__init__(self, "imdb", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file



def test_align():
     image = cv2.imread("nets/99_1.0.jpg")
     bounds, lmarks = gen_face(image)
     crops = MTCNN_DETECT.extract_image_chips(image, lmarks, padding=0.4)
     if len(crops) == 0:
         raise Exception("no crops~~ %s"%image_path)
     bounds, lmarks = gen_face(crops[0])
     org_box, first_lmarks = bounds[0], lmarks[0]
     trible_box = gen_boundbox(org_box, first_lmarks)
     pitch, yaw, roll = get_rotation_angle(image, first_lmarks)
     print(pitch, yaw, roll)
     cv2.imwrite("test.jpg", crops[0])

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    WikiProc("/data/bin.wen/repos/ry_cv_services/caster/profile/dataset/wiki_crop", "./dataset/data", overwrite=True).process(nums=-1)
    #ImdbProc("/data/build/dataset/imdb_crop", "./dataset/data",  overwrite=True).process(nums=-1)
