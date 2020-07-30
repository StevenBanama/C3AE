#coding=utf-8
import sys

import argparse
import os
import re
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
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from .serilize import BaseProc
from pose import get_rotation_angle

'''
    save all preprocessing to file with feather format
    所有数据固化传输以feather(), feather文件不能超过2g需要做分片
'''
COLUMS = ["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"]
FEMALE = 0
MALE = 1

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def gen_boundbox(box, landmark):
    # gen trible boundbox
    ymin, xmin, ymax, xmax = map(int, [box[1], box[0], box[3], box[2]])
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark[2], landmark[2+5])
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # 包含五官最小的框
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])

def gen_face(detector, image, image_path=""):
    ret = detector.detect_face(image) 
    if not ret:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = ret
    if len(bounds) > 1:
        print("!!!!!,", bounds, lmarks)
        raise Exception("more than one face %s"%image_path)
    return ret

MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)

class ProfileProc(BaseProc):

    def __init__(self, name, data_dir, output_dir, overwrite=False, tf_dir="../data/", sample_rate=0.25):
        BaseProc.__init__(self, name, data_dir, output_dir, COLUMS, overwrite, tf_dir, sample_rate)

    def _trans2tf_record(self, dataframe, trunck_num, sub_dir="train"):
        logging.info("not implemented %s"%sub_dir)

class WikiProc(ProfileProc):
    # 0 for female and 1 for male

    def __init__(self, data_dir, output_dir, mat_file="wiki.mat", *args, **kwargs):
        ProfileProc.__init__(self, "wiki", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

    def _process(self, nums=-1):
        mat_path = os.path.join(self.data_dir, self.mat_file)
        logging.info(mat_path)
        meta = loadmat(mat_path)
        print(meta)
        full_path = [os.path.abspath(os.path.join(self.data_dir, p[0])) for p in meta[self.name][0, 0]["full_path"][0][:nums]]
        dob = meta[self.name][0, 0]["dob"][0][:nums]  # Matlab serial date number
        mat_gender = meta[self.name][0, 0]["gender"][0][:nums]
        photo_taken = meta[self.name][0, 0]["photo_taken"][0][:nums]  # year
        face_score = meta[self.name][0, 0]["face_score"][0][:nums]
        second_face_score = meta[self.name][0, 0]["second_face_score"][0][:nums]

        mat_age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

        mat_2_pd = pd.DataFrame({"full_path": full_path, "age": mat_age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
        rows, cols = mat_2_pd.shape
        frames = [self.sin_task(MTCNN_DETECT, mat_2_pd)]
        self.dataframe = pd.concat(frames, ignore_index=True)

    def sin_task(self, detector, predata):
        dataset1 = predata.apply(lambda x: self.crop_and_trans_images(detector, x), axis=1)

        dataset1 = dataset1[(dataset1.age >= 0) & (dataset1.age <= 100)]

        dataset1 = dataset1[dataset1.gender != np.nan]

        dataset1 = dataset1[COLUMS]
        print("dddddddd",dataset1)
        dataset1 = dataset1[dataset1.landmarks != np.array([]).dumps()]
        return dataset1

    def crop_and_trans_images(self, detector, series):
        # imdb 数据存在多张人脸，所以对于多人脸的数据直接清除掉
        image_path = series.full_path
        try:
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if not np.isnan(series.second_face_score):
                raise Exception("secend face is not None~---%s~-%s- %s"%(series.name, series.age, image_path))
            bounds, lmarks = gen_face(detector, image, image_path)
            crops = detector.extract_image_chips(image, lmarks, padding=0.4)  # aligned face with padding 0.4 in papper
            if len(crops) == 0:
                print("cant detect!!!!")
                raise Exception("no crops~~ %s---%s"%(image_path, series.age))
            if len(crops) > 1:
                raise Exception("more than one face~---%s~-- %s"%(series.name, image_path))
            bounds, lmarks = gen_face(detector, crops[0], image_path)  # recaculate landmar
            org_box, first_lmarks = bounds[0], lmarks[0]
            trible_box = gen_boundbox(org_box, first_lmarks)
            pitch, yaw, roll = get_rotation_angle(crops[0], first_lmarks) # gen face rotation for filtering
            image = crops[0]   # select the first align face and replace
        except Exception as ee:
            logging.info("exception as ee: %s"%ee)
            print(ee)
            trible_box = np.array([])
            org_box, first_lmarks = np.array([]), np.array([])
            pitch, yaw, roll = np.nan, np.nan, np.nan
            age = np.nan
            gender = np.nan
        status, buf = cv2.imencode(".jpg", image)
        series["image"] = buf.tostring() 
        series["org_box"] = org_box.dumps()  # xmin, ymin, xmax, ymax
        series["landmarks"] = first_lmarks.dumps()  # y1..y5, x1..x5
        series["trible_box"] = trible_box.dumps() 
        series["yaw"] = yaw
        series["pitch"] = pitch
        series["roll"] = roll

        return series

    def rectify_data(self):
        logging.info(self.dataframe.groupby(["age"]).agg(["count"]))
        sample = []
        max_nums = 500.0
        for x in range(100):
            age_set = self.dataframe[self.dataframe.age == x]
            cur_age_num = len(age_set)
            if cur_age_num > max_nums:
                age_set = age_set.sample(frac=max_nums / cur_age_num, random_state=2007)
            sample.append(age_set)
        self.dataframe = pd.concat(sample, ignore_index=True)
        self.dataframe.age = self.dataframe.age 
        logging.info(self.dataframe.groupby(["age", "gender"]).agg(["count"]))


class ImdbProc(WikiProc):
    # 0 for female and 1 for male

    def __init__(self, data_dir, output_dir, mat_file="imdb.mat", *args, **kwargs):
        ProfileProc.__init__(self, "imdb", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

class AsiaProc(WikiProc):
    # https://github.com/JingchunCheng/All-Age-Faces-Dataset
    def __init__(self, data_dir, output_dir, mat_file="AsiaAllAge.mat", *args, **kwargs):
        ProfileProc.__init__(self, "asia", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

    def read_dir(self, nums=-1, ptn="(\d+)A(\d+).jpg"):
        paths, genders, ages = [], [], []
        face_scores, second_face_scores = [], [] 
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames[:nums]:
                path = os.path.abspath(os.path.join(root, fname))
                match = re.search(ptn, path)
                if not match:
                    continue
                identity_id, age = match.groups() 
                gender = 0 if int(identity_id) <= 7380 else 1  # 0: female 1: male
                paths.append(path)
                genders.append(gender)
                ages.append(int(age))
                face_scores.append(1.0)
                second_face_scores.append(np.nan)
        return paths, genders, ages, face_scores, second_face_scores

    def _process(self, nums=-1):
        full_path, mat_gender, mat_age, face_score, second_face_score = self.read_dir(nums)
        mat_2_pd = pd.DataFrame({"full_path": full_path, "age": mat_age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
        rows, cols = mat_2_pd.shape
        frames = [self.sin_task(MTCNN_DETECT, mat_2_pd)]

        self.dataframe = pd.concat(frames, ignore_index=True)


class UTKProc(AsiaProc):
    # https://susanqq.github.io/UTKFace/

    def __init__(self, data_dir, output_dir, mat_file="", *args, **kwargs):
        ProfileProc.__init__(self, "utk", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

    def read_dir(self, nums=-1, ptn="(\d+)_(\d+)_(\d+)_.*?\.jpg$"):
        '''
            group(1): age
            group(2): gender 0: man 1: female
            group(3): race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)
        '''
        paths, genders, ages = [], [], []
        face_scores, second_face_scores = [], [] 
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames[:nums]:
                path = os.path.abspath(os.path.join(root, fname))
                match = re.search(ptn, path)
                if not match:
                    continue
                age, gender, race = match.groups() 
                gender = MALE if gender == "0" else FEMALE  # 0: female 1: male to 0:male
                age = int(age)
                paths.append(path)
                genders.append(gender)
                ages.append(int(age))
                face_scores.append(1.0)
                second_face_scores.append(np.nan)
        return paths, genders, ages, face_scores, second_face_scores

class AFADProc(AsiaProc):
    # https://github.com/afad-dataset/tarball

    def __init__(self, data_dir, output_dir, mat_file="", *args, **kwargs):
        ProfileProc.__init__(self, "afad", data_dir, output_dir, *args, **kwargs)
        self.mat_file=mat_file

    def read_dir(self, nums=-1, ptn="(\d+)/(\d+)/.*?jpg$"):
        '''
            group(1): age
            group(2): 112: female 111: male

        '''
        paths, genders, ages = [], [], []
        face_scores, second_face_scores = [], [] 
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames[:nums]:
                path = os.path.abspath(os.path.join(root, fname))
                match = re.search(ptn, path)
                if not match:
                    continue
                age, gender = match.groups() 
                gender = FEMALE if gender == "112" else MALE  # 112: female 111: male
                age = int(age)
                paths.append(path)
                genders.append(gender)
                ages.append(int(age))
                face_scores.append(1.0)
                second_face_scores.append(np.nan)
        return paths, genders, ages, face_scores, second_face_scores



def test_align():
     image = cv2.imread("timg4.jpg")
     bounds, lmarks = gen_face(MTCNN_DETECT, image)
     crops = MTCNN_DETECT.extract_image_chips(image, lmarks, padding=0.4)
     if len(crops) == 0:
         raise Exception("no crops~~ %s"%image_path)
     bounds, lmarks = gen_face(MTCNN_DETECT, crops[0])
     org_box, first_lmarks = bounds[0], lmarks[0]
     trible_box = gen_boundbox(org_box, first_lmarks)
     pitch, yaw, roll = get_rotation_angle(image, first_lmarks)
     print(pitch, yaw, roll)
     cv2.imwrite("test.jpg", crops[0])

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='preprocessing')

    parser.add_argument(
        '--source', default="wiki", type=str,
        choices=['wiki', 'imdb', "asia", "utk", "afad"],
        help='"wiki|imdb" or regrex pattern of feather')

    parser.add_argument(
        '-d', '--dest', default="./dataset/data/", type=str,
        help='save_path')

    parser.add_argument(
        '-i', '--input_path', default="./dataset/wiki_crop", type=str,
        choices=["./dataset/wiki_crop", "./dataset/imdb_crop", "./dataset/asia_align_face", "./dataset/UTKFace", "./dataset/AFAD-Full"],
        help='the path of dataset to load')

    parser.add_argument(
        '-p', '--padding', default=0.4, type=float,
        help='face padding')

    return parser.parse_args()

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    params = init_parse()
    if params.source == "wiki":
        WikiProc(params.input_path, params.dest, overwrite=True).process(nums=-1)
    elif params.source == "imdb":
        ImdbProc(params.input_path, params.dest,  overwrite=True).process(nums=-1)
    elif params.source == "asia":
        AsiaProc(params.input_path, params.dest,  overwrite=True).process(nums=-1)
    elif params.source == "utk":
        UTKProc(params.input_path, params.dest,  overwrite=True).process(nums=-1)
    elif params.source == "afad":
        AFADProc(params.input_path, params.dest,  overwrite=True).process(nums=-1)
    else:
        raise Exception("fatal source")
