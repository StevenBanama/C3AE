#coding=utf-8
import os
import math
import pandas as pd
import tensorflow as tf
import numpy as np
import logging
from sklearn.model_selection import train_test_split


'''
    基础的数据处理基类:
         现在输入的数据必须满足pandas的输入格式, 各自实现输入对应接口
         输出统一成pandas的feather格式
    主要是方便标注后台的数据能够顺利的接入
'''

class BaseProc:

    def __init__(self, name, data_dir, output_dir, heads, overwrite=False, tf_dir="../data/", sample_rate=0.1):
        '''
            @name: proc name
            @data_dir: 预处理数据目录
            @output_dir: 输出目录
            @heads: 数据titles
        '''
        self.data_dir = data_dir
        self.name = name
        self.output_dir = output_dir
        self.heads = heads
        self.dataframe = pd.DataFrame(columns=self.heads)
        self.overwrite = overwrite
        self.tf_dir = tf_dir
        self.can_run = True
        self.sample_rate = sample_rate

    def replica_check(self):
        '''
            检查重复任务
        '''
        if self.overwrite:
             return True
        for dirname, dirs, fnames in os.walk(self.output_dir):
            for fname in fnames:
                if self.name in fname:
                    return False
        return True

    def process(self, *args, **kwargs):
        logging.info("name:%s"%self.name)
        self.can_run = self.replica_check()
        if not self.can_run:
            logging.info("已存在重复文件则不需要重新处理")
            self.reload_data()
        else:
            self._process(*args, **kwargs)
            self.save()
       
        self.dataframe = self.dataframe.dropna(axis=0)
        self.rectify_data()
        self.transtf_record()


    def rectify_data(self):
        '''
            主要对脏数据进行清理
        '''
        pass

    def reload_data(self):
        '''
           重新加载历史数据
        '''
        import feather
        dataset = pd.DataFrame(columns=self.heads)
        for rdir, dlist, fnames in os.walk(self.output_dir):
            fnames = filter(lambda x: x.endswith(".feather"), fnames)
            fnames = filter(lambda x: x.count(self.name), fnames)
            if fnames:
                file_paths = map(lambda name: os.path.join(rdir, name), fnames) 
                frames = map(lambda path: feather.read_dataframe(path), file_paths)
                dataset = pd.concat(frames, ignore_index=True)
        self.dataframe = dataset
        return dataset 


    def _process(self, *args, **kwargs):
        return NotImplemented

    def save(self, chunkSize=5000):
        if not self.can_run:
            return
        chunk_start = 0
        dataframe = self.dataframe.reset_index()[self.heads]
        while(chunk_start < len(self.dataframe)):
            dir_path = os.path.join(self.output_dir, self.name + "_" + str(int(chunk_start / chunkSize)) + ".feather")
            tmp_pd = dataframe[chunk_start:chunk_start + chunkSize].copy().reset_index()
            tmp_pd.to_feather(dir_path)
            chunk_start += chunkSize

    def transtf_record(self, record_size=10000):
        self.train_sets, self.test_sets = train_test_split(self.dataframe, train_size=1 - self.sample_rate, test_size=self.sample_rate, random_state=2017)
        self.train_sets.reset_index(drop=True, inplace=True)
        self.test_sets.reset_index(drop=True, inplace=True)
        train_nums = self.train_sets.shape[0]
        test_nums = self.test_sets.shape[0]

        train_file_nums = int(math.ceil(1.0 * train_nums / record_size))
        test_file_nums = int(math.ceil(1.0 * test_nums/ record_size))
        train_idx = np.linspace(0, train_nums, train_file_nums, dtype=np.int)
        test_idx = np.linspace(0, test_nums, test_file_nums, dtype=np.int)
        for steps in train_idx:
            next_steps = min(steps+record_size, train_nums)
            self._trans2tf_record(self.train_sets[steps:next_steps].copy().reset_index(drop=True), steps // record_size, "train")
        for steps in test_idx:
            next_steps = min(steps+record_size, test_nums)
            self._trans2tf_record(self.test_sets[steps:next_steps].copy().reset_index(drop=True), steps // record_size, "test")

    def _trans2tf_record(self, dataframe, trunck_num, sub_dir="train"):
        """
            各子类需要自行实现相关代码
        """
        return NotImplemented

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
