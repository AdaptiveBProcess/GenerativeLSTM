# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:21:35 2019

@author: Manuel Camargo
"""
import os
import numpy as np

from time import time
from keras.callbacks import Callback
from support_modules import support as sup
 

class TimingCallback(Callback):
    def __init__(self, output_folder):
        self.logs=[]
        self.output_folder=output_folder
        
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time()-self.starttime)
    def on_train_end(self, logs={}):
        log_file = os.path.join('output_files', 'training_times.csv')
        data = [{'output_folder': self.output_folder,
                'train_epochs': len(self.logs),
                'avg_time': np.mean(self.logs),
                'min_time': np.min(self.logs),
                'max_time': np.max(self.logs)}]
        if os.path.exists(log_file):
            sup.create_csv_file(data, log_file, mode='a')
        else:
            sup.create_csv_file_header(data, log_file)
        