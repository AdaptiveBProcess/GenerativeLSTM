# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:56:26 2019

@author: Manuel Camargo
"""

import os
import datetime

from keras.callbacks import Callback
 

class CleanSavedModelsCallback(Callback):
    def __init__(self, output_folder, num_models):
        self.logs=[]
        self.num_models=num_models
        self.path=output_folder

    def on_epoch_end(self, epoch, logs={}):
        files = self.create_folder_list(self)
        for file in files:
            os.unlink(os.path.join(self.path, file))
        
    def create_folder_list(self, logs={}): 
        file_list = list()
        for _, _, files in os.walk(self.path):
            files_filtered = list()
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == '.h5':
                    files_filtered.append(f)
            creation_list = list() 
            for f in files_filtered:
                date=os.path.getmtime(os.path.join(self.path, f))
                creation_list.append(dict(filename=f, creation=datetime.datetime.utcfromtimestamp(date)))
            creation_list = sorted(creation_list, key=lambda x:x['creation'], reverse=True)
            for f in creation_list[self.num_models:]:
                file_list.append(f['filename'])
        return file_list

        