# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:53:54 2020

@author: Manuel Camargo
"""
from model_training.models import model_shared_cat as mshcat
from model_training.models import model_shared_cat_intercase as mshcati
from model_training.models import model_seq2seq as seq
from model_training.models import model_seq2seq_intercase as seqi
from model_training.models import model_cnn_lstm as cnnl
from model_training.models import model_cnn_lstm_intercase as cnnli

class ModelLoader():

    def __init__(self, parms):
        self.parms = parms

    def train(self, model_type, examples, ac_weights, rl_weights, output_folder):
        loader = self._get_trainer(model_type)
        loader(examples, ac_weights, rl_weights, output_folder, self.parms)

    def _get_trainer(self, model_type):
        if model_type == 'shared_cat':
            return mshcat._training_model
        elif model_type == 'shared_cat_inter':
            return mshcati._training_model
        elif model_type == 'shared_cat_inter_full':
            return mshcati._training_model
        elif model_type == 'shared_cat_rd':
            return mshcati._training_model
        elif model_type == 'shared_cat_wl':
            return mshcati._training_model
        elif model_type == 'shared_cat_cx':
            return mshcati._training_model
        elif model_type == 'shared_cat_city':
            return mshcati._training_model
        elif model_type == 'shared_cat_snap':
            return mshcati._training_model
        elif model_type == 'cnn_lstm':
            return cnnl._training_model
        elif model_type == 'cnn_lstm_inter':
            return cnnli._training_model
        elif model_type == 'cnn_lstm_inter_full':
            return cnnli._training_model
        elif model_type == 'seq2seq':
            return seq._training_model
        elif model_type == 'seq2seq_inter':
            return seqi._training_model
        else:
            raise ValueError(model_type)
