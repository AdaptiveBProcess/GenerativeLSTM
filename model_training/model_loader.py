# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:53:54 2020

@author: Manuel Camargo
"""
from model_training.models import model_specialized as mspec
from model_training.models import model_concatenated as mcat
from model_training.models import model_time_act as mtime


from model_training.models import model_shared_cat as mshcat
from model_training.models import model_shared_cat_intercase as mshcati
from model_training.models import model_seq2seq as seq
from model_training.models import model_seq2seq_intercase as seqi
from model_training.models import model_cnn_lstm as cnnl
from model_training.models import model_cnn_lstm_intercase as cnnli


class ModelLoader():

    def __init__(self, parms):
        self.parms = parms
        self._trainers = dict()
        self.trainer_dispatcher = {'specialized': mspec._training_model,
                                   'concatenated': mcat._training_model,
                                   'shared_cat': mshcat._training_model,
                                   'shared_cat_inter': mshcati._training_model,
                                   'cnn_lstm': cnnl._training_model}

    def train(self, model_type, examples, ac_weights, rl_weights, output_folder):
        loader = self._get_trainer(model_type)
        loader(examples, ac_weights, rl_weights, output_folder, self.parms)

    def register_model(self, model_type, trainer):
        try:
            self._trainers[model_type] = self.trainer_dispatcher[trainer]
        except KeyError:
            raise ValueError(trainer)

    def _get_trainer(self, model_type):
        trainer = self._trainers.get(model_type)
        if not trainer:
            raise ValueError(model_type)
        return trainer