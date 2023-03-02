# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:53:54 2020

@author: Manuel Camargo
"""
import tensorflow as tf

from model_training.models import model_specialized as mspec
from model_training.models import model_concatenated as mcat
from model_training.models import model_shared_cat as mshcat

from model_training.models import model_gru_specialized as mspecg
from model_training.models import model_gru_concatenated as mcatg
from model_training.models import model_gru_shared_cat as mshcatg


from model_training.models import model_shared_cat_cx as mshcati
from model_training.models import model_concatenated_cx as mcati
from model_training.models import model_gru_concatenated_cx as mcatgi
from model_training.models import model_gru_shared_cat_cx as mshcatgi

# from model_training.models import model_shared_cat_intercase as mshcati
# from model_training.models import model_concatenated_inter as mcati
# from model_training.models import model_gru_concatenated_inter as mcatgi
# from model_training.models import model_gru_shared_cat_intercase as mshcatgi

class ModelLoader():

    def __init__(self, parms):
        self.parms = parms
        self._trainers = dict()
        self.trainer_dispatcher = {'specialized': mspec._training_model,
                                   'concatenated': mcat._training_model,
                                   'concatenated_cx': mcati._training_model,
                                   'shared_cat': mshcat._training_model,
                                   'shared_cat_cx': mshcati._training_model,
                                   'specialized_gru': mspecg._training_model,
                                   'concatenated_gru': mcatg._training_model,
                                   'concatenated_gru_cx': mcatgi._training_model,
                                   'shared_cat_gru': mshcatg._training_model,
                                   'shared_cat_gru_cx': mshcatgi._training_model,
                                   # 'cnn_lstm': cnnl._training_model,
                                   # 'gan': mgan._training_model
                                   }

    def train(self, model_type, train_vec, valdn_vec, ac_weights, rl_weights, output_folder):
        loader = self._get_trainer(model_type)
        tf.compat.v1.reset_default_graph()
        return loader(train_vec, 
                      valdn_vec, 
                      ac_weights, 
                      rl_weights, 
                      output_folder, 
                      self.parms)

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