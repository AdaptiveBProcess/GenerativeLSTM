# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:01:20 2021

@author: Manuel Camargo
"""
import os
import sys
import csv
import json
import getopt

import numpy as np
import pandas as pd
import configparser as cp

import utils.support as sup
import readers.log_splitter as ls
import readers.log_reader as lr

import tensorflow as tf
import samples_creator as sc
import features_manager as feat

from models import model_specialized as mspec
from models import model_concatenated as mcat
from models import model_shared_cat as mshcat

from models import model_gru_specialized as mspecg
from models import model_gru_concatenated as mcatg
from models import model_gru_shared_cat as mshcatg

from models import model_shared_cat_cx as mshcati
from models import model_concatenated_cx as mcati
from models import model_gru_concatenated_cx as mcatgi
from models import model_gru_shared_cat_cx as mshcatgi

class SlurmWorker():
    """
    Hyperparameter-optimizer class
    """
    def __init__(self, argv):
        """constructor"""
        self.parms = dict()
        try:
            opts, _ = getopt.getopt( argv, "h:p:f:r:", 
                                    ['parms_file=', 
                                     'output_folder=', 
                                     'res_files='])
            for opt, arg in opts:
                key = self.catch_parameter(opt)
                self.parms[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
        # output_folder = os.path.join('..', 'output_files')
        # Output Files
        self.temp_output = os.path.join(os.getcwd(),
                                        self.parms['output_folder'])
        self.res_files = os.path.join(self.temp_output,
                                      self.parms['res_files'])
        self.load_parameters()
        column_names = {'Case ID': 'caseid',
                        'Activity': 'task',
                        'lifecycle:transition': 'event_type',
                        'Resource': 'user'}
        self.parms['one_timestamp'] = False  # Only one timestamp in the log
        self.parms['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': self.parms['one_timestamp'],
            'filter_d_attrib': False}
        print(self.parms)
        self.log = self.load_log_test(
            os.path.join(self.temp_output, 'opt_parms'), self.parms)
        self.read_embeddings(self.parms)
        loss = self.exec_pipeline()
        self._define_response(self.parms, loss)
        print('COMPLETED')


    @staticmethod
    def catch_parameter(opt):
        """Change the captured parameters names"""
        switch = {'-h': 'help', '-p': 'parms_file', 
                  '-f': 'output_folder', '-r': 'res_files'}
        return switch[opt]

    @staticmethod
    def load_log_test(output_route, parms):
        df_train = lr.LogReader(
            os.path.join(output_route, 'train.csv'),
            parms['read_options'])
        df_train = pd.DataFrame(df_train.data)
        df_train = df_train[~df_train.task.isin(['Start', 'End'])]
        return df_train

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.temp_output, 
                            'opt_parms',
                            self.parms['parms_file'])
        with open(path) as file:
            data = json.load(file)
            parms = {k: v for k, v in data.items()}
            self.parms = {**self.parms, **parms}
            self.ac_index = {k: int(v) for k, v in data['ac_index'].items()}
            self.rl_index = {k: int(v) for k, v in data['rl_index'].items()}
            file.close()
        self.index_ac = {v: k for k, v in self.ac_index.items()}
        self.index_rl = {v: k for k, v in self.rl_index.items()}
    
    def read_embeddings(self, params):
        # Load embedded matrix
        ac_emb_name = 'ac_' + params['file'].split('.')[0]+'.emb'
        rl_emb_name = 'rl_' + params['file'].split('.')[0]+'.emb'
        if os.path.exists(os.path.join(os.getcwd(),
                                       'input_files',
                                       'embedded_matix',
                                        ac_emb_name)):
            self.ac_weights = self.load_embedded(self.index_ac, ac_emb_name)
            self.rl_weights = self.load_embedded(self.index_rl, rl_emb_name)

    def exec_pipeline(self):
        print(self.parms)
        # status = STATUS_OK
        # Path redefinition
        self.parms = self._temp_path_redef(self.parms)
        # Model definition
        model_def = self.read_model_definition(self.parms['model_type'])
        # Scale values
        log, self.parms = self._scale_values(self.log, self.parms, model_def)
        # split validation
        log_valdn, log_train = self.split_timeline(
            0.8, log, self.parms['one_timestamp'])
        print('train split size:', len(log_train))
        print('valdn split size:', len(log_valdn))
        # Vectorize input
        vectorizer = sc.SequencesCreator(
            self.parms['read_options']['one_timestamp'], 
            self.ac_index, self.rl_index)
        vectorizer.register_vectorizer(self.parms['model_type'],
                                       model_def['vectorizer'])
        train_vec = vectorizer.vectorize(self.parms['model_type'],
                                         log_train,
                                         self.parms,
                                         model_def['additional_columns'])
        valdn_vec = vectorizer.vectorize(self.parms['model_type'],
                                         log_valdn,
                                         self.parms,
                                         model_def['additional_columns'])
        # Train
        m_loader = ModelLoader(self.parms)
        m_loader.register_model(self.parms['model_type'],
                                model_def['trainer'])
        tf.compat.v1.reset_default_graph()
        model = m_loader.train(self.parms['model_type'],
                               train_vec, 
                               valdn_vec,
                               self.ac_weights,
                               self.rl_weights,
                               self.parms['output'],
                               os.path.join(os.getcwd(), 
                                            'output_files', 
                                            'training_times.csv'))
        # evaluation
        x_input = {'ac_input': valdn_vec['prefixes']['activities'],
                   'rl_input': valdn_vec['prefixes']['roles'],
                   't_input': valdn_vec['prefixes']['times']}
        if self.parms['model_type'] in ['shared_cat_cx', 
                                       'concatenated_cx',
                                       'shared_cat_gru_cx', 
                                       'concatenated_gru_cx']:
            x_input['inter_input']= valdn_vec['prefixes']['inter_attr']
        acc = model.evaluate(
            x=x_input,
            y={'act_output': valdn_vec['next_evt']['activities'],
               'role_output': valdn_vec['next_evt']['roles'],
               'time_output': valdn_vec['next_evt']['times']},
            return_dict=True)
        # rsp = self._define_response(self.parms, status, acc['loss'])
        print("-- End of trial --")
        return acc['loss']

    def _temp_path_redef(self, settings, **kwargs) -> dict:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
        return settings

    @staticmethod
    def _scale_values(log, params, model_def):
        # Features treatement
        inp = feat.FeaturesMannager(params)
        # Register scaler
        inp.register_scaler(params['model_type'], model_def['scaler'])
        # Scale features
        log, params['scale_args'] = inp.calculate(
            log, model_def['additional_columns'])
        return log, params

    def _define_response(self, parms, loss, **kwargs) -> None:
        measurements = list()
        measurements.append({'loss': loss,
                             'sim_metric': 'val_loss',
                             'n_size': parms['n_size'],
                             'l_size': parms['l_size'],
                             'lstm_act': parms['lstm_act'],
                             'dense_act': parms['dense_act'],
                             'model_type': parms['model_type'],
                             'norm_method': parms['norm_method'],
                             'scale_args': parms['scale_args'],
                             'optim': parms['optim'],
                             'output': parms['output'][len(os.getcwd())+1:]})
        if os.path.getsize(self.res_files) > 0:
            sup.create_csv_file(measurements, self.res_files, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.res_files)

    @staticmethod
    def split_timeline(size: float, log: pd.DataFrame, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(log)
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(log)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        log_valdn = (valdn.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        log_train = (train.sort_values(key, ascending=True)
                          .reset_index(drop=True))
        return log_valdn, log_train

    @staticmethod
    def read_model_definition(model_type):
        model_def = dict()
        Config = cp.ConfigParser(interpolation=None)
        Config.read(os.path.join(os.getcwd(), 'models_spec.ini'))
        #File name with extension
        model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type,'additional_columns'), dtype='str')
        model_def['scaler'] = Config.get(
            model_type, 'scaler')
        model_def['vectorizer'] = Config.get(
            model_type, 'vectorizer')
        model_def['trainer'] = Config.get(
            model_type, 'trainer')
        return model_def

    @staticmethod
    def load_embedded(index, filename):
        """Loading of the embedded matrices.
        parms:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
        """
        weights = list()
        input_folder = os.path.join(os.getcwd(), 'input_files', 'embedded_matix')
        with open(os.path.join(input_folder, filename), 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in filereader:
                cat_ix = int(row[0])
                if index[cat_ix] == row[1].strip():
                    weights.append([float(x) for x in row[2:]])
            csvfile.close()
        return np.array(weights)


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
                                   'shared_cat_gru_cx': mshcatgi._training_model}

    def train(self, model_type, train_vec, valdn_vec, ac_weights, rl_weights, output_folder, log_path):
        loader = self._get_trainer(model_type)
        return loader(train_vec, 
                      valdn_vec, 
                      ac_weights, 
                      rl_weights, 
                      output_folder, 
                      self.parms,
                      log_path=log_path)

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

if __name__ == "__main__":
    worker = SlurmWorker(sys.argv[1:])
