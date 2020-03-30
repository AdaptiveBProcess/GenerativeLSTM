# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:07:19 2020

@author: Manuel Camargo
"""
import os
import csv

import pandas as pd
import numpy as np

from support_modules.readers import log_reader as lr
from support_modules import nn_support as nsup
from support_modules import support as sup


from model_training import examples_creator as exc
from model_training import features_manager as feat
from model_training import model_loader as mload


class ModelTrainer():
    """
    This is the man class encharged of the model training
    """

    def __init__(self, params):
        """constructor"""
        self.log = self.load_log(params)
        self.output_folder = os.path.join('output_files', sup.folder_id())
        # Split validation partitions
        self.log_train = pd.DataFrame()
        self.log_test = pd.DataFrame()
        # Activities and roles indexes
        self.ac_index = dict()
        self.index_ac = dict()

        self.rl_index = dict()
        self.index_rl = dict()
        # Training examples
        self.examples = dict()
        # Embedded dimensions
        self.ac_weights = list()
        self.rl_weights = list()
        # Preprocess the event-log
        self.preprocess(params)
        # Train model
        m_loader = mload.ModelLoader(params)
        m_loader.train(params['model_type'],
                       self.examples,
                       self.ac_weights,
                       self.rl_weights,
                       self.output_folder)

    def preprocess(self, params):
        # Features treatement
        inp = feat.FeaturesMannager(params)
        self.log = inp.calculate(params, self.log)
        # indexes creation
        self.indexing()
        # split validation
        self.split_train_test(0.3, params['one_timestamp'])
        # create examples
        seq_creator = exc.SequencesCreator(self.log_train,
                                           self.ac_index,
                                           self.rl_index)
        self.examples = seq_creator.vectorize(params['model_type'], params)
        # Load embedded matrix
        self.ac_weights = self.load_embedded(
            self.index_ac, 'ac_' + params['file_name'].split('.')[0]+'.emb')
        self.rl_weights = self.load_embedded(
            self.index_rl, 'rl_' + params['file_name'].split('.')[0]+'.emb')
        # Export parameters
        self.export_parms(params)

    @staticmethod
    def load_log(params):
        loader = LogLoader(os.path.join('input_files', params['file_name']),
                           params['read_options'])
        return loader.load(params['model_type'])

    def indexing(self):
        # Activities index creation
        self.ac_index = self.create_index(self.log, 'task')
        self.ac_index['start'] = 0
        self.ac_index['end'] = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}
        # Roles index creation
        self.rl_index = self.create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end'] = len(self.rl_index)
        self.index_rl = {v: k for k, v in self.rl_index.items()}
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log['ac_index'] = self.log.apply(ac_idx, axis=1)
        rl_idx = lambda x: self.rl_index[x['role']]
        self.log['rl_index'] = self.log.apply(rl_idx, axis=1)

    @staticmethod
    def create_index(log_df, column):
        """Creates an idx for a categorical attribute.
        parms:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
        """
        temp_list = log_df[[column]].values.tolist()
        subsec_set = {(x[0]) for x in temp_list}
        subsec_set = sorted(list(subsec_set))
        alias = dict()
        for i, _ in enumerate(subsec_set):
            alias[subsec_set[i]] = i + 1
        return alias

    def split_train_test(self, percentage: float, one_timestamp: bool) -> None:
        """
        Split an event log dataframe to peform split-validation

        Parameters
        ----------
        percentage : float, validation percentage.
        one_timestamp : bool, Support only one timestamp.
        """
        cases = self.log.caseid.unique()
        num_test_cases = int(np.round(len(cases)*percentage))
        test_cases = cases[:num_test_cases]
        train_cases = cases[num_test_cases:]
        df_test = self.log[self.log.caseid.isin(test_cases)]
        df_train = self.log[self.log.caseid.isin(train_cases)]
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        self.log_test = (df_test
                         .sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = (df_train
                          .sort_values(key, ascending=True)
                          .reset_index(drop=True))

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
        input_folder = os.path.join('input_files', 'embedded_matix')
        with open(os.path.join(input_folder, filename), 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in filereader:
                cat_ix = int(row[0])
                if index[cat_ix] == row[1].strip():
                    weights.append([float(x) for x in row[2:]])
            csvfile.close()
        return np.array(weights)

    def export_parms(self, parms):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            os.makedirs(os.path.join(self.output_folder, 'parameters'))

        parms['index_ac'] = self.index_ac
        parms['index_rl'] = self.index_rl
        if parms['model_type'] in ['shared_cat', 'shared_cat_inter']:
            parms['dim'] = dict(
                samples=str(self.examples['prefixes']['activities'].shape[0]),
                time_dim=str(self.examples['prefixes']['activities'].shape[1]),
                features=str(len(self.ac_index)))
        else:
            parms['dim'] = dict(
                samples=str(self.examples['encoder_input_data']['activities'].shape[0]),
                time_dim=str(self.examples['encoder_input_data']['activities'].shape[1]),
                features=str(len(self.ac_index)))
        parms['max_dur'] = self.examples['max_dur']

        sup.create_json(parms, os.path.join(self.output_folder,
                                            'parameters',
                                            'model_parameters.json'))
        sup.create_csv_file_header(self.log_test.to_dict('records'),
                                   os.path.join(self.output_folder,
                                                'parameters',
                                                'test_log.csv'))


class LogLoader():

    def __init__(self, path, read_options):
        """constructor"""
        self.path = path
        self.read_options = read_options

    def load(self, model_type):
        loader = self._get_loader(model_type)
        return loader()

    def _get_loader(self, model_type):
        if model_type in ['seq2seq_inter_full', 'shared_cat_inter_full']:
            return self._load_to_inter_full
        elif model_type in ['seq2seq_inter', 'shared_cat_inter', 'shared_cat']:
            return self._load_to_inter
        else:
            raise ValueError(model_type)

    def _load_to_inter_full(self):
        keep_cols = ['caseid', 'task', 'user', 'start_timestamp',
                     'end_timestamp', 'ac_index', 'event_id', 'rl_index',
                     'Unnamed: 0', 'dur', 'ev_duration', 'role', 'ev_rd']
        self.read_options['filter_d_attrib'] = False
        log = lr.LogReader(self.path, self.read_options)
        log_df = pd.DataFrame(log.data)
        log_df = log_df[~log_df.task.isin(['Start', 'End'])]
        # Scale loaded inter-case features
        colnames = list(log_df.columns.difference(keep_cols))
        for col in colnames:
            log_df = nsup.scale_feature(log_df, col, 'max', True)
        return log_df

    def _load_to_inter(self):
        self.read_options['filter_d_attrib'] = True
        log = lr.LogReader(self.path, self.read_options)
        log_df = pd.DataFrame(log.data)
        log_df = log_df[~log_df.task.isin(['Start', 'End'])]
        return log_df
