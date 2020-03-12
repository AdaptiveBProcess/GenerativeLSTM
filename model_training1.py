# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:07:19 2020

@author: Manuel Camargo
"""
import os


import pandas as pd
# import numpy as np

from support_modules.readers import log_reader as lr
from support_modules import nn_support as nsup


class ModelTrainer():
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, params):
        """constructor"""
        self.params = params
        inp = InputPreprocessor(params)
        print(inp.log)


class InputPreprocessor():

    def __init__(self, params):
        """constructor"""
        self.log = self.load_log(params)

    @staticmethod
    def load_log(params):
        loader = LogLoader(os.path.join('input_files', params['file_name']),
                           params['read_options'])
        loader.load(params['model_type'])

    # def proprocess_log():
    #     self.add_roles_information(params)
    #     self.calculate_intercase_features()
    #     self.create_activities_index()
    #     self.create_roles_index()

    # def add_roles_information(params):
    #     # Resource pool discovery
    #     res_analyzer = rl.ResourcePoolAnalyser(log_df, sim_threshold=parms['rp_similarity'])
    #     # Role discovery
    #     log_df_resources = pd.DataFrame.from_records(res_analyzer.resource_table)
    #     log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    #     # Dataframe creation
    #     log_df = log_df.merge(log_df_resources, on='user', how='left')
    #     log_df = log_df[log_df.task != 'Start']
    #     log_df = log_df[log_df.task != 'End']
    #     log_df = log_df.reset_index(drop=True)

    # def calculate_intercase_features():
    #     # Calculate general inter-case features
    #     if parms['model_type'] in ['seq2seq_inter', 'shared_cat_inter', 'shared_cat_inter_full']:
    #         log_df = inf.calculate_intercase_features(parms, log_df, log_df_resources)

    # def create_activities_index():
    #     # Index creation
    #     ac_index = create_index(log_df, 'task')
    #     ac_index['start'] = 0
    #     ac_index['end'] = len(ac_index)
    #     index_ac = {v: k for k, v in ac_index.items()}

    # def create_roles_index():
    #     rl_index = create_index(log_df, 'role')
    #     rl_index['start'] = 0
    #     rl_index['end'] = len(rl_index)
    #     index_rl = {v: k for k, v in rl_index.items()}


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
        elif model_type in ['seq2seq_inter', 'shared_cat_inter']:
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
        # Scale loaded inter-case features
        colnames = list(log_df.columns.difference(keep_cols))
        for col in colnames:
            log_df = nsup.scale_feature(log_df, col, 'max', True)
        return log_df

    def _load_to_inter(self):
        self.read_options['filter_d_attrib'] = True
        log = lr.LogReader(self.path, self.read_options)
        log_df = pd.DataFrame(log.data)
        return log_df
