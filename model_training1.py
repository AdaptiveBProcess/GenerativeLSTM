# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:07:19 2020

@author: Manuel Camargo
"""
import os

import pandas as pd
# import numpy as np
import itertools
from operator import itemgetter

from support_modules.readers import log_reader as lr
from support_modules import nn_support as nsup
from support_modules import role_discovery as rl
from support_modules.intercase_features import intercase_features as inf


class ModelTrainer():
    """
    This class evaluates the tasks durations and associates resources to it
    """

    def __init__(self, params):
        """constructor"""
        self.log = self.load_log(params)
        # Load indexes
        self.ac_index = dict()
        self.rl_index = dict()

        inp = InputPreprocessor(params)
        self.log = inp.preprocess(params, self.log)
        print(self.log)

        self.create_activities_index()
        self.create_roles_index()

        # ac_idx = lambda x: ac_index[x['task']]
        # log_df['ac_index'] = log_df.apply(ac_idx, axis=1)

        # rl_idx = lambda x: rl_index[x['role']]
        # log_df['rl_index'] = log_df.apply(rl_idx, axis=1)

    @staticmethod
    def load_log(params):
        loader = LogLoader(os.path.join('input_files', params['file_name']),
                           params['read_options'])
        return loader.load(params['model_type'])

    def create_activities_index(self):
        # Index creation
        self.ac_index = self.create_index(self.log, 'task')
        self.ac_index['start'] = 0
        self.ac_index['end'] = len(self.ac_index)
        index_ac = {v: k for k, v in self.ac_index.items()}

    def create_roles_index(self):
        self.rl_index = self.create_index(self.log, 'role')
        self.rl_index['start'] = 0
        self.rl_index['end'] = len(self.rl_index)
        index_rl = {v: k for k, v in self.rl_index.items()}

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


class InputPreprocessor():

    def __init__(self, params):
        """constructor"""
        self.rp_sim = params['rp_sim']
        self.model_type = params['model_type']
        self.one_timestamp = params['one_timestamp']
        self.resources = pd.DataFrame

    def preprocess(self, params, log):
        log = self.add_resources(log)
        log = self.add_intercases(log, params)
        log = self.add_calculated_times(log)
        return log

    def add_resources(self, log):
        # Resource pool discovery
        res_analyzer = rl.ResourcePoolAnalyser(log, sim_threshold=self.rp_sim)
        # Role discovery
        self.resources = pd.DataFrame.from_records(res_analyzer.resource_table)
        self.resources = self.resources.rename(index=str,
                                               columns={"resource": "user"})
        # Add roles information
        log = log.merge(self.resources, on='user', how='left')
        log = log[~log.task.isin(['Start', 'End'])]
        log = log.reset_index(drop=True)
        return log

    def add_intercases(self, log, params):
        # Add intercase features
        if self.model_type in ['seq2seq_inter', 'shared_cat_inter',
                               'shared_cat_inter_full']:
            # TODO: Refactor this to delete params as input
            log = inf.calculate_intercase_features(params, log, self.resources)
        return log

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log_df: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log = log.to_dict('records')
        if self.one_timestamp:
            log = sorted(log, key=lambda x: x['caseid'])
            for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
                events = list(group)
                events = sorted(events, key=itemgetter('end_timestamp'))
                for i in range(0, len(events)):
                    # In one-timestamp approach the first activity of the trace
                    # is taken as instantsince there is no previous timestamp
                    # to find a range
                    if i == 0:
                        events[i]['dur'] = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                        events[i]['dur'] = dur
        else:
            log = sorted(log, key=itemgetter('start_timestamp'))
            for event in log:
                # on the contrary is btw start and complete timestamp
                event['dur'] = (event['end_timestamp'] -
                                event['start_timestamp']).total_seconds()
        return pd.DataFrame.from_dict(log)


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
