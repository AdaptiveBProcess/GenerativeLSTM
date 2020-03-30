# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:28:18 2020

@author: Manuel Camargo
"""
import itertools

import pandas as pd
import numpy as np


class NextEventSamplesCreator():
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        self.log = pd.DataFrame
        self.ac_index = dict()
        self.rl_index = dict()

    def create_samples(self, params, log, ac_index, rl_index):
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index
        sampler = self._get_model_specific_sampler(params['model_type'])
        return sampler(params)

    def _get_model_specific_sampler(self, model_type):
        if model_type == 'shared_cat':
            return self._sample_next_event_shared_cat
        elif model_type == 'shared_cat_inter':
            return self._sample_next_event_shared_cat_inter
        else:
            raise ValueError(model_type)

    def _sample_next_event_shared_cat(self, parms):
        """
        Extraction of prefixes and expected suffixes from event log.
        Args:
            df_test (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        columns = ['ac_index', 'rl_index', 'dur_norm']
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        examples = {'prefixes': dict(), 'next_evt': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = [self.log[i][x][:idx]
                         for idx in range(1, len(self.log[i][x]))]
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                examples['prefixes'][equi[x]] = (
                    examples['prefixes'][equi[x]] + serie
                    if i > 0 else serie)
                examples['next_evt'][equi[x]] = (
                    examples['next_evt'][equi[x]] + y_serie
                    if i > 0 else y_serie)
        return examples

    def _sample_next_event_shared_cat_inter(self, parms):
        """Extraction of prefixes and expected suffixes from event log.
        Args:
            df_test (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        columns = ['caseid', 'task', 'user', 'start_timestamp',
                   'end_timestamp', 'dur_log', 'role', 'event_id',
                   'ev_duration', 'dur', 'ev_rd']
        columns = [x for x in list(self.log.columns) if x not in columns]
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        examples = {'prefixes': dict(), 'next_evt': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        x_inter_dict, y_inter_dict = dict(), dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = [self.log[i][x][:idx]
                         for idx in range(1, len(self.log[i][x]))]
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x in list(equi.keys()):
                    examples['prefixes'][equi[x]] = (
                        examples['prefixes'][equi[x]] + serie if i > 0 else serie)
                    examples['next_evt'][equi[x]] = (
                        examples['next_evt'][equi[x]] + y_serie if i > 0 else y_serie)
                else:
                    x_inter_dict[x] = (x_inter_dict[x] + serie
                                       if i > 0 else serie)
                    y_inter_dict[x] = (y_inter_dict[x] + y_serie
                                       if i > 0 else y_serie)
        # Reshape intercase attributes (prefixes, n-gram size, # attributes)
        examples['prefixes']['inter_attr'] = list()
        x_inter_dict = pd.DataFrame(x_inter_dict)
        for row in x_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            examples['prefixes']['inter_attr'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        examples['next_evt']['inter_attr'] = list()
        y_inter_dict = pd.DataFrame(y_inter_dict)
        for row in y_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[2]))
            examples['next_evt']['inter_attr'].append(new_row)
        return examples

    def reformat_events(self, columns, one_timestamp):
        """Creates series of activities, roles and relative times per trace.
        Args:
            log_df: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        temp_data = list()
        log_df = self.log.to_dict('records')
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'ac_index':
                    serie.insert(0, self.ac_index[('start')])
                    serie.append(self.ac_index[('end')])
                elif x == 'rl_index':
                    serie.insert(0, self.rl_index[('start')])
                    serie.append(self.rl_index[('end')])
                else:
                    serie.insert(0, 0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data
