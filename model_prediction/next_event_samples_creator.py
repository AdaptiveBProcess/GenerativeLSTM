# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:28:18 2020

@author: Manuel Camargo
"""
import itertools

import pandas as pd
import numpy as np
import keras.utils as ku


class NextEventSamplesCreator():
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        self.log = pd.DataFrame
        self.ac_index = dict()
        self.rl_index = dict()
        self._samplers = dict()
        self._samp_dispatcher = {'basic': self._sample_next_event,
                                 'inter': self._sample_next_event_inter}

    def create_samples(self, params, log, ac_index, rl_index, add_cols):
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index
        columns = self.define_columns(add_cols, params['one_timestamp'])
        sampler = self._get_model_specific_sampler(params['model_type'])
        return sampler(columns, params)

    @staticmethod
    def define_columns(add_cols, one_timestamp):
        columns = ['ac_index', 'rl_index', 'dur_norm']
        add_cols = [x+'_norm' if x != 'weekday' else x for x in add_cols ]
        columns.extend(add_cols)
        if not one_timestamp:
            columns.extend(['wait_norm'])
        return columns
    # def define_columns(add_cols, one_timestamp):
    #     columns = ['ac_index', 'rl_index', 'dur_norm']
    #     add_cols = [x+'_norm' for x in add_cols]
    #     columns.extend(add_cols)
    #     if not one_timestamp:
    #         columns.extend(['wait_norm'])
    #     return columns

    def register_sampler(self, model_type, sampler):
        try:
            self._samplers[model_type] = self._samp_dispatcher[sampler]
        except KeyError:
            raise ValueError(sampler)

    def _get_model_specific_sampler(self, model_type):
        sampler = self._samplers.get(model_type)
        if not sampler:
            raise ValueError(model_type)
        return sampler

    def _sample_next_event(self, columns, parms):
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
        times = ['dur_norm'] if parms['one_timestamp'] else ['dur_norm', 'wait_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}
        vec = {'prefixes': dict(),
               'next_evt': dict()}
        x_times_dict = dict()
        y_times_dict = dict()
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        # n-gram definition
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = [self.log[i][x][:idx]
                         for idx in range(1, len(self.log[i][x]))]
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x in list(equi.keys()):
                    vec['prefixes'][equi[x]] = (
                        vec['prefixes'][equi[x]] + serie
                        if i > 0 else serie)
                    vec['next_evt'][equi[x]] = (
                        vec['next_evt'][equi[x]] + y_serie
                        if i > 0 else y_serie)
                elif x in times:
                    x_times_dict[x] = (
                        x_times_dict[x] + serie if i > 0 else serie)
                    y_times_dict[x] = (
                        y_times_dict[x] + y_serie if i > 0 else y_serie)
        vec['prefixes']['times'] = list()
        x_times_dict = pd.DataFrame(x_times_dict)
        for row in x_times_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            vec['prefixes']['times'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        vec['next_evt']['times'] = list()
        y_times_dict = pd.DataFrame(y_times_dict)
        for row in y_times_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[2]))
            vec['next_evt']['times'].append(new_row)
        return vec

    def _sample_next_event_inter_old(self, columns, parms):
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
                        examples['prefixes'][equi[x]] + serie
                        if i > 0 else serie)
                    examples['next_evt'][equi[x]] = (
                        examples['next_evt'][equi[x]] + y_serie
                        if i > 0 else y_serie)
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

    def _sample_next_event_inter(self, columns, parms):
        """
        Dataframe vectorizer to process intercase or data atributes features.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        print(self.log.dtypes)
        times = ['dur_norm'] if parms['one_timestamp'] else ['dur_norm', 'wait_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}
        vec = {'prefixes': dict(),
               'next_evt': dict()}
        x_weekday = list()
        y_weekday = list()
        # times
        x_times_dict = dict()
        y_times_dict = dict()
        # intercases
        x_inter_dict = dict()
        y_inter_dict = dict()
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = [self.log[i][x][:idx]
                          for idx in range(1, len(self.log[i][x]))]
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x in list(equi.keys()):
                    vec['prefixes'][equi[x]] = (
                        vec['prefixes'][equi[x]] + serie if i > 0 else serie)
                    vec['next_evt'][equi[x]] = (
                        vec['next_evt'][equi[x]] + y_serie
                        if i > 0 else y_serie)
                elif x in times:
                    x_times_dict[x] = (
                        x_times_dict[x] + serie if i > 0 else serie)
                    y_times_dict[x] = (
                        y_times_dict[x] + y_serie if i > 0 else y_serie)
                elif x == 'weekday':
                    x_weekday = (
                        x_weekday + serie if i > 0 else serie)
                    y_weekday = (
                        y_weekday + y_serie if i > 0 else y_serie)
                else:
                    x_inter_dict[x] = (
                        x_inter_dict[x] + serie if i > 0 else serie)
                    y_inter_dict[x] = (
                        y_inter_dict[x] + y_serie if i > 0 else y_serie)
        # # Transform task, dur and role prefixes in vectors
        # for value in equi.values():
        #     vec['prefixes'][value] = np.array(vec['prefixes'][value])
        #     vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # # one-hot encode target values
        # vec['next_evt']['activities'] = ku.to_categorical(
        #     vec['next_evt']['activities'], num_classes=len(self.ac_index))
        # vec['next_evt']['roles'] = ku.to_categorical(
        #     vec['next_evt']['roles'], num_classes=len(self.rl_index))
        # # reshape times
        # for key, value in x_times_dict.items():
        #     x_times_dict[key] = np.array(value)
        #     x_times_dict[key] = x_times_dict[key].reshape(
        #         (x_times_dict[key].shape[0], x_times_dict[key].shape[1], 1))
        # vec['prefixes']['times'] = np.dstack(list(x_times_dict.values()))
        # # Reshape y times attributes (suffixes, number of attributes)
        # vec['next_evt']['times'] = np.dstack(list(y_times_dict.values()))[0]
        # # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
        # for key, value in x_inter_dict.items():
        #     x_inter_dict[key] = np.array(value)
        #     x_inter_dict[key] = x_inter_dict[key].reshape(
        #         (x_inter_dict[key].shape[0], x_inter_dict[key].shape[1], 1))
        # vec['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
        # # Reshape y intercase attributes (suffixes, number of attributes)
        # vec['next_evt']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
        # if 'weekday' in columns:
        #     # Onehot encode weekday
        #     x_weekday = ku.to_categorical(x_weekday, num_classes=7)
        #     y_weekday = ku.to_categorical(y_weekday, num_classes=7)
        #     vec['prefixes']['inter_attr'] = np.concatenate(
        #         [vec['prefixes']['inter_attr'], x_weekday], axis=2)
        #     vec['next_evt']['inter_attr'] = np.concatenate(
        #         [vec['next_evt']['inter_attr'], y_weekday], axis=1)
        # Reshape intercase attributes (prefixes, n-gram size, # attributes)
        vec['prefixes']['inter_attr'] = list()
        x_inter_dict = pd.DataFrame(x_inter_dict)
        for row, wd in zip(x_inter_dict.values, x_weekday):
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            x_weekday = ku.to_categorical(x_weekday, num_classes=7)
            y_weekday = ku.to_categorical(y_weekday, num_classes=7)
            vec['prefixes']['inter_attr'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        vec['next_evt']['inter_attr'] = list()
        y_inter_dict = pd.DataFrame(y_inter_dict)
        for row in y_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[2]))
            vec['next_evt']['inter_attr'].append(new_row)
        print(vec['prefixes']['inter_attr'])
        return vec

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
