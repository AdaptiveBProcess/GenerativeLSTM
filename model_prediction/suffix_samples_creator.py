# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:03:26 2020

@author: Manuel Camargo
"""
import itertools

import pandas as pd
import numpy as np


class SuffixSamplesCreator():
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        self.log = pd.DataFrame
        self.ac_index = dict()
        self.rl_index = dict()
        self._samplers = dict()
        self._samp_dispatcher = {'basic': self._sample_suffix,
                                 'inter': self._sample_suffix_inter}

    def create_samples(self, params, log, ac_index, rl_index, add_cols):
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index
        columns = self.define_columns(add_cols)
        sampler = self._get_model_specific_sampler(params['model_type'])
        return sampler(columns, params)

    @staticmethod
    def define_columns(add_cols):
        columns = ['ac_index', 'rl_index', 'dur_norm']
        add_cols = [x+'_norm' for x in add_cols]
        columns.extend(add_cols)
        return columns

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

    def _sample_suffix(self, columns, parms):
        """
        Extraction of prefixes and expected suffixes from event log.
        Args:
            self.log (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        # columns = ['ac_index', 'rl_index', 'dur_norm']
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        spl = {'prefixes': dict(), 'suffixes': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append(self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:])
                spl['prefixes'][equi[x]] = (
                    spl['prefixes'][equi[x]] + serie if i > 0 else serie)
                spl['suffixes'][equi[x]] = (
                    spl['suffixes'][equi[x]] + y_serie if i > 0 else y_serie)
        return spl

    def _sample_suffix_inter(self, columns, parms):
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        spl = {'prefixes': dict(), 'suffixes': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        x_inter_dict, y_inter_dict = dict(), dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append(self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:])
                if x in list(equi.keys()):
                    spl['prefixes'][equi[x]] = (
                        spl['prefixes'][equi[x]] + serie if i > 0 else serie)
                    spl['suffixes'][equi[x]] = (
                        spl['suffixes'][equi[x]] + y_serie if i > 0 else y_serie)
                else:
                    x_inter_dict[x] = (
                        x_inter_dict[x] + serie if i > 0 else serie)
                    y_inter_dict[x] = (
                        y_inter_dict[x] + y_serie if i > 0 else y_serie)
        # Reshape intercase attributes (prefixes, n-gram size, # attributes)
        spl['prefixes']['inter_attr'] = list()
        x_inter_dict = pd.DataFrame(x_inter_dict)
        for row in x_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            spl['prefixes']['inter_attr'].append(new_row)
        # Reshape intercase expected attributes (prefixes, # attributes)
        spl['suffixes']['inter_attr'] = list()
        y_inter_dict = pd.DataFrame(y_inter_dict)
        for row in y_inter_dict.values:
            new_row = [np.array(x) for x in row]
            new_row = np.dstack(new_row)
            new_row = new_row.reshape((new_row.shape[1], new_row.shape[2]))
            spl['suffixes']['inter_attr'].append(new_row)
        return spl

    def _suffix_seq2seq_inter(self, parms):
        """Extraction of prefixes and expected suffixes from event log.
        Args:
            self.log (dataframe): testing dataframe in pandas format.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            pref_size (int): size of the prefixes to extract.
        Returns:
            list: list of prefixes and expected sufixes.
        """
        columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp',
                   'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
        columns = [x for x in list(self.log.columns) if x not in columns]
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        max_length = parms['dim']['time_dim']
        spl = {'prefixes': dict(), 'suffixes': dict()}
        # n-gram definition
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        x_inter_dict = dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append([0]*(max_length - idx) + self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:])
                if x in list(equi.keys()):
                    spl['prefixes'][equi[x]] = (
                        spl['prefixes'][equi[x]] + serie if i > 0 else serie)
                    spl['suffixes'][equi[x]] = (
                        spl['suffixes'][equi[x]] + y_serie if i > 0 else y_serie)
                else:
                    x_inter_dict[x] =(
                        x_inter_dict[x] + serie if i > 0 else serie)
        for value in equi.values():
            spl['prefixes'][value] = np.array(spl['prefixes'][value])
        # Reshape times
        spl['prefixes']['times'] = spl['prefixes']['times'].reshape(
                (spl['prefixes']['times'].shape[0],
                 spl['prefixes']['times'].shape[1], 1))
        # Reshape intercase attributes (prefixes, n-gram size, # attributes)
        for key, value in x_inter_dict.items():
            x_inter_dict[key] = np.array(value)
            x_inter_dict[key] = (
                x_inter_dict[key].reshape((x_inter_dict[key].shape[0],
                                           x_inter_dict[key]
                                           .shape[1], 1)))
        spl['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
        return spl

# =============================================================================
# Reformat
# =============================================================================
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
