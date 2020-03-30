# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:13:15 2020

@author: Manuel Camargo
"""
import itertools
import numpy as np

from nltk.util import ngrams
import keras.utils as ku

from support_modules import nn_support as nsup


class SequencesCreator():

    def __init__(self, log, ac_index, rl_index):
        """constructor"""
        self.log = log
        self.ac_index = ac_index
        self.rl_index = rl_index

    def vectorize(self, model_type, params):
        loader = self._get_vectorizer(model_type)
        return loader(params)

    def _get_vectorizer(self, model_type):
        if model_type == 'shared_cat':
            return self._vectorize_shared_cat
        elif model_type == 'shared_cat_inter':
            return self._vectorize_shared_cat_inter
        elif model_type == 'seq2seq':
            return self._vectorize_seq2seq
        elif model_type == 'seq2seq_inter':
            return self._vectorize_seq2seq_inter
        elif model_type == 'shared_cat_inter_full':
            return self._vectorize_shared_cat_inter_full
        else:
            raise ValueError(model_type)

    def _vectorize_shared_cat(self, parms):
        """
        Example function with types documented in the docstring.
        parms:
            log_df (dataframe): event log data.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        equi = {'ac_index': 'activities',
                'rl_index': 'roles',
                'dur_norm': 'times'}
        self.log = nsup.scale_feature(self.log, 'dur', parms['norm_method'])
        columns = list(equi.keys())
        vec = {'prefixes': dict(),
               'next_evt': dict(),
               'max_dur': np.max(self.log.dur)}
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        # n-gram definition
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = list(ngrams(self.log[i][x], parms['n_size'],
                                     pad_left=True, left_pad_symbol=0))
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                vec['prefixes'][equi[x]] = vec['prefixes'][equi[x]] + serie if i > 0 else serie
                vec['next_evt'][equi[x]] = vec['next_evt'][equi[x]] + y_serie if i > 0 else y_serie

        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        vec['prefixes']['times'] = vec['prefixes']['times'].reshape(
                (vec['prefixes']['times'].shape[0],
                 vec['prefixes']['times'].shape[1], 1))
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(
            vec['next_evt']['activities'], num_classes=len(self.ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(
            vec['next_evt']['roles'], num_classes=len(self.rl_index))
        return vec

    def _vectorize_shared_cat_inter(self, parms):
        """Example function with types documented in the docstring.
        parms:
            log_df (dataframe): event log data.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
    #    log_df = log_df[log_df.caseid=='Case28']
        self.log = nsup.scale_feature(self.log, 'dur', parms['norm_method'])
        columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp',
                   'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
        columns = [x for x in list(self.log.columns) if x not in columns]
        vec = {'prefixes':dict(), 'next_evt':dict(), 'max_dur':np.max(self.log.dur)}
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        # n-gram definition
        equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
        x_inter_dict = dict()
        y_inter_dict = dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = list(ngrams(self.log[i][x], parms['n_size'],
                                     pad_left=True, left_pad_symbol=0))
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x in list(equi.keys()):
                    vec['prefixes'][equi[x]] = vec['prefixes'][equi[x]] + serie if i > 0 else serie
                    vec['next_evt'][equi[x]] = vec['next_evt'][equi[x]] + y_serie if i > 0 else y_serie
                else:
                    x_inter_dict[x] = x_inter_dict[x] + serie if i > 0 else serie
                    y_inter_dict[x] = y_inter_dict[x] + y_serie if i > 0 else y_serie
        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        vec['prefixes']['times'] = vec['prefixes']['times'].reshape(
                (vec['prefixes']['times'].shape[0],
                 vec['prefixes']['times'].shape[1], 1))
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(vec['next_evt']['activities'],
                                                        num_classes=len(self.ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(vec['next_evt']['roles'],
                                                        num_classes=len(self.rl_index))
        # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
        for key, value in x_inter_dict.items():
            x_inter_dict[key] = np.array(value)
            x_inter_dict[key] = x_inter_dict[key].reshape((x_inter_dict[key].shape[0],
                       x_inter_dict[key].shape[1], 1))
        vec['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
        # Reshape y intercase attributes (suffixes, number of attributes)
        for key, value in y_inter_dict.items():
            x_inter_dict[key] = np.array(value)
        vec['next_evt']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
        return vec


    def _vectorize_seq2seq(self, parms):
        """Example function with types documented in the docstring.
        parms:
            log_df (dataframe): event log data.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        columns = ['ac_index', 'rl_index', 'dur_norm']
        self.log = nsup.scale_feature(self.log, 'dur', parms['norm_method'])
        examples = {'encoder_input_data': dict(),
                    'decoder_target_data': dict(),
                    'max_dur': np.max(self.log.dur)}
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        max_length = np.max([len(x['ac_index']) for x in
                             self.reformat_events(['ac_index'],
                                                  parms['one_timestamp'])])
        # n-gram definition
        equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append([0]*(max_length - idx) + self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:] + [0]*(max_length - len(self.log[i][x][idx:])))
                examples['encoder_input_data'][equi[x]] = examples['encoder_input_data'][equi[x]] + serie if i > 0 else serie
                examples['decoder_target_data'][equi[x]] = examples['decoder_target_data'][equi[x]] + y_serie if i > 0 else y_serie
        for value in equi.values():
            examples['encoder_input_data'][value]= np.array(examples['encoder_input_data'][value])
            examples['decoder_target_data'][value]= np.array(examples['decoder_target_data'][value])
        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        examples['encoder_input_data']['times'] = examples['encoder_input_data']['times'].reshape(
                (examples['encoder_input_data']['times'].shape[0],
                 examples['encoder_input_data']['times'].shape[1], 1))
        examples['decoder_target_data']['times'] = examples['decoder_target_data']['times'].reshape(
                (examples['decoder_target_data']['times'].shape[0],
                 examples['decoder_target_data']['times'].shape[1], 1))

        # One hot encode decoder_target_data
        examples['decoder_target_data']['activities'] = ku.to_categorical(
            examples['decoder_target_data']['activities'],
            num_classes=len(self.ac_index))
        examples['decoder_target_data']['roles'] = ku.to_categorical(
            examples['decoder_target_data']['roles'],
            num_classes=len(self.rl_index))
        return examples


    def _vectorize_seq2seq_inter(self, parms):
        """
        Example function with types documented in the docstring.
        parms:
            log_df (dataframe): event log data.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
    #    log_df = log_df[log_df.caseid=='Case28']
        self.log = nsup.scale_feature(self.log, 'dur', parms['norm_method'])
        columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp',
                   'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
        columns = [x for x in list(self.log.columns) if x not in columns]
        examples = {'encoder_input_data':dict(), 'decoder_target_data':dict(), 'max_dur':np.max(self.log.dur)}
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        max_length = np.max([len(x['ac_index']) for x in
                             self.reformat_events(['ac_index'],
                                                  parms['one_timestamp'])])
        # n-gram definition
        equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
        x_inter_dict = dict()
        for i, _ in enumerate(self.log):
            for x in columns:
                serie, y_serie = list(), list()
                for idx in range(1, len(self.log[i][x])):
                    serie.append([0]*(max_length - idx) + self.log[i][x][:idx])
                    y_serie.append(self.log[i][x][idx:] + [0]*(max_length - len(self.log[i][x][idx:])))
                if x in list(equi.keys()):
                    examples['encoder_input_data'][equi[x]] = examples['encoder_input_data'][equi[x]] + serie if i > 0 else serie
                    examples['decoder_target_data'][equi[x]] = examples['decoder_target_data'][equi[x]] + y_serie if i > 0 else y_serie
                else:
                    x_inter_dict[x] = x_inter_dict[x] + serie if i > 0 else serie
        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            examples['encoder_input_data'][value]= np.array(examples['encoder_input_data'][value])
            examples['decoder_target_data'][value]= np.array(examples['decoder_target_data'][value])
        # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
        examples['encoder_input_data']['times'] = examples['encoder_input_data']['times'].reshape(
                (examples['encoder_input_data']['times'].shape[0],
                 examples['encoder_input_data']['times'].shape[1], 1))
        examples['decoder_target_data']['times'] = examples['decoder_target_data']['times'].reshape(
                (examples['decoder_target_data']['times'].shape[0],
                 examples['decoder_target_data']['times'].shape[1], 1))

        # One hot encode decoder_target_data
        examples['decoder_target_data']['activities'] = ku.to_categorical(examples['decoder_target_data']['activities'],
                                                        num_classes=len(self.ac_index))
        examples['decoder_target_data']['roles'] = ku.to_categorical(examples['decoder_target_data']['roles'],
                                                        num_classes=len(self.rl_index))
        # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
        for key, value in x_inter_dict.items():
            x_inter_dict[key] = np.array(value)
            x_inter_dict[key] = x_inter_dict[key].reshape((x_inter_dict[key].shape[0],
                       x_inter_dict[key].shape[1], 1))
        examples['encoder_input_data']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
    #    examples['decoder_target_data']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
        return examples

    def _vectorize_shared_cat_inter_full(self, parms):
        return 'tara!!'

    # =============================================================================
    # Reformat events
    # =============================================================================
    def reformat_events(self, columns, one_timestamp):
        """Creates series of activities, roles and relative times per trace.
        parms:
            self.log: dataframe.
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
