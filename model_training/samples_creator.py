# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:13:15 2020

@author: Manuel Camargo
"""
import itertools
import numpy as np
import random

from nltk.util import ngrams
import keras.utils as ku


class SequencesCreator():

    def __init__(self, log, one_timestamp, ac_index, rl_index):
        """constructor"""
        self.log = log
        self.one_timestamp = one_timestamp
        self.ac_index = ac_index
        self.rl_index = rl_index
        self._vectorizers = dict()
        self._vec_dispatcher = {'basic': self._vectorize_seq,
                                'inter': self._vectorize_seq_inter,
                                'gan': self.gan_simple}

    def vectorize(self, model_type, params, add_cols):
        columns = self.define_columns(add_cols, self.one_timestamp)
        loader = self._get_vectorizer(model_type)
        return loader(params, columns)

    def register_vectorizer(self, model_type, vectorizer):
        try:
            self._vectorizers[model_type] = self._vec_dispatcher[vectorizer]
        except KeyError:
            raise ValueError(vectorizer)

    def _get_vectorizer(self, model_type):
        vectorizer = self._vectorizers.get(model_type)
        if not vectorizer:
            raise ValueError(model_type)
        return vectorizer

    @staticmethod
    def define_columns(add_cols, one_timestamp):
        columns = ['ac_index', 'rl_index', 'dur_norm']
        add_cols = [x+'_norm' for x in add_cols]
        columns.extend(add_cols)
        if not one_timestamp:
            columns.extend(['wait_norm'])
        return columns

    def _vectorize_seq(self, parms, columns):
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        # TODO: reorganizar este metoo para poder vectorizar los tiempos
        # con uno o dos features de tiempo, posiblemente la idea es
        # hacer equi como si fueran intercases.
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
                serie = list(ngrams(self.log[i][x], parms['n_size'],
                                    pad_left=True, left_pad_symbol=0))
                y_serie = [x[-1] for x in serie]
                serie = serie[:-1]
                y_serie = y_serie[1:]
                if x in list(equi.keys()):
                    vec['prefixes'][equi[x]] = (vec['prefixes'][equi[x]] + serie
                                                if i > 0 else serie)
                    vec['next_evt'][equi[x]] = (vec['next_evt'][equi[x]] + y_serie
                                                if i > 0 else y_serie)
                elif x in times:
                    x_times_dict[x] = (
                        x_times_dict[x] + serie if i > 0 else serie)
                    y_times_dict[x] = (
                        y_times_dict[x] + y_serie if i > 0 else y_serie)

        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(
            vec['next_evt']['activities'], num_classes=len(self.ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(
            vec['next_evt']['roles'], num_classes=len(self.rl_index))
        # reshape times
        for key, value in x_times_dict.items():
            x_times_dict[key] = np.array(value)
            x_times_dict[key] = x_times_dict[key].reshape(
                (x_times_dict[key].shape[0], x_times_dict[key].shape[1], 1))
        vec['prefixes']['times'] = np.dstack(list(x_times_dict.values()))
        # Reshape y times attributes (suffixes, number of attributes)
        vec['next_evt']['times'] = np.dstack(list(y_times_dict.values()))[0]
        return vec

    def _vectorize_seq_inter(self, parms, columns):
        """
        Dataframe vectorizer to process intercase or data atributes features.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        times = ['dur_norm'] if parms['one_timestamp'] else ['dur_norm', 'wait_norm']
        equi = {'ac_index': 'activities', 'rl_index': 'roles'}
        vec = {'prefixes': dict(),
               'next_evt': dict()}
        x_times_dict = dict()
        y_times_dict = dict()
        # intercases
        x_inter_dict = dict()
        y_inter_dict = dict()
        self.log = self.reformat_events(columns, parms['one_timestamp'])
        for i, _ in enumerate(self.log):
            for x in columns:
                serie = list(ngrams(self.log[i][x], parms['n_size'],
                                    pad_left=True, left_pad_symbol=0))
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
                else:
                    x_inter_dict[x] = (
                        x_inter_dict[x] + serie if i > 0 else serie)
                    y_inter_dict[x] = (
                        y_inter_dict[x] + y_serie if i > 0 else y_serie)
        # Transform task, dur and role prefixes in vectors
        for value in equi.values():
            vec['prefixes'][value] = np.array(vec['prefixes'][value])
            vec['next_evt'][value] = np.array(vec['next_evt'][value])
        # one-hot encode target values
        vec['next_evt']['activities'] = ku.to_categorical(
            vec['next_evt']['activities'], num_classes=len(self.ac_index))
        vec['next_evt']['roles'] = ku.to_categorical(
            vec['next_evt']['roles'], num_classes=len(self.rl_index))
        # reshape times
        for key, value in x_times_dict.items():
            x_times_dict[key] = np.array(value)
            x_times_dict[key] = x_times_dict[key].reshape(
                (x_times_dict[key].shape[0], x_times_dict[key].shape[1], 1))
        vec['prefixes']['times'] = np.dstack(list(x_times_dict.values()))
        # Reshape y times attributes (suffixes, number of attributes)
        vec['next_evt']['times'] = np.dstack(list(y_times_dict.values()))[0]
        # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
        for key, value in x_inter_dict.items():
            x_inter_dict[key] = np.array(value)
            x_inter_dict[key] = x_inter_dict[key].reshape(
                (x_inter_dict[key].shape[0], x_inter_dict[key].shape[1], 1))
        vec['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
        # Reshape y intercase attributes (suffixes, number of attributes)
        vec['next_evt']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
        return vec

    # def _vectorize_seq_inter(self, parms, columns):
    #     vec = {'prefixes': dict(),
    #            'next_evt': dict()}
    #     self.log = self.reformat_events(columns, parms['one_timestamp'])
    #     # n-gram definition
    #     equi = {'ac_index': 'activities',
    #             'rl_index': 'roles',
    #             'dur_norm': 'times'}
    #     x_inter_dict = dict()
    #     y_inter_dict = dict()
    #     for i, _ in enumerate(self.log):
    #         for x in columns:
    #             serie = list(ngrams(self.log[i][x], parms['n_size'],
    #                                 pad_left=True, left_pad_symbol=0))
    #             y_serie = [x[-1] for x in serie]
    #             serie = serie[:-1]
    #             y_serie = y_serie[1:]
    #             if x in list(equi.keys()):
    #                 vec['prefixes'][equi[x]] = (
    #                     vec['prefixes'][equi[x]] + serie if i > 0 else serie)
    #                 vec['next_evt'][equi[x]] = (
    #                     vec['next_evt'][equi[x]] + y_serie
    #                     if i > 0 else y_serie)
    #             else:
    #                 x_inter_dict[x] = (
    #                     x_inter_dict[x] + serie if i > 0 else serie)
    #                 y_inter_dict[x] = (
    #                     y_inter_dict[x] + y_serie if i > 0 else y_serie)
    #     # Transform task, dur and role prefixes in vectors
    #     for value in equi.values():
    #         vec['prefixes'][value] = np.array(vec['prefixes'][value])
    #         vec['next_evt'][value] = np.array(vec['next_evt'][value])
    #     # Reshape dur (prefixes, n-gram size, 1) i.e. time distribute
    #     vec['prefixes']['times'] = vec['prefixes']['times'].reshape(
    #             (vec['prefixes']['times'].shape[0],
    #               vec['prefixes']['times'].shape[1], 1))
    #     # one-hot encode target values
    #     vec['next_evt']['activities'] = ku.to_categorical(
    #         vec['next_evt']['activities'], num_classes=len(self.ac_index))
    #     vec['next_evt']['roles'] = ku.to_categorical(
    #         vec['next_evt']['roles'], num_classes=len(self.rl_index))
    #     # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
    #     for key, value in x_inter_dict.items():
    #         x_inter_dict[key] = np.array(value)
    #         x_inter_dict[key] = x_inter_dict[key].reshape(
    #             (x_inter_dict[key].shape[0], x_inter_dict[key].shape[1], 1))
    #     vec['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
    #     # Reshape y intercase attributes (suffixes, number of attributes)
    #     for key, value in y_inter_dict.items():
    #         x_inter_dict[key] = np.array(value)
    #     vec['next_evt']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
    #     return vec

    def gan_simple(self, parms, columns):
        print(columns)
        vec = {'training':dict()}
        pairs = self.log.copy()
        pairs = pairs[['ac_index', 'rl_index']]
        pairs = pairs.to_records(index=False).tolist()
        # Vectorize discriminator training real inputs
        vec['training']['activities'] = [x[0] for x in pairs]
        vec['training']['activities'] = ku.to_categorical(
            vec['training']['activities'], num_classes=len(self.ac_index))
        vec['training']['roles'] = [x[1] for x in pairs]
        vec['training']['roles'] = ku.to_categorical(
            vec['training']['roles'], num_classes=len(self.rl_index))
        vec['training']['class'] = np.zeros(len(pairs))
        
        # If the discriminator will be pretrained create pretraining examples
        if parms['gan_pretrain']:
            # one third of real events randomly selected
            n_positive = int(round(len(pairs)/3))
            negative_ratio = 2
    
            batch_size = n_positive * (1 + negative_ratio)
            batch = np.zeros((batch_size, 3))
            pairs_set = set(pairs)
            activities = list(self.ac_index.keys())
            roles = list(self.rl_index.keys())
            # randomly choose positive examples
            idx = 0
            for idx, (activity, role) in enumerate(
                    random.sample(pairs, n_positive)):
                batch[idx, :] = (activity, role, 0)
            # Increment idx by 1
            idx += 1
            # Add negative examples until reach batch size
            while idx < batch_size:
                # random selection
                random_ac = random.randrange(len(activities))
                random_rl = random.randrange(len(roles))
                # Check to make sure this is not a positive example
                if (random_ac, random_rl) not in pairs_set:
                    # Add to batch and increment index,  0 due classification task
                    batch[idx, :] = (random_ac, random_rl, 1)
                    idx += 1
            vec['pretraining'] = dict()
            # Make sure to shuffle order
            np.random.shuffle(batch)
            vec['pretraining']['activities'] = ku.to_categorical(
                batch[:, 0], num_classes=len(self.ac_index))
            vec['pretraining']['roles'] = ku.to_categorical(
                batch[:, 1], num_classes=len(self.rl_index))
            vec['pretraining']['class'] = batch[:, 2]
        return vec

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