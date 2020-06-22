# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:18:18 2020

@author: Manuel Camargo
"""
import pandas as pd
import numpy as np
from datetime import datetime

import itertools
from operator import itemgetter

from support_modules import role_discovery as rl
from model_training.intercase_features import intercase_features as inf
from model_training.intercase_features import resource_dedication as rd


class FeaturesMannager():

    def __init__(self, params):
        """constructor"""
        self.rp_sim = params['rp_sim']
        self.model_type = params['model_type']
        self.one_timestamp = params['one_timestamp']
        self.resources = pd.DataFrame
        self.norm_method = params['norm_method']

    def calculate(self, log):
        log = self.add_resources(log)
        log = self.filter_intercases(log)
        log = self.add_calculated_times(log)
        return self.scale_features(log)

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

    def filter_intercases(self, log):
        # Add intercase features
        columns = ['caseid', 'task', 'user', 'end_timestamp', 'role']
        if self.model_type in ['seq2seq', 'shared_cat',
                               'cnn_lstm', 'shared_cat_cx']:
            log = log[columns]
        elif self.model_type in ['seq2seq_inter', 'shared_cat_inter',
                               'cnn_lstm_inter', 'shared_cat_inter_full',
                               'cnn_lstm_inter_full']:
            columns.extend(['ev_et', 'ev_et_t', 'ev_rd', 'ev_rp_occ'])
            log = log[columns]
        elif self.model_type in ['shared_cat_rd']:
            columns.extend(['ev_rd', 'ev_rp_occ'])
            log = log[columns]
        elif self.model_type in ['shared_cat_wl']:
            columns.extend(['ev_et', 'ev_et_t'])
            log = log[columns]
        elif self.model_type in ['shared_cat_city']:
            columns.extend(['city1','city2','city3'])
            log = log[columns]
        elif self.model_type in ['shared_cat_snap']:
            columns.extend(['snap1','snap2','snap3'])
            log = log[columns]
        else:
            raise ValueError(model_type)
        return log

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log['acc_cycle'] = 0
        log['daytime'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                        acc = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                        acc = (events[i]['end_timestamp'] -
                               events[0]['end_timestamp']).total_seconds()
                else:
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    acc = (events[i]['end_timestamp'] -
                           events[0]['start_timestamp']).total_seconds()
                events[i]['dur'] = dur
                events[i]['acc_cycle'] = acc
                time = events[i][ordk].time()
                time = time.second + time.minute*60 + time.hour*3600
                events[i]['daytime'] = time
        return pd.DataFrame.from_dict(log)

    def scale_features(self, log):
        scaler = self._get_scaler(self.model_type)
        return scaler(log)

    def _get_scaler(self, model_type):
        if model_type == 'shared_cat':
            return self._scale_base
        elif model_type == 'cnn_lstm':
            return self._scale_base
        elif model_type == 'seq2seq':
            return self._scale_base
        elif model_type == 'shared_cat_inter':
            return self._scale_inter
        elif model_type == 'shared_cat_inter_full':
            return self._scale_inter
        elif model_type == 'cnn_lstm_inter_full':
            return self._scale_inter
        elif model_type == 'cnn_lstm_inter':
            return self._scale_inter
        elif model_type == 'shared_cat_rd':
            return self._scale_rd
        elif model_type == 'shared_cat_wl':
            return self._scale_wl
        elif model_type == 'shared_cat_cx':
            return self._scale_cx
        elif model_type == 'shared_cat_city':
            return self._scale_city
        elif model_type == 'shared_cat_snap':
            return self._scale_snap
        else:
            raise ValueError(model_type)

    def _scale_base(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        return log, scale_args

    def _scale_inter(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        log, _ = self.scale_feature(log, 'daytime', 'day_secs', True)
        for col in ['ev_et', 'ev_et_t', 'ev_rd', 'ev_rp_occ', 'acc_cycle']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    def _scale_inter_full(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        for col in ['ev_et', 'ev_et_t', 'ev_rd', 'ev_rp_occ']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)

    def _scale_rd(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        for col in ['ev_rd', 'ev_rp_occ']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    def _scale_wl(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        for col in ['ev_et', 'ev_et_t']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    def _scale_cx(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        log, _ = self.scale_feature(log, 'acc_cycle', self.norm_method, True)
        log, _ = self.scale_feature(log, 'daytime', 'day_secs', True)
        return log, scale_args

    def _scale_city(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        for col in ['city1','city2','city3']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    def _scale_snap(self, log):
        log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        for col in ['snap1','snap2','snap3']:
            log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    # =========================================================================
    # Scale features
    # =========================================================================
    @staticmethod
    def scale_feature(log, feature, method, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        scale_args = dict()
        if method == 'lognorm':
            log[feature + '_log'] = np.log1p(log[feature])
            max_value = np.max(log[feature+'_log'])
            min_value = np.min(log[feature+'_log'])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature+'_log'], min_value), (max_value - min_value))
            log = log.drop((feature + '_log'), axis=1)
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'normal':
            max_value = np.max(log[feature])
            min_value = np.min(log[feature])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature], min_value), (max_value - min_value))
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'standard':
            mean = np.mean(log[feature])
            std = np.std(log[feature])
            log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                               std)
            scale_args = {'mean': mean, 'std': std}
        elif method == 'max':
            max_value = np.max(log[feature])
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method == 'day_secs':
            max_value = 86400
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method is None:
            log[feature+'_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log, scale_args
