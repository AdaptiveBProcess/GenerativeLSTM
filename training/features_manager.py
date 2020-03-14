# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:18:18 2020

@author: Manuel Camargo
"""
import pandas as pd

import itertools
from operator import itemgetter

from support_modules import role_discovery as rl
from support_modules.intercase_features import intercase_features as inf


class FeaturesMannager():

    def __init__(self, params):
        """constructor"""
        self.rp_sim = params['rp_sim']
        self.model_type = params['model_type']
        self.one_timestamp = params['one_timestamp']
        self.resources = pd.DataFrame

    def calculate(self, params, log):
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
            log: dataframe.
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
