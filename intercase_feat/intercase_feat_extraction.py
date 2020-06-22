# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:31:11 2020

@author: Manuel Camargo
"""
# Imports
import os
import pandas as pd
import itertools
import numpy as np

# import multiprocessing as mp

from operator import itemgetter

from support_modules.readers import log_reader as lr
from support_modules import role_discovery as rl
from support_modules import support as sup


class IntercasFeatureExtractor():
    """
    This is the man class encharged of the Feature extraction
    """

    def __init__(self, params, log, user_role):
        """constructor"""
        self.log = log
        self.settings = params
        self.sub_group = params['sub_group']
        self.expanded_log = pd.DataFrame
        self.event_slides = pd.DataFrame
        self.user_role = user_role

        # create input folder
        self.process_temp_folder(params['temp_input'])
        # create output folder
        self.process_temp_folder(params['temp_output'])
        # Split and process eventlog
        self.data_source_preparation()
        # Process folds
        self.process_folds()

    def data_source_preparation(self):
        self.log['event_id'] = self.log.index
        self.log = self.calculate_event_duration(self.log)
        # Split event log
        self.folding_creation(self.log,
                              self.settings['splits'],
                              self.settings['temp_input'])

    @staticmethod
    def folding_creation(log, splits, output):
        log = log.sort_values(by='end_timestamp')
        idxs = [x for x in range(0, len(log),
                                 round(len(log)/splits))]
        idxs.append(len(log))
        log['lead_trail'] = False
        folds = [pd.DataFrame(log.iloc[idxs[i-1]:idxs[i]])
                 for i in range(1, len(idxs))]
        for i in range(1, len(folds)):
            fold = folds[i]
            # Find initial incomplete traces
            inc_traces = pd.DataFrame(fold.groupby('caseid')
                                      .first()
                                      .reset_index())
            inc_traces = inc_traces[inc_traces.pos_trace > 0]
            inc_traces = inc_traces['caseid'].to_list()
            # find completion of incomplete traces
            prev_fold = folds[i - 1]
            times = prev_fold[(prev_fold.caseid.isin(inc_traces)) &
                              (prev_fold.lead_trail == False)]
            del inc_traces
            # Define timespan for leading events
            minimum = times.end_timestamp.min()
            leading = prev_fold[(prev_fold.end_timestamp >= minimum) &
                                (prev_fold.lead_trail == False)]
            leading = leading.caseid.to_list()
            leading = pd.DataFrame(prev_fold[prev_fold.caseid.isin(leading)])
            minimum = leading.groupby('caseid').tail(2).reset_index()
            minimum = minimum.end_timestamp.min()
            leading = leading[leading.end_timestamp >= minimum]
            leading['lead_trail'] = True
            # Attach leading events
            folds[i] = pd.concat([leading, fold], axis=0, ignore_index=True)
            del leading
            del fold

        for i in range(0, len(folds)-1):
            fold = folds[i]
            # Find initial incomplete traces
            inc_traces = pd.DataFrame(fold.groupby('caseid')
                                      .last()
                                      .reset_index())
            inc_traces = inc_traces[inc_traces.pos_trace < inc_traces.trace_len]
            inc_traces = inc_traces['caseid'].to_list()
            # find completion of incomplete traces
            next_fold = folds[i + 1]
            times = next_fold[(next_fold.caseid.isin(inc_traces)) &
                              (next_fold.lead_trail == False)]
            del inc_traces
            # Define timespan for leading events
            maximum = times.end_timestamp.max()
            trailing = pd.DataFrame(
                next_fold[(next_fold.end_timestamp <= maximum) &
                          (next_fold.lead_trail == False)])
            trailing['lead_trail'] = True
            # Attach leading events
            folds[i] = pd.concat([fold, trailing], axis=0, ignore_index=True)
            del trailing
            del fold
        # Export folds
        for i, fold in enumerate(folds):
            fold.to_csv(os.path.join(output,'split_'+str(i+1)+'.csv'))

    def process_folds(self):
        for fold in self.create_file_list(self.settings['temp_input']):
            print('processing split', fold, sep=':')
            log_path = os.path.join(self.settings['temp_input'], fold)
            self.log = pd.read_csv(log_path, index_col='Unnamed: 0')
            self.log['end_timestamp'] = pd.to_datetime(self.log['end_timestamp'],
                                                        format='%Y-%m-%d %H:%M:%S')
            self.log = self.log.sort_values(by='event_id')
            print('Expanding event-log')
            self.expanded_log_creation()
            print('Calculating features')
            self.calculate_features()
            print('filter leading events')
            self.log = self.log[self.log.lead_trail==False]
            self.log['fold'] = fold
            self.log.to_csv(os.path.join(self.settings['temp_output'], fold))
            # clean memory
            del self.expanded_log
            del self.log

        # Read proceced folds
        print('Processing outputs')
        folds = list()
        for filename in self.create_file_list(self.settings['temp_output']):
            df = pd.read_csv(os.path.join(self.settings['temp_output'], filename),
                             index_col='Unnamed: 0')
            folds.append(df)

        processed_log = pd.concat(folds, axis=0, ignore_index=True)
        processed_log = processed_log.sort_values(by='event_id')
        # Clean folders
        self.process_temp_folder(self.settings['temp_input'])
        os.rmdir(self.settings['temp_input'])
        self.process_temp_folder(self.settings['temp_output'])
        os.rmdir(self.settings['temp_output'])
        processed_log.to_csv(os.path.join(
            'outputs', 'inter_'+self.settings['file_name'].split('.')[0]+'.csv'))

# =============================================================================
# Expanded log management
# =============================================================================

    def expanded_log_creation(self):
        # Matching events with slices
        ranges = self.split_events(self.log)
        ranges = self.match_slices(ranges)
        ranges_slides = {k: r['events'] for k, r in ranges.items()}
        ranges = pd.DataFrame.from_dict(ranges, orient='index')
        ranges = ranges[['start', 'end', 'duration']]
        ranges_slides = (pd.DataFrame.from_dict(ranges_slides, orient='index')
                         .stack()
                         .reset_index()
                         .drop('level_1', axis=1)
                         .rename(columns={'level_0': 'slide', 0: 'event_id'}))
        ranges_slides = ranges_slides.merge(ranges,
                                            left_on='slide',
                                            right_index=True, how='left')
        del ranges
        # Join with event log
        self.expanded_log = ranges_slides.merge(
            self.log, on='event_id', how='left')
        self.expanded_log = self.expanded_log.rename(
                columns={'userroleid': 'role'})
        wi_cols = ['event_id', 'ev_duration', 'slide', 'end_timestamp',
                   'duration', 'user', 'role', 'task']
        self.expanded_log = self.expanded_log[wi_cols]

    @staticmethod
    def split_events(log):
        log = log.to_dict('records')
        splitted_events = list()
        # Define date-time ranges in event log
        log = sorted(log, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instant since there is no previous
                # timestamp to find a range
                if i == 0:
                    splitted_events.append((events[i]['end_timestamp'], True,
                                            events[i]['event_id']))
                    splitted_events.append((events[i]['end_timestamp'], False,
                                            events[i]['event_id']))
                else:
                    splitted_events.append((events[i-1]['end_timestamp'], True,
                                            events[i]['event_id']))
                    splitted_events.append((events[i]['end_timestamp'], False,
                                            events[i]['event_id']))
        splitted_events.sort(key=lambda tup: tup[0])
        return splitted_events


    @staticmethod
    def match_slices(splitted_events):
        current_set = set()
        ranges = dict()
        current_start = -1
        for endpoint, is_start, symbol in splitted_events:
            if is_start:
                if current_start != -1 and endpoint != current_start and \
                        endpoint >= current_start and current_set:
                    ranges[(len(ranges))] = {
                            'duration': (endpoint-current_start).total_seconds(),
                            'start': current_start,
                            'end': endpoint,
                            'events': current_set.copy()}
                current_start = endpoint
                current_set.add(symbol)
            else:
                if current_start != -1 and endpoint >= current_start and current_set:
                    ranges[(len(ranges))] = {
                            'duration': (endpoint-current_start).total_seconds(),
                            'start': current_start,
                            'end': endpoint,
                            'events': current_set.copy()}
                current_set.remove(symbol)
                current_start = endpoint
        return ranges

# =============================================================================
# Features calculation
# =============================================================================

    def calculate_features(self):
        event_slides = dict()
        for key, group in self.expanded_log.groupby(['event_id']):
            event_slides[key] = group['slide'].to_list()

        # pool = mp.Pool(mp.cpu_count())
        # work_items_objects = [pool.apply_async(
        #     self.filter_calculate,
        #     args=(event_slides, event['event_id'], event['end_timestamp']))
        #     for event in self.log[['event_id', 'end_timestamp']].to_dict('records')]
        # pool.close()
        # pool.join()
        ev_list = self.log[self.log.lead_trail==False]
        ev_list = ev_list[['event_id', 'end_timestamp']].to_dict('records')
        work_items_feat = list()
        for event in ev_list:
            work_items_feat.append(self.filter_calculate(
                event_slides, event['event_id'], event['end_timestamp']))
        # work_items_feat = [r.get() for r in work_items_objects]
        work_items_feat = pd.DataFrame(work_items_feat)
        self.log = self.log.merge(work_items_feat, on='event_id', how='left')


    def filter_calculate(self, event_slides, event_id, end_timestamp):
        work_items = self.expanded_log[self.expanded_log.slide.isin(
            event_slides[event_id])]
        work_items = work_items[
            (work_items.end_timestamp <= end_timestamp) &
            (work_items.duration > 0)]
        # if event_id==114:
        #     print(work_items)
        if self.sub_group == 'pd':
            if work_items.empty:
                return {'event_id': event_id, 'ev_et': 0, 'ev_et_t': 0}
            work_items = self.calculate_work_item_pd(work_items)
            return self.calculate_event_features_pd(work_items, event_id)
        elif self.sub_group == 'rw':
            if work_items.empty:
                return {'event_id': event_id, 'ev_rd': 0, 'ev_rp_occ': 0}
            work_items = self.calculate_work_item_rw(work_items, self.user_role)
            return self.calculate_event_features_rw(work_items, event_id)
        elif self.sub_group == 'inter':
            if work_items.empty:
                return {'event_id': event_id,
                        'ev_et': 0, 'ev_et_t': 0,
                        'ev_rd': 0, 'ev_rp_occ': 0}
            work_items = self.calculate_work_item_pd(work_items)
            work_items = self.calculate_work_item_rw(work_items, self.user_role)
            rw = self.calculate_event_features_rw(work_items, event_id)
            pd = self.calculate_event_features_pd(work_items, event_id)
            return {**rw, **pd}
        else:
            raise ValueError(self.sub_group)

    @staticmethod
    def calculate_work_item_pd(expanded_log):
        # Total enabled tasks i.e. workload (count)(work item)
        # SEQUENCED SELECT COUNT(event_id) FROM expanded_log GROUP BY slide;
        wi_et = pd.DataFrame(
            expanded_log.groupby(['slide'])
            .count()['event_id']
            .reset_index(name='wi_et'))
        # Count of distinct enable tasks i.e. task workload (count)(work item)
        wi_et_t = pd.DataFrame(
            expanded_log.groupby(['slide', 'task'])
            .count()['event_id']
            .reset_index(name='wi_et_t'))
        # Join features to the extended log
        expanded_log = expanded_log.merge(wi_et, on='slide', how='left')
        expanded_log = expanded_log.merge(wi_et_t, on=['slide', 'task'], how='left')
        return expanded_log

    @staticmethod
    def calculate_work_item_rw(expanded_log, df_resources):
        # wi_mt: Same resource ejecuting multiple tasks i.e. multitasking
        # (count)(work item)
        wi_mt = pd.DataFrame(
            expanded_log.groupby(['slide', 'user'])
            .count()['event_id']
            .reset_index(name='wi_mt'))
        # Active resources per pool and resource pool ocupation
        pool_size = (df_resources.groupby(['role'])
                     .count()['user']
                     .reset_index(name='psize'))
        wi_rlc = pd.DataFrame(
            expanded_log.groupby(['slide', 'role'])
            .count()['event_id']
            .reset_index(name='wi_rlc'))
        wi_rlc = wi_rlc.merge(pool_size, on='role', how='left')
        wi_rlc['rp_occ'] = wi_rlc['wi_rlc']/wi_rlc['psize']
        # Join features to the extended log
        expanded_log = expanded_log.merge(wi_mt, on=['slide', 'user'], how='left')
        expanded_log = expanded_log.merge(wi_rlc, on=['slide', 'role'], how='left')
        return expanded_log


    @staticmethod
    def calculate_event_features_pd(work_items, event_id):
        # Weigth of slide calculation (slide duration / event duration)
        work_items['wi_w'] = np.divide(work_items['duration'],
                                         work_items['ev_duration'])
        # Total enabled tasks weighted average i.e. process workload
        work_items['wi_w_et'] = np.multiply(work_items['wi_w'],
                                              work_items['wi_et'])
        ev_et = work_items['wi_w_et'].sum()
        # Count of distinct enable tasks weighted average i.e. task workload
        work_items['wi_w_et_t'] = np.multiply(work_items['wi_w'],
                                                work_items['wi_et_t'])
        ev_et_t = work_items['wi_w_et_t'].sum()
        return {'event_id': event_id, 'ev_et': ev_et, 'ev_et_t': ev_et_t}


    @staticmethod
    def calculate_event_features_rw(work_items, event_id):
        # Weigth of slide calculation (slide duration / event duration)
        work_items['wi_w'] = np.divide(work_items['duration'],
                                       work_items['ev_duration'])
        # Resource dedication per event
        # work_items['wi_rd'] = np.divide(work_items['duration'],
        #                                   work_items['wi_mt'])
        # Resource dedication per event
        work_items['wi_rd'] = np.multiply(work_items['wi_w'],
                                          work_items['wi_mt'])
        ev_rd = work_items['wi_rd'].sum()
        # Resource pool ocupation weighted average
        work_items['wi_w_rp_occ'] = np.multiply(work_items['wi_w'],
                                                work_items['rp_occ'])
        ev_rp_occ = work_items['wi_w_rp_occ'].sum()
        return {'event_id': event_id, 'ev_rd': ev_rd, 'ev_rp_occ': ev_rp_occ}

# =============================================================================
# Support Methods
# =============================================================================

    @staticmethod
    def calculate_event_duration(log):
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            length = len(events)
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instant since there is no previous
                # timestamp to find a range
                events[i]['pos_trace'] = i
                events[i]['trace_len'] = length
                if i == 0:
                    events[i]['ev_duration'] = 0
                    events[i]['ev_acc_duration'] = 0
                else:
                    dur = (
                        events[i]['end_timestamp'] - events[i-1]['end_timestamp']
                        ).total_seconds()
                    events[i]['ev_duration'] = dur
                    events[i]['ev_acc_duration'] = (events[i-1]['ev_acc_duration']
                                                    + dur)
        return pd.DataFrame.from_dict(sorted(log, key=lambda x: x['event_id']))

    @staticmethod
    def create_file_list(path):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for f in files:
                file_list.append(f)
        return file_list

    @staticmethod
    def process_temp_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # clean folder
        for _, _, files in os.walk(folder):
            for file in files:
                os.unlink(os.path.join(folder, file))


def read_log(parameters):
    # Event-log reading options
    parameters['read_options']['filter_d_attrib'] = True
    # Event-log
    log = lr.LogReader(os.path.join('input_files', parameters['file_name']),
                       parameters['read_options'])
    log = pd.DataFrame(log.data)
    return log[~log.task.isin(['Start', 'End'])]


def extract_features(parameters):
    parameters['temp_input'] = os.path.join('input_files', sup.folder_id())
    parameters['temp_output'] = os.path.join('output_files', sup.folder_id())

    log = read_log(parameters)
    # Users info addition
    res_analyzer = rl.ResourcePoolAnalyser(
        log, sim_threshold=parameters['rp_sim'])
    user_role = pd.DataFrame(res_analyzer.resource_table)
    user_role.rename(columns={'resource': 'user'}, inplace=True)
    log = log.merge(user_role, on='user', how='left')
    extractor = IntercasFeatureExtractor(parameters, log, user_role)
