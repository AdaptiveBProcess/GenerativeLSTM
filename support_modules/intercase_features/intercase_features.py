# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import os
import itertools
from operator import itemgetter


import pandas as pd
import numpy as np
from support_modules import nn_support as nsup
from support_modules import role_discovery as rl
from support_modules.readers import log_reader as lr


def calculate_intercase_features(args, log, df_resources):
    """Main method of intercase features calculations.
    Args:
        args (dict): parameters for training the network.
    """
    # Event indexing
    log['event_id'] = log.index
    log = calculate_event_duration(log, args)
    # splitt events in start and complete
    splitted_events = splitt_events(log, args)
    # Matching events with slices
    ranges = match_slices(splitted_events)
    # Reshaping of ranges
    simplified = dict()
    for i in range(0, len(ranges)):
        simplified[i] = list(ranges[i]['events'])
    ranges_df = pd.DataFrame.from_dict(ranges, orient='index')
    ranges_dur = ranges_df[['start', 'end', 'duration']]
    ranges_slides = (pd.DataFrame.from_dict(simplified, orient='index')
                        .stack()
                        .reset_index()
                        .drop('level_1', axis=1)
                        .rename(columns={'level_0': 'slide', 0: 'event_id'}))
    ranges_slides = ranges_slides.merge(ranges_dur, left_on='slide', right_index=True, how='left')
    # Join with event log
    expanded_log = ranges_slides.merge(log, on='event_id', how='left')
    # Work item level features calculation
    expanded_log = calculate_work_item_features(expanded_log, df_resources)
    # Event level features calculation
    log = calculate_event_features(expanded_log, log)
    log = nsup.scale_feature(log, 'ev_et', 'max', True)
    log = nsup.scale_feature(log, 'ev_et_t', 'max', True)
    log = nsup.scale_feature(log, 'ev_rp_occ', 'max', True)
    log = nsup.scale_feature(log, 'ev_acc_t', 'activity', True)
    return log
    

def calculate_work_item_features(expanded_log, df_resources):
#   Total enabled tasks (count)(work item) 
#   SEQUENCED SELECT COUNT(event_id) FROM expanded_log GROUP BY slide;
    wi_et = pd.DataFrame(
            expanded_log.groupby(['slide'])
            .count()['event_id']
            .reset_index(name = 'wi_et'))
#   Count of distinct enable tasks (count)(work item)
    wi_et_t = pd.DataFrame(
            expanded_log.groupby(['slide', 'task'])
            .count()['event_id']
            .reset_index(name = 'wi_et_t'))
#   wi_mt: Same resource ejecuting multiple tasks (count)(work item)
    wi_mt = pd.DataFrame(
            expanded_log.groupby(['slide', 'user'])
            .count()['event_id']
            .reset_index(name = 'wi_mt'))
#   Active resources per pool and resource pool ocupation 
    pool_size = df_resources.groupby(['role']).count()['user'].reset_index(name = 'psize')
#    print(pool_size)
    wi_rlc = pd.DataFrame(
            expanded_log.groupby(['slide', 'role'])
            .count()['event_id']
            .reset_index(name = 'wi_rlc'))
    wi_rlc = wi_rlc.merge(pool_size, on='role', how='left')
    wi_rlc['rp_occ'] = wi_rlc['wi_rlc']/wi_rlc['psize']
#   Accumulated time per activity
    wi_acc_t = pd.DataFrame(
            expanded_log.groupby(['slide', 'task'])
            .sum()['duration']
            .reset_index(name = 'wi_acc_t'))
#   Join features to the extended log 
    expanded_log = expanded_log.merge(wi_et, on='slide', how='left')
    expanded_log = expanded_log.merge(wi_et_t, on=['slide', 'task'], how='left')
    expanded_log = expanded_log.merge(wi_mt, on=['slide', 'user'], how='left')
    expanded_log = expanded_log.merge(wi_rlc, on=['slide', 'role'], how='left')
    expanded_log = expanded_log.merge(wi_acc_t, on=['slide', 'task'], how='left')
    return expanded_log

    
def calculate_event_features(expanded_log, log):
#   Weigth of slide calculation (slide duration / event duration)
    expanded_log['wi_w'] = np.divide(expanded_log['duration'],
                                        expanded_log['ev_duration'])
#   Total enabled tasks weighted average
    expanded_log['wi_w_et'] = np.multiply(expanded_log['wi_w'],
                                              expanded_log['wi_et'])
    ev_et = pd.DataFrame(
            expanded_log.groupby(['event_id'])
            .sum()['wi_w_et']
            .reset_index(name = 'ev_et'))
#   Count of distinct enable tasks weighted average
    expanded_log['wi_w_et_t'] = np.multiply(expanded_log['wi_w'],
                                              expanded_log['wi_et_t'])
    ev_et_t = pd.DataFrame(
            expanded_log.groupby(['event_id'])
            .sum()['wi_w_et_t']
            .reset_index(name = 'ev_et_t'))
#   Resource dedication per event
    expanded_log['wi_rd'] = np.divide(expanded_log['duration'],
                                              expanded_log['wi_mt'])
    ev_rd = pd.DataFrame(
            expanded_log.groupby(['event_id'])
            .sum()['wi_rd']
            .reset_index(name = 'ev_rd'))
#   Resource pool ocupation weighted average
    expanded_log['wi_w_rp_occ'] = np.multiply(expanded_log['wi_w'],
                                                  expanded_log['rp_occ'])
    ev_rp_occ = pd.DataFrame(
            expanded_log.groupby(['event_id'])
            .sum()['wi_w_rp_occ']
            .reset_index(name = 'ev_rp_occ'))
#   Accumulated duration per activity
#    expanded_log['wi_ac_t'] = np.multiply(expanded_log['wi_w'],
#                                            expanded_log['wi_acc_t'])
    ev_acc_t = pd.DataFrame(
            expanded_log.groupby(['event_id'])
            .sum()['wi_acc_t']
            .reset_index(name = 'ev_acc_t'))
#   Join with log dataframe
    log = log.merge(ev_et, on='event_id', how='left')
    log = log.merge(ev_et_t, on='event_id', how='left')
    log = log.merge(ev_rd, on='event_id', how='left')
    log = log.merge(ev_rp_occ, on='event_id', how='left')
    log = log.merge(ev_acc_t, on='event_id', how='left')
#   Rate of resource dedication per event 
    log['ev_rd_p'] = np.divide(log['ev_rd'],log['ev_duration'])
    log['ev_rd_p'].fillna(0, inplace=True)
    return log
    
def calculate_event_duration(log, args):
    log = log.to_dict('records')
    if args['one_timestamp']:
        log = sorted(log, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace is taken as instant
                # since there is no previous timestamp to find a range
                if i == 0:
                    events[i]['ev_duration'] = 0
                    events[i]['ev_acc_duration'] = 0
                else:
                    dur = (events[i]['end_timestamp']-events[i-1]['end_timestamp']).total_seconds() 
                    events[i]['ev_duration'] = dur
                    events[i]['ev_acc_duration'] = events[i-1]['ev_acc_duration'] + dur
    else:
        log = sorted(log, key=itemgetter('start_timestamp'))
        for event in log:
            # on the contrary is btw start and complete timestamp 
            event['ev_duration']=(event['end_timestamp'] - event['start_timestamp']).total_seconds()
    return pd.DataFrame.from_dict(sorted(log, key=lambda x: x['event_id']))
    

def splitt_events(log, args):
    log = log.to_dict('records')
    splitted_events = list()
    # Define date-time ranges in event log   
    if args['one_timestamp']:
        log = sorted(log, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace is taken as instant
                # since there is no previous timestamp to find a range
                if i == 0:
                    splitted_events.append((events[i]['end_timestamp'], True, events[i]['end_timestamp'], events[i]['event_id']))
                    splitted_events.append((events[i]['end_timestamp'], False, events[i]['end_timestamp'], events[i]['event_id']))
                else:
                    splitted_events.append((events[i-1]['end_timestamp'], True, events[i]['end_timestamp'], events[i]['event_id']))
                    splitted_events.append((events[i]['end_timestamp'], False, events[i-1]['end_timestamp'], events[i]['event_id']))
    else:
        log = sorted(log, key=itemgetter('start_timestamp'))
        for event in log:
            # on the contrary is btw start and complete timestamp 
            splitted_events.append((event['start_timestamp'], True, event['end_timestamp'], event['event_id']))
            splitted_events.append((event['end_timestamp'], False, event['start_timestamp'], event['event_id']))

    splitted_events.sort(key=lambda tup: tup[0])
    return splitted_events

def match_slices(splitted_events):
    current_set = set()
    ranges = dict()
    current_start = -1

    for endpoint, is_start, other, symbol in splitted_events:
        if is_start:
            if current_start != -1 and endpoint != current_start and \
                   endpoint >= current_start and current_set:
                ranges[(len(ranges))]={
                        'duration': (endpoint-current_start).total_seconds(),
                        'start': current_start,
                        'end': endpoint,
                        'events': current_set.copy()}
            current_start = endpoint
            current_set.add(symbol)
        else:
            if current_start != -1 and endpoint >= current_start and current_set:
                ranges[(len(ranges))]={
                        'duration': (endpoint-current_start).total_seconds(),
                        'start': current_start,
                        'end': endpoint,
                        'events': current_set.copy()}
            current_set.remove(symbol)
            current_start = endpoint
    return ranges   