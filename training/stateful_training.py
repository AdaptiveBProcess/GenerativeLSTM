# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import os
import csv
import math
import itertools

import keras.utils as ku

import pandas as pd
import numpy as np

from nltk.util import ngrams

from models import model_shared_stateful as stf

from support_modules.readers import log_reader as lr
from support_modules import role_discovery as rl
from support_modules import nn_support as nsup
from support_modules import support as sup

def training_model(timeformat, args, no_loops=False):
    """Main method of the training module.
    Args:
        timeformat (str): event-log date-time format.
        args (dict): parameters for training the network.
        no_loops (boolean): remove loops fom the event-log (optional).
    """
    parameters = dict()
    log = lr.LogReader(os.path.join('input_files', args['file_name']),
                       timeformat, timeformat, one_timestamp=True)
    _, resource_table = rl.read_resource_pool(log, sim_percentage=0.50)
    # Role discovery
    log_df_resources = pd.DataFrame.from_records(resource_table)
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    # Dataframe creation
    log_df = pd.DataFrame.from_records(log.data)
    log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)

    if no_loops:
        log_df = nsup.reduce_loops(log_df)
    # Index creation
    ac_index = create_index(log_df, 'task')
    ac_index['start'] = 0
    ac_index['end'] = len(ac_index)
    index_ac = {v: k for k, v in ac_index.items()}

    rl_index = create_index(log_df, 'role')
    rl_index['start'] = 0
    rl_index['end'] = len(rl_index)
    index_rl = {v: k for k, v in rl_index.items()}

    # Load embedded matrix
    ac_weights = load_embedded(index_ac, 'ac_'+ args['file_name'].split('.')[0]+'.emb')
    rl_weights = load_embedded(index_rl, 'rl_'+ args['file_name'].split('.')[0]+'.emb')
    # Calculate relative times
    log_df = add_calculated_features(log_df, ac_index, rl_index)
    # Split validation datasets
    log_df_train, log_df_test = nsup.split_train_test(log_df, 0.3) # 70%/30%
    # Input vectorization
    vec = vectorization(log_df_train, ac_index, rl_index, args)
    # Parameters export
    output_folder = os.path.join('output_files', sup.folder_id())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'parameters'))

    parameters['event_log'] = args['file_name']
    parameters['exp_desc'] = args
    parameters['index_ac'] = index_ac
    parameters['index_rl'] = index_rl
    parameters['dim'] = dict(samples=str(np.sum([x.shape[0] for x in vec['prefixes']['x_ac_inp']])),
                             time_dim=str(vec['prefixes']['x_ac_inp'][0].shape[1]),
                             features=str(len(ac_index)))
    parameters['max_tbtw'] = vec['max_tbtw']

    sup.create_json(parameters, os.path.join(output_folder,
                                             'parameters',
                                             'model_parameters.json'))
    sup.create_csv_file_header(log_df_test.to_dict('records'),
                               os.path.join(output_folder,
                                            'parameters',
                                            'test_log.csv'))
#    print([x.shape for x in vec['prefixes']['x_ac_inp']])
    stf.training_model(vec, ac_weights, rl_weights, output_folder, args)


# =============================================================================
# Load embedded matrix
# =============================================================================

def load_embedded(index, filename):
    """Loading of the embedded matrices.
    Args:
        index (dict): index of activities or roles.
        filename (str): filename of the matrix file.
    Returns:
        numpy array: array of weights.
    """
    weights = list()
    input_folder = os.path.join('input_files', 'embedded_matix')
    with open(os.path.join(input_folder, filename), 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if index[cat_ix] == row[1].strip():
                weights.append([float(x) for x in row[2:]])
        csvfile.close()
    return np.array(weights)

# =============================================================================
# Pre-processing: n-gram vectorization
# =============================================================================
def vectorization(log_df, ac_index, rl_index, args):
    """Example function with types documented in the docstring.
    Args:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        args (dict): parameters for training the network
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    if args['norm_method'] == 'max':
        max_tbtw = np.max(log_df.tbtw)
        norm = lambda x: x['tbtw']/max_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)
    elif args['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        log_df['tbtw_log'] = log_df.apply(logit, axis=1)
        max_tbtw = np.max(log_df.tbtw_log)
        norm = lambda x: x['tbtw_log']/max_tbtw
        log_df['tbtw_norm'] = log_df.apply(norm, axis=1)
        log_df = reformat_events(log_df, ac_index, rl_index)

    vec = {'prefixes':dict(), 'next_evt':dict(), 'max_tbtw':max_tbtw}
    # n-gram definition
    vec['prefixes']['x_ac_inp'] = list()
    vec['prefixes']['x_rl_inp'] = list() 
    vec['prefixes']['xt_inp'] = list()
    vec['next_evt']['y_ac_inp'] = list()
    vec['next_evt']['y_rl_inp'] = list()
    vec['next_evt']['yt_inp'] = list()
    
    for i, _ in enumerate(log_df):
        ac_n_grams = list(ngrams(log_df[i]['ac_order'], args['n_size'],
                                 pad_left=True, left_pad_symbol=0))
        rl_n_grams = list(ngrams(log_df[i]['rl_order'], args['n_size'],
                                 pad_left=True, left_pad_symbol=0))
        tn_grams = list(ngrams(log_df[i]['tbtw'], args['n_size'],
                               pad_left=True, left_pad_symbol=0))
        x_ac_inp = np.array([ac_n_grams[0]])
        x_rl_inp = np.array([rl_n_grams[0]])
        xt_inp = np.array([tn_grams[0]])
        y_ac_inp = np.array(ac_n_grams[1][-1])
        y_rl_inp = np.array(rl_n_grams[1][-1])
        yt_inp = np.array(tn_grams[1][-1])
        for j in range(1, len(ac_n_grams)-1):
            x_ac_inp = np.concatenate((x_ac_inp, np.array([ac_n_grams[j]])), axis=0)
            x_rl_inp = np.concatenate((x_rl_inp, np.array([rl_n_grams[j]])), axis=0)
            xt_inp = np.concatenate((xt_inp, np.array([tn_grams[j]])), axis=0)
            y_ac_inp = np.append(y_ac_inp, np.array(ac_n_grams[j+1][-1]))
            y_rl_inp = np.append(y_rl_inp, np.array(rl_n_grams[j+1][-1]))
            yt_inp = np.append(yt_inp, np.array(tn_grams[j+1][-1]))
        xt_inp = xt_inp.reshape((xt_inp.shape[0], xt_inp.shape[1], 1))
        y_ac_inp = ku.to_categorical(y_ac_inp, num_classes=len(ac_index))
        y_rl_inp = ku.to_categorical(y_rl_inp, num_classes=len(rl_index))
        vec['prefixes']['x_ac_inp'].append(x_ac_inp)
        vec['prefixes']['x_rl_inp'].append(x_rl_inp) 
        vec['prefixes']['xt_inp'].append(xt_inp)
        vec['next_evt']['y_ac_inp'].append(y_ac_inp)
        vec['next_evt']['y_rl_inp'].append(y_rl_inp)
        vec['next_evt']['yt_inp'].append(yt_inp)
    return vec


def add_calculated_features(log_df, ac_index, rl_index):
    """Appends the indexes and relative time to the dataframe.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        Dataframe: The dataframe with the calculated features added.
    """
    ac_idx = lambda x: ac_index[x['task']]
    log_df['ac_index'] = log_df.apply(ac_idx, axis=1)

    rl_idx = lambda x: rl_index[x['role']]
    log_df['rl_index'] = log_df.apply(rl_idx, axis=1)

    log_df['tbtw'] = 0
    log_df['tbtw_norm'] = 0

    log_df = log_df.to_dict('records')

    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    for _, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        for i, _ in enumerate(trace):
            if i != 0:
                trace[i]['tbtw'] = (trace[i]['end_timestamp'] -
                                    trace[i-1]['end_timestamp']).total_seconds()

    return pd.DataFrame.from_records(log_df)

def reformat_events(log_df, ac_index, rl_index):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    log_df = log_df.to_dict('records')

    temp_data = list()
    log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        ac_order = [x['ac_index'] for x in trace]
        rl_order = [x['rl_index'] for x in trace]
        tbtw = [x['tbtw_norm'] for x in trace]
        ac_order.insert(0, ac_index[('start')])
        ac_order.append(ac_index[('end')])
        rl_order.insert(0, rl_index[('start')])
        rl_order.append(rl_index[('end')])
        tbtw.insert(0, 0)
        tbtw.append(0)
        temp_dict = dict(caseid=key,
                         ac_order=ac_order,
                         rl_order=rl_order,
                         tbtw=tbtw)
        temp_data.append(temp_dict)

    return temp_data


# =============================================================================
# Support
# =============================================================================


def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
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

def max_serie(log_df, serie):
    """Returns the max and min value of a column.
    Args:
        log_df: dataframe.
        serie: name of the serie.
    Returns:
        max and min value.
    """
    max_value, min_value = 0, 0
    for record in log_df:
        if np.max(record[serie]) > max_value:
            max_value = np.max(record[serie])
        if np.min(record[serie]) > min_value:
            min_value = np.min(record[serie])
    return max_value, min_value

def max_min_std(val, max_value, min_value):
    """Standardize a number between range.
    Args:
        val: Value to be standardized.
        max_value: Maximum value of the range.
        min_value: Minimum value of the range.
    Returns:
        Standardized value between 0 and 1.
    """
    std = (val - min_value) / (max_value - min_value)
    return std
