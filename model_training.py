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

from models import model_specialized as msp
from models import model_concatenated as mcat
from models import model_shared as msh
from models import model_shared_cat as mshcat
from models import model_joint as mj

from support_modules.readers import log_reader as lr
from support_modules import role_discovery as rl
from support_modules import nn_support as nsup
from support_modules import support as sup

def training_model(file_name, start_timeformat, end_timeformat, args, no_loops=False):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    parameters = dict()
    input_file_path = os.path.join('input_files', file_name)
    log = lr.LogReader(input_file_path, start_timeformat, end_timeformat, one_timestamp=True)
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
    ac_weights = load_embedded(index_ac, 'ac_'+ file_name.split('.')[0]+'.emb')
    rl_weights = load_embedded(index_rl, 'rl_'+ file_name.split('.')[0]+'.emb')
    # Calculate relative times
    log_df = add_calculated_features(log_df, ac_index, rl_index)

    # Split validation datasets
    log_df_train, log_df_test = nsup.split_train_test(log_df, 0.3) # 70%/30%

    # Input vectorization
    x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp, yt_inp, max_tbtw = vectorization(
        log_df_train, ac_index,
        rl_index, args)

    # Parameters export
    output_folder = os.path.join('output_files', sup.folder_id())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'parameters'))

    parameters['event_log'] = file_name
    parameters['exp_desc'] = args
    parameters['index_ac'] = index_ac
    parameters['index_rl'] = index_rl
    parameters['dim'] = dict(samples=str(x_ac_inp.shape[0]),
                             time_dim=str(x_ac_inp.shape[1]),
                             features=str(len(ac_index)))
    parameters['max_tbtw'] = max_tbtw

    sup.create_json(parameters, os.path.join(output_folder,
                                             'parameters',
                                             'model_parameters.json'))
    sup.create_csv_file_header(log_df_test.to_dict('records'),
                               os.path.join(output_folder,
                                            'parameters',
                                            'test_log.csv'))

    if args['model_type'] == 'joint':
        mj.training_model(x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp,
                          yt_inp, ac_weights, rl_weights, output_folder, args)
    elif args['model_type'] == 'shared':
        msh.training_model(x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp,
                           yt_inp, ac_weights, rl_weights, output_folder, args)
    elif args['model_type'] == 'specialized':
        msp.training_model(x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp,
                           yt_inp, ac_weights, rl_weights, output_folder, args)
    elif args['model_type'] == 'concatenated':
        mcat.training_model(x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp,
                            yt_inp, ac_weights, rl_weights, output_folder, args)
    elif args['model_type'] == 'shared_cat':
        mshcat.training_model(x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp,
                              yt_inp, ac_weights, rl_weights, output_folder, args)

# =============================================================================
# Load embedded matrix
# =============================================================================

def load_embedded(index, filename):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    weights = list()
    input_folder = os.path.join('input_files', 'embedded_matix')
    with open(os.path.join(input_folder, filename), 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if index[cat_ix] == row[1].strip():
                weights.append(list(map(lambda x: float(x), row[2:])))
        csvfile.close()
    return np.array(weights)

# =============================================================================
# Pre-processing: n-gram vectorization
# =============================================================================

def vectorization(log_df, ac_index, rl_index, args):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
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

    x_ac_inp, x_rl_inp, xt_inp = list(), list(), list()
    y_ac_inp, y_rl_inp, yt_inp = list(), list(), list()

    # n-gram definition
    for i in log_df:
        ac_n_grams = list(ngrams(i['ac_order'], args['n_size'], pad_left=True, left_pad_symbol=0))
        rl_n_grams = list(ngrams(i['rl_order'], args['n_size'], pad_left=True, left_pad_symbol=0))
        tn_grams = list(ngrams(i['tbtw'], args['n_size'], pad_left=True, left_pad_symbol=0))
        for j in range(0, len(ac_n_grams)-1):
            x_ac_inp.append(list(ac_n_grams[j]))
            x_rl_inp.append(list(rl_n_grams[j]))
            xt_inp.append(list(tn_grams[j]))
            y_ac_inp.append(list(ac_n_grams[j+1])[-1])
            y_rl_inp.append(list(rl_n_grams[j+1])[-1])
            yt_inp.append(list(tn_grams[j+1])[-1])

    x_ac_inp = np.array(x_ac_inp)
    x_rl_inp = np.array(x_rl_inp)
    xt_inp = np.array(xt_inp)
    xt_inp = xt_inp.reshape((xt_inp.shape[0], xt_inp.shape[1], 1))

    y_ac_inp = np.array(y_ac_inp)
    y_rl_inp = np.array(y_rl_inp)
    yt_inp = np.array(yt_inp)

    y_ac_inp = ku.to_categorical(y_ac_inp, num_classes=len(ac_index))
    y_rl_inp = ku.to_categorical(y_rl_inp, num_classes=len(rl_index))

    return x_ac_inp, x_rl_inp, xt_inp, y_ac_inp, y_rl_inp, yt_inp, max_tbtw

def add_calculated_features(data, ac_index, rl_index):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    ac_idx = lambda x: ac_index[x['task']]
    data['ac_index'] = data.apply(ac_idx, axis=1)

    rl_idx = lambda x: rl_index[x['role']]
    data['rl_index'] = data.apply(rl_idx, axis=1)

    data['tbtw'] = 0
    data['tbtw_norm'] = 0

    data = data.to_dict('records')

    data = sorted(data, key=lambda x: (x['caseid'], x['end_timestamp']))
    for _, group in itertools.groupby(data, key=lambda x: x['caseid']):
        trace = list(group)
        for i, _ in enumerate(trace):
            if i != 0:
                trace[i]['tbtw'] = (trace[i]['end_timestamp'] -
                                    trace[i-1]['end_timestamp']).total_seconds()

    return pd.DataFrame.from_records(data)

def reformat_events(data, ac_index, rl_index):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    data = data.to_dict('records')

    temp_data = list()
    data = sorted(data, key=lambda x: (x['caseid'], x['end_timestamp']))
    for key, group in itertools.groupby(data, key=lambda x: x['caseid']):
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
