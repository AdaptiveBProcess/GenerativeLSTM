# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:53:43 2019

@author: Manuel Camargo
"""
import os
import csv
import itertools
from operator import itemgetter

import keras.utils as ku

import pandas as pd
import numpy as np

from nltk.util import ngrams

from models import model_shared_cat as mshcat
from models import model_shared_cat_intercase as mshcati
from models import model_seq2seq as seq
from models import model_seq2seq_intercase as seqi

from support_modules.readers import log_reader as lr
from support_modules.intercase_features import intercase_features as inf
from support_modules import role_discovery as rl
from support_modules import nn_support as nsup
from support_modules import support as sup
from support_modules import forest_importances as fi


def training_model(parms):
    """Main method of the training module.
    parms:
        parms (dict): parms for training the network.
    """
    # Dataframe creation
    # Filter load local inter-case features or filter them
    if parms['model_type'] in ['seq2seq_inter_full', 'shared_cat_inter_full']:
        keep_cols = ['caseid', 'task', 'user', 'start_timestamp',
                     'end_timestamp', 'ac_index', 'event_id', 'rl_index',
                     'Unnamed: 0', 'dur', 'ev_duration', 'role', 'ev_rd']
        parms['read_options']['filter_d_attrib'] = False
        log = lr.LogReader(os.path.join('input_files', parms['file_name']), parms['read_options'])
        log_df = pd.DataFrame(log.data)
        # Scale loaded inter-case features
        colnames = list(log_df.columns.difference(keep_cols))
        for col in colnames:
            log_df = nsup.scale_feature(log_df, col, 'max', True)
    else:
        parms['read_options']['filter_d_attrib'] = True
        log = lr.LogReader(os.path.join('input_files', parms['file_name']), parms['read_options'])
        log_df = pd.DataFrame(log.data)
    # Resource pool discovery
    # _, resource_table = rl.read_resource_pool(log_df, sim_percentage=parms['rp_similarity'], dataframe=True)
    res_analyzer = rl.ResourcePoolAnalyser(log_df, sim_threshold=parms['rp_similarity'])
    # Role discovery
    log_df_resources = pd.DataFrame.from_records(res_analyzer.resource_table)
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    # Dataframe creation
    log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)
    
    # Calculate general inter-case features
    if parms['model_type'] in ['seq2seq_inter', 'shared_cat_inter', 'shared_cat_inter_full']:
        log_df = inf.calculate_intercase_features(parms, log_df, log_df_resources)
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
    ac_weights = load_embedded(index_ac, 'ac_'+ parms['file_name'].split('.')[0]+'.emb')
    rl_weights = load_embedded(index_rl, 'rl_'+ parms['file_name'].split('.')[0]+'.emb')
    # Calculate relative times
    log_df = add_calculated_features(log_df, ac_index, rl_index, parms)
    # Split validation datasets
    log_df_train, log_df_test = nsup.split_train_test(log_df, 0.3, parms['one_timestamp'])
    #   Output folder 
    output_folder = os.path.join('output_files', sup.folder_id())
    # Input vectorization
    if parms['model_type'] == 'shared_cat':
        examples = vectorization_shared_cat(log_df_train, ac_index, rl_index, parms)
        export_parms(output_folder, parms, log_df_test, index_ac, ac_index, index_rl, examples)
        mshcat.training_model(examples, ac_weights, rl_weights, output_folder, parms)
    elif parms['model_type'] == 'shared_cat_inter':
        examples = vectorization_shared_cat_inter(log_df_train, ac_index, rl_index, parms)
        export_parms(output_folder, parms, log_df_test, index_ac, ac_index, index_rl, examples)
        mshcati.training_model(examples, ac_weights, rl_weights, output_folder, parms)
    elif parms['model_type'] == 'seq2seq':
        examples = vectorization_seq2seq(log_df_train, log_df, ac_index, rl_index, parms)
        export_parms(output_folder, parms, log_df_test, index_ac, ac_index, index_rl, examples)
        seq.training_model(examples, ac_weights, rl_weights, output_folder, parms)
    elif parms['model_type'] == 'seq2seq_inter':
        examples = vectorization_seq2seq_inter(log_df_train, log_df, ac_index, rl_index, parms)
        export_parms(output_folder, parms, log_df_test, index_ac, ac_index, index_rl, examples)
        seqi.training_model(examples, ac_weights, rl_weights, output_folder, parms)
    elif parms['model_type'] == 'shared_cat_inter_full':
        log_df = nsup.feat_sel_eval_correlation(log_df, 0.8, keep_cols=keep_cols)
        fi.calculate_importances(log_df, keep_cols)
#        sup.create_csv_file_header(correlation.to_dict('records'), 'corr_matrix.csv')



# =============================================================================
# Pre-processing: n-gram vectorization
# =============================================================================
def vectorization_shared_cat(log_df, ac_index, rl_index, parms):
    """Example function with types documented in the docstring.
    parms:
        log_df (dataframe): event log data.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        parms (dict): parms for training the network
    Returns:
        dict: Dictionary that contains all the LSTM inputs.
    """
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    log_df = nsup.scale_feature(log_df, 'dur', parms['norm_method'])
    columns = list(equi.keys())
    vec = {'prefixes':dict(), 'next_evt':dict(), 'max_dur':np.max(log_df.dur)}
    log_df = reformat_events(log_df, ac_index, rl_index, columns, parms)
    # n-gram definition
    for i, _ in enumerate(log_df):
        for x in columns:
            serie = list(ngrams(log_df[i][x], parms['n_size'],
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
    vec['next_evt']['activities'] = ku.to_categorical(vec['next_evt']['activities'],
                                                    num_classes=len(ac_index))
    vec['next_evt']['roles'] = ku.to_categorical(vec['next_evt']['roles'],
                                                    num_classes=len(rl_index))
    return vec


def vectorization_shared_cat_inter(log_df, ac_index, rl_index, parms):
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
    log_df = nsup.scale_feature(log_df, 'dur', parms['norm_method'])
    columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp', 
               'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
    columns = [x for x in list(log_df.columns) if x not in columns]
    vec = {'prefixes':dict(), 'next_evt':dict(), 'max_dur':np.max(log_df.dur)}
    log_df = reformat_events(log_df, ac_index, rl_index, columns, parms)
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    x_inter_dict = dict()
    y_inter_dict = dict()
    for i, _ in enumerate(log_df):
        for x in columns:
            serie = list(ngrams(log_df[i][x], parms['n_size'],
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
                                                    num_classes=len(ac_index))
    vec['next_evt']['roles'] = ku.to_categorical(vec['next_evt']['roles'],
                                                    num_classes=len(rl_index))
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

def vectorization_seq2seq(df_train, log_df, ac_index, rl_index, parms):
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
    df_train = nsup.scale_feature(df_train, 'dur', parms['norm_method'])
    examples = {'encoder_input_data':dict(), 'decoder_target_data':dict(), 'max_dur':np.max(df_train.dur)}
    df_train = reformat_events(df_train, ac_index, rl_index, columns, parms)
    max_length = np.max([len(x['ac_index']) for x in 
                         reformat_events(log_df, ac_index, rl_index, ['ac_index'], parms)])
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    for i, _ in enumerate(df_train):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_train[i][x])):
                serie.append([0]*(max_length - idx) + df_train[i][x][:idx])
                y_serie.append(df_train[i][x][idx:] + [0]*(max_length - len(df_train[i][x][idx:])))
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
    examples['decoder_target_data']['activities'] = ku.to_categorical(examples['decoder_target_data']['activities'],
                                                    num_classes=len(ac_index))
    examples['decoder_target_data']['roles'] = ku.to_categorical(examples['decoder_target_data']['roles'],
                                                    num_classes=len(rl_index))
   
    return examples

def vectorization_seq2seq_inter(df_train, log_df, ac_index, rl_index, parms):
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
    df_train = nsup.scale_feature(df_train, 'dur', parms['norm_method'])
    columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp', 
               'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
    columns = [x for x in list(df_train.columns) if x not in columns]
    examples = {'encoder_input_data':dict(), 'decoder_target_data':dict(), 'max_dur':np.max(df_train.dur)}
    df_train = reformat_events(df_train, ac_index, rl_index, columns, parms)
    max_length = np.max([len(x['ac_index']) for x in 
                         reformat_events(log_df, ac_index, rl_index, ['ac_index'], parms)])
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    x_inter_dict = dict()
    for i, _ in enumerate(log_df):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_train[i][x])):
                serie.append([0]*(max_length - idx) + df_train[i][x][:idx])
                y_serie.append(df_train[i][x][idx:] + [0]*(max_length - len(df_train[i][x][idx:])))
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
                                                    num_classes=len(ac_index))
    examples['decoder_target_data']['roles'] = ku.to_categorical(examples['decoder_target_data']['roles'],
                                                    num_classes=len(rl_index))
    # Reshape intercase attributes (prefixes, n-gram size, number of attributes) 
    for key, value in x_inter_dict.items():
        x_inter_dict[key] = np.array(value)
        x_inter_dict[key] = x_inter_dict[key].reshape((x_inter_dict[key].shape[0],
                   x_inter_dict[key].shape[1], 1))        
    examples['encoder_input_data']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
#    examples['decoder_target_data']['inter_attr'] = np.dstack(list(y_inter_dict.values()))[0]
    return examples

# =============================================================================
# Reformat events
# =============================================================================
def reformat_events(log_df, ac_index, rl_index, columns, parms):
    """Creates series of activities, roles and relative times per trace.
    parms:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    temp_data = list()
    log_df = log_df.to_dict('records')
    if parms['one_timestamp']:
        log_df = sorted(log_df, key=lambda x: (x['caseid'], x['end_timestamp']))
    else:
        log_df = sorted(log_df, key=lambda x: (x['caseid'], x['start_timestamp']))
    for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
        trace = list(group)
        temp_dict = dict()
        for x in columns:
            serie = [y[x] for y in trace]
            if x == 'ac_index':
                serie.insert(0, ac_index[('start')])
                serie.append(ac_index[('end')])
            elif x == 'rl_index':
                serie.insert(0, rl_index[('start')])
                serie.append(rl_index[('end')])
            else:
                serie.insert(0, 0)
                serie.append(0)
            temp_dict = {**{x: serie},**temp_dict}
        temp_dict = {**{'caseid': key},**temp_dict}
        temp_data.append(temp_dict)
    return temp_data

# =============================================================================
# Load embedded matrix
# =============================================================================

def load_embedded(index, filename):
    """Loading of the embedded matrices.
    parms:
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
# Support
# =============================================================================

def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    parms:
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

def add_calculated_features(log_df, ac_index, rl_index, parms):
    """Appends the indexes and relative time to the dataframe.
    parms:
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

    log_df['dur'] = 0

    log_df = log_df.to_dict('records')
    if parms['one_timestamp']:
        log_df = sorted(log_df, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace is taken as instant
                # since there is no previous timestamp to find a range
                if i == 0:
                    events[i]['dur'] = 0
                else:
                    dur = (events[i]['end_timestamp']-events[i-1]['end_timestamp']).total_seconds()
                    events[i]['dur'] = dur
    else:
        log_df = sorted(log_df, key=itemgetter('start_timestamp'))
        for event in log_df:
            # on the contrary is btw start and complete timestamp 
            event['dur']=(event['end_timestamp'] - event['start_timestamp']).total_seconds()
    return pd.DataFrame.from_dict(log_df)

def export_parms(output_folder, parms, log_df_test, index_ac, ac_index, index_rl, vec):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'parameters'))

    parms['index_ac'] = index_ac
    parms['index_rl'] = index_rl
    if parms['model_type'] in ['shared_cat', 'shared_cat_inter']:
        parms['dim'] = dict(samples=str(vec['prefixes']['activities'].shape[0]),
                                 time_dim=str(vec['prefixes']['activities'].shape[1]),
                                 features=str(len(ac_index)))
    else:
        parms['dim'] = dict(samples=str(vec['encoder_input_data']['activities'].shape[0]),
                                 time_dim=str(vec['encoder_input_data']['activities'].shape[1]),
                                 features=str(len(ac_index)))
    parms['max_dur'] = vec['max_dur']

    sup.create_json(parms, os.path.join(output_folder,
                                             'parameters',
                                             'model_parameters.json'))
    sup.create_csv_file_header(log_df_test.to_dict('records'),
                               os.path.join(output_folder,
                                            'parameters',
                                            'test_log.csv'))
    
