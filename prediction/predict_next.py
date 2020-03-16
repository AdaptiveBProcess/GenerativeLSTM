# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import os
import itertools
import random
import time


from keras.models import load_model

import pandas as pd
import numpy as np

from support_modules import support as sup
from support_modules import nn_support as nsup
from support_modules.readers import log_reader as lr


def predict_next(parameters, is_single_exec=True):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """
    output_route = os.path.join('output_files', parameters['folder'])
    model_name, _ = os.path.splitext(parameters['model_file'])
    # Loading of testing dataframe
    parameters['read_options']['filter_d_attrib'] = False
    df_test = lr.LogReader(os.path.join(output_route, 'parameters', 'test_log.csv'), parameters['read_options'])
    df_test = pd.DataFrame(df_test.data)
    
    # Loading of parameters from training
    with open(os.path.join(output_route, 'parameters', 'model_parameters.json')) as file:
        data = json.load(file)
        parameters = {**parameters, **{k: v for k, v in data.items()}}
        parameters['dim'] = {k: int(v) for k, v in data['dim'].items()}
        parameters['max_dur'] = float(data['max_dur'])
        parameters['index_ac'] = {int(k): v for k, v in data['index_ac'].items()}
        parameters['index_rl'] = {int(k): v for k, v in data['index_rl'].items()}
        file.close()

    df_test = nsup.scale_feature(df_test, 'dur', parameters['norm_method'])
    
    ac_index = {v:k for k, v in parameters['index_ac'].items()}
    rl_index = {v:k for k, v in parameters['index_rl'].items()}

#   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 1},
                {'imp': 'Arg Max', 'rep': 1}]
#    variants = [{'imp': 'Arg Max', 'rep': 1}]
#   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))
#   Examples definition
    if parameters['model_type'] == 'shared_cat':
        examples = create_pref_suf(df_test, ac_index, rl_index, parameters)
    elif parameters['model_type'] == 'shared_cat_inter':
        examples = create_pref_suf_inter(df_test, ac_index, rl_index, parameters)
#   Prediction and measurement
    for var in variants:
        act_measures, role_measures = list(), list()
        for i in range(0, var['rep']):
            seed = time.time()
            random.seed(seed)
            results = predict(model, examples, var['imp'], parameters)
            act_accuracy = np.divide(np.sum(results['ac_true']),len(results['ac_true']))
            role_accuracy = np.divide(np.sum(results['rl_true']),len(results['rl_true']))
            # Save results
            exp_desc = parameters.copy()
            exp_desc.pop('read_options', None)
            exp_desc.pop('column_names', None)
            exp_desc.pop('one_timestamp', None)
            exp_desc.pop('reorder', None)
            exp_desc.pop('index_ac', None)
            exp_desc.pop('index_rl', None)
            exp_desc.pop('dim', None)
            exp_desc.pop('max_dur', None)

            act_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **{'accuracy': act_accuracy},
                                    **{'random_seed':seed}, **exp_desc})
            role_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **{'accuracy': role_accuracy},
                                    **{'random_seed':seed}, **exp_desc})
            save_results(act_measures, 'activity', is_single_exec, parameters)
            save_results(role_measures, 'roles', is_single_exec, parameters)

def save_results(measurements, feature, is_single_exec, parameters):    
    output_route = os.path.join('output_files', parameters['folder'])
    model_name, _ = os.path.splitext(parameters['model_file'])
    if measurements:    
        if is_single_exec:
                sup.create_csv_file_header(measurements, os.path.join(output_route,
                                                                      model_name +'_'+feature+'_next_event.csv'))
        else:
            if os.path.exists(os.path.join('output_files', feature+'_next_event.csv')):
                sup.create_csv_file(measurements, os.path.join('output_files',
                                                               feature+'_next_event.csv'), mode='a')
            else:
                sup.create_csv_file_header(measurements, os.path.join('output_files',
                                                               feature+'_next_event.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def predict(model, examples, imp, parameters):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    results = {'ac_true':list(), 'rl_true':list()}
    for i, _ in enumerate(examples['prefixes']['activities']):
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['activities'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((1,parameters['dim']['time_dim']))
                
        x_rl_ngram = np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['roles'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((1,parameters['dim']['time_dim']))

        # times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['times'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], 1))])
        # add intercase features if necessary
        if parameters['model_type'] == 'shared_cat':
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        elif parameters['model_type'] == 'shared_cat_inter':
            # times input shape(1,5,1)
            inter_attr_num = examples['prefixes']['inter_attr'][i].shape[1]
            x_inter_ngram = np.array([np.append(
                    np.zeros((parameters['dim']['time_dim'], inter_attr_num)),
                    examples['prefixes']['inter_attr'][i],
                    axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], inter_attr_num))])
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
        # predict
        predictions = model.predict(inputs)
        if imp == 'Random Choice':
            # Use this to get a random choice following as PDF the predictions
            pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
            pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
        elif imp == 'Arg Max':
            # Use this to get the max prediction
            pos = np.argmax(predictions[0][0])
            pos1 = np.argmax(predictions[1][0])
        # Activities accuracy evaluation
        if pos == examples['next_evt']['activities'][i]:
            results['ac_true'].append(1)
        else:
            results['ac_true'].append(0)
        # Roles accuracy evaluation
        if pos1 == examples['next_evt']['roles'][i]:
            results['rl_true'].append(1)
        else:
            results['rl_true'].append(0)
    sup.print_done_task()
    return results

# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test, ac_index, rl_index, parameters):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    columns = ['ac_index', 'rl_index', 'dur_norm']
    df_test = reformat_events(df_test, ac_index, rl_index, columns, parameters)
    examples = {'prefixes':dict(), 'next_evt':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    for i, _ in enumerate(df_test):
        for x in columns:
            serie = [df_test[i][x][:idx] for idx in range(1, len(df_test[i][x]))]
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
            examples['next_evt'][equi[x]] = examples['next_evt'][equi[x]] + y_serie if i > 0 else y_serie
    return examples

def create_pref_suf_inter(df_test, ac_index, rl_index, parameters):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    columns = ['caseid', 'task', 'user', 'start_timestamp', 'end_timestamp', 
               'dur_log', 'role', 'event_id', 'ev_duration', 'dur', 'ev_rd']
    columns = [x for x in list(df_test.columns) if x not in columns]
    df_test = reformat_events(df_test, ac_index, rl_index, columns, parameters)
    examples = {'prefixes':dict(), 'next_evt':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    x_inter_dict, y_inter_dict = dict(), dict()
    for i, _ in enumerate(df_test):
        for x in columns:
            serie = [df_test[i][x][:idx] for idx in range(1, len(df_test[i][x]))]
            y_serie = [x[-1] for x in serie]
            serie = serie[:-1]
            y_serie = y_serie[1:]
            if x in list(equi.keys()): 
                examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
                examples['next_evt'][equi[x]] = examples['next_evt'][equi[x]] + y_serie if i > 0 else y_serie
            else:
                x_inter_dict[x] = x_inter_dict[x] + serie if i > 0 else serie
                y_inter_dict[x] = y_inter_dict[x] + y_serie if i > 0 else y_serie
    # Reshape intercase attributes (prefixes, n-gram size, number of attributes)
    examples['prefixes']['inter_attr'] = list()
    x_inter_dict = pd.DataFrame(x_inter_dict)
    for row in x_inter_dict.values:
        new_row = [np.array(x) for x in row]
        new_row = np.dstack(new_row)        
        new_row =  new_row.reshape((new_row.shape[1], new_row.shape[2]))
        examples['prefixes']['inter_attr'].append(new_row)
    # Reshape intercase expected attributes (prefixes, number of attributes)
    examples['next_evt']['inter_attr'] = list()
    y_inter_dict = pd.DataFrame(y_inter_dict)    
    for row in y_inter_dict.values:
        new_row = [np.array(x) for x in row]
        new_row = np.dstack(new_row)
        new_row =  new_row.reshape((new_row.shape[2]))
        examples['next_evt']['inter_attr'].append(new_row)
    return examples

def reformat_events(log_df, ac_index, rl_index, columns, args):
    """Creates series of activities, roles and relative times per trace.
    Args:
        log_df: dataframe.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
    Returns:
        list: lists of activities, roles and relative times.
    """
    temp_data = list()
    log_df = log_df.to_dict('records')
    if args['one_timestamp']:
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