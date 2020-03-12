# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import os
import itertools
import jellyfish as jf
import random
import time

from keras.models import load_model

import pandas as pd
import numpy as np

from support_modules import support as sup
from support_modules import nn_support as nsup
from support_modules.readers import log_reader as lr


def predict_suffix_full(parameters, is_single_exec=True):
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
    print(parameters)

    df_test = nsup.scale_feature(df_test, 'dur', parameters['norm_method'])
    
    ac_alias = create_alias(len(parameters['index_ac']))
    rl_alias = create_alias(len(parameters['index_rl']))
    
    ac_index = {v:k for k, v in parameters['index_ac'].items()}
    rl_index = {v:k for k, v in parameters['index_rl'].items()}

#   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 1},
                {'imp': 'Arg Max', 'rep': 1}]
#    variants = [{'imp': 'Random Choice', 'rep': 1}]
#   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))
#   Examples definition    
    if parameters['model_type'] == 'shared_cat':
        examples = create_pref_suf(df_test, ac_index, rl_index, parameters)
    elif parameters['model_type'] == 'shared_cat_inter':
        examples = create_pref_suf_inter(df_test, ac_index, rl_index, parameters)
    elif parameters['model_type'] == 'seq2seq':
        examples = create_pref_suf_seq2seq(df_test, ac_index, rl_index, parameters)
    elif parameters['model_type'] == 'seq2seq_inter':
        examples = create_pref_suf_seq2seq_inter(df_test, ac_index, rl_index, parameters)
    
#   Prediction and measurement
    for var in variants:
        act_measures, role_measures, time_measures = list(), list(), list()
        act_sim_measures, role_sim_measures = list(), list()
        for i in range(0, var['rep']):
            exp_desc = parameters.copy()
            exp_desc.pop('read_options', None)
            exp_desc.pop('column_names', None)
            exp_desc.pop('one_timestamp', None)
            exp_desc.pop('reorder', None)
            exp_desc.pop('index_ac', None)
            exp_desc.pop('index_rl', None)
            exp_desc.pop('dim', None)
            exp_desc.pop('max_dur', None)
            if parameters['model_type'] in ['seq2seq', 'seq2seq_inter']:
                seed = time.time()
                random.seed(seed)
                results = predict_seq2seq(model, examples, var['imp'], parameters)
                act_accuracy = np.divide(np.sum(results['ac_true']),len(results['ac_true']))
                sim_ac = dl_measure(results['ac_suff_pred'], ac_alias)
                role_accuracy = np.divide(np.sum(results['rl_true']),len(results['rl_true']))
                sim_rl = dl_measure(results['rl_suff_pred'], rl_alias)
                mae_times = mae_measure(results['rem_time_pred'], parameters)
                sim_ac = calculate_by_size(sim_ac)
                sim_rl = calculate_by_size(sim_rl)

                # Save results
                act_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **{'accuracy': act_accuracy},
                                    **{'random_seed':seed}, **exp_desc})
                act_sim_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **sim_ac,
                                    **{'random_seed':seed}, **exp_desc})
    
                role_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **{'accuracy': role_accuracy},
                                    **{'random_seed':seed}, **exp_desc})
                role_sim_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **sim_rl,
                                    **{'random_seed':seed}, **exp_desc})
                time_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **{'mae': mae_times},
                                    **{'random_seed':seed}, **exp_desc})
    
                save_results(act_measures, 'activity', 'next_event', is_single_exec, parameters)
                save_results(act_sim_measures, 'activity', 'suffix', is_single_exec, parameters)
                save_results(role_measures, 'roles', 'next_event', is_single_exec, parameters)
                save_results(role_sim_measures, 'roles', 'suffix', is_single_exec, parameters)
                save_results(time_measures, 'times', 'suffix', is_single_exec, parameters)
            elif parameters['model_type'] in ['shared_cat', 'shared_cat_inter']:
                seed = time.time()
                random.seed(seed)
                results = predict(model, examples, var['imp'], parameters, 100)
                sim_ac = dl_measure(results['ac_suff_pred'], ac_alias)
                sim_rl = dl_measure(results['rl_suff_pred'], rl_alias)
                mae_times = mae_measure(results['rem_time_pred'], parameters)
                sim_ac = calculate_by_size(sim_ac)
                sim_rl = calculate_by_size(sim_rl)
                # Save results
                act_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **sim_ac,
                                    **{'random_seed':seed}, **exp_desc})
                role_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **sim_rl,
                                    **{'random_seed':seed}, **exp_desc})
                time_measures.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                            implementation=var['imp']), **{'mae': mae_times},
                                    **{'random_seed':seed}, **exp_desc})
                save_results(act_measures, 'activity', 'suffix', is_single_exec, parameters)
                save_results(role_measures, 'roles', 'suffix', is_single_exec, parameters)
                save_results(time_measures, 'times', 'suffix', is_single_exec, parameters)


def save_results(measurements, feature, measure_name, is_single_exec, parameters):    
    output_route = os.path.join('output_files', parameters['folder'])
    model_name, _ = os.path.splitext(parameters['model_file'])
    if measurements:    
        if is_single_exec:
                sup.create_csv_file_header(measurements,
                                           os.path.join(output_route, model_name +'_'+
                                                        feature+'_' + measure_name + '.csv'))
        else:
            if os.path.exists(os.path.join('output_files', feature+'_suffix.csv')):
                sup.create_csv_file(measurements, os.path.join('output_files', feature+'_' + 
                                                               measure_name + '.csv'), mode='a')
            else:
                sup.create_csv_file_header(measurements, os.path.join('output_files', feature+'_' + 
                                                               measure_name + '.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def predict(model, examples, imp, parameters, max_trace_size):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    results = {'ac_suff_pred':list(), 'rl_suff_pred':list(), 'rem_time_pred':list()}
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

        # Times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
                np.zeros(parameters['dim']['time_dim']),
                np.array(examples['prefixes']['times'][i]),
                axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], 1))])
        if parameters['model_type'] == 'shared_cat':
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        elif parameters['model_type'] == 'shared_cat_inter':
            inter_attr_num = examples['prefixes']['inter_attr'][i].shape[1]
            x_inter_ngram = np.array([np.append(
                    np.zeros((parameters['dim']['time_dim'], inter_attr_num)),
                    examples['prefixes']['inter_attr'][i],
                    axis=0)[-parameters['dim']['time_dim']:].reshape((parameters['dim']['time_dim'], inter_attr_num))])
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
        
        pref_size = len(examples['prefixes']['activities'][i])
        acum_dur = 0
        ac_suf, rl_suf = list(), list()
        for _  in range(1, max_trace_size):
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
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
            x_t_ngram = np.append(x_t_ngram, [predictions[2]], axis=1)
            x_t_ngram = np.delete(x_t_ngram, 0, 1)
            if parameters['model_type'] == 'shared_cat':
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif parameters['model_type'] == 'shared_cat_inter':
                x_inter_ngram = np.append(x_inter_ngram, [predictions[3]], axis=1)
                x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)
            if parameters['norm_method'] == 'lognorm':
                acum_dur += np.expm1(predictions[2][0][0] * parameters['max_dur'])
            else:
                acum_dur += np.rint(predictions[2][0][0] * parameters['max_dur'])
            if parameters['index_ac'][pos] == 'end':
                break
        results['ac_suff_pred'].append({'predicted': ac_suf,
                                        'expected': examples['suffixes']['activities'][i],
                                        'pref_size': pref_size})
        results['rl_suff_pred'].append({'predicted': rl_suf,
                                        'expected': examples['suffixes']['roles'][i],
                                        'pref_size': pref_size})
        results['rem_time_pred'].append({'predicted': acum_dur,
                                         'expected': examples['suffixes']['times'][i],
                                         'pref_size': pref_size})
    sup.print_done_task()
    return results

def predict_seq2seq(model, examples, imp, parameters):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    results = {'ac_true':list(), 'rl_true':list(), 'ac_suff_pred':list(),
               'rl_suff_pred':list(), 'rem_time_pred':list()}
    for i in range(0, len(examples['prefixes']['activities'])):
        act_prefix = examples['prefixes']['activities'][i].reshape(
                (1, examples['prefixes']['activities'][i].shape[0]))
        rl_prefix = examples['prefixes']['roles'][i].reshape(
                (1, examples['prefixes']['roles'][i].shape[0]))
        times_prefix = examples['prefixes']['times'][i].reshape(
                (1, examples['prefixes']['times'][i].shape[0],
                 examples['prefixes']['times'][i].shape[1]))
        if parameters['model_type'] == 'seq2seq':
            inputs = [act_prefix, rl_prefix, times_prefix]
        elif parameters['model_type'] == 'seq2seq_inter':
            inter_prefix = examples['prefixes']['inter_attr'][i].reshape(
                    (1, examples['prefixes']['inter_attr'][i].shape[0],
                     examples['prefixes']['inter_attr'][i].shape[1]))
            inputs = [act_prefix, rl_prefix, times_prefix, inter_prefix]
            
        pref_size = len([x for x in examples['prefixes']['activities'][i][1:] if x > 0])
        predictions = model.predict(inputs)
        if imp == 'Random Choice':
            # Use this to get a random choice following as PDF the predictions
            act_pred = [np.random.choice(np.arange(0, len(x)), p=x) for x in predictions[0][0]]
            rl_pred = [np.random.choice(np.arange(0, len(x)), p=x) for x in predictions[1][0]]
        elif imp == 'Arg Max':
            # Use this to get the max prediction
            act_pred = [np.argmax(x) for x in predictions[0][0]]
            rl_pred = [np.argmax(x) for x in predictions[1][0]]
        # Activities accuracy evaluation
        if act_pred[0] == examples['suffixes']['activities'][i][0]:
            results['ac_true'].append(1)
        else:
            results['ac_true'].append(0)
        # Roles accuracy evaluation
        if rl_pred[0] == examples['suffixes']['roles'][i][0]:
            results['rl_true'].append(1)
        else:
            results['rl_true'].append(0)
        # Activities suffixes
        idx = define_pred_index(act_pred, parameters)
        act_pred = act_pred[:idx]
        rl_pred = rl_pred[:idx]
        time_pred = predictions[2][0][:idx]
        if parameters['norm_method'] == 'lognorm':
            time_pred = np.sum(np.expm1(
                    np.multiply(time_pred, parameters['max_dur'])))
        else:
            time_pred = np.sum(np.rint(
                    np.multiply(time_pred, parameters['max_dur'])))

        time_expected = 0
        if parameters['norm_method'] == 'lognorm':
            time_expected = np.sum(np.expm1(np.multiply(
                    examples['suffixes']['times'][i], parameters['max_dur'])))
        else:
            time_expected = np.sum(np.rint(np.multiply(
                    examples['suffixes']['times'][i], parameters['max_dur'])))
        # Append results
        results['ac_suff_pred'].append({'predicted': act_pred,
                                        'expected': examples['suffixes']['activities'][i],
                                        'pref_size': pref_size})
        results['rl_suff_pred'].append({'predicted': rl_pred,
                                        'expected': examples['suffixes']['roles'][i],
                                        'pref_size': pref_size})
        results['rem_time_pred'].append({'predicted': time_pred,
                                        'expected': time_expected,
                                        'pref_size': pref_size})
    sup.print_done_task() 
    return results

# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test,ac_index, rl_index, parameters):
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
    examples = {'prefixes':dict(), 'suffixes':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    for i, _ in enumerate(df_test):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_test[i][x])):
                serie.append(df_test[i][x][:idx])
                y_serie.append(df_test[i][x][idx:])
            examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
            examples['suffixes'][equi[x]] = examples['suffixes'][equi[x]] + y_serie if i > 0 else y_serie
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
    examples = {'prefixes':dict(), 'suffixes':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    x_inter_dict, y_inter_dict = dict(), dict()
    for i, _ in enumerate(df_test):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_test[i][x])):
                serie.append(df_test[i][x][:idx])
                y_serie.append(df_test[i][x][idx:])
            if x in list(equi.keys()): 
                examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
                examples['suffixes'][equi[x]] = examples['suffixes'][equi[x]] + y_serie if i > 0 else y_serie
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
    examples['suffixes']['inter_attr'] = list()
    y_inter_dict = pd.DataFrame(y_inter_dict)    
    for row in y_inter_dict.values:
        new_row = [np.array(x) for x in row]
        new_row = np.dstack(new_row)
        new_row =  new_row.reshape((new_row.shape[1], new_row.shape[2]))
        examples['suffixes']['inter_attr'].append(new_row)
    return examples

def create_pref_suf_seq2seq(df_test, ac_index, rl_index, parameters):
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
    df_train = reformat_events(df_test, ac_index, rl_index, columns, parameters)
    max_length = parameters['dim']['time_dim']
    examples = {'prefixes':dict(), 'suffixes':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    for i, _ in enumerate(df_train):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_train[i][x])):
                serie.append([0]*(max_length - idx) + df_train[i][x][:idx])
                y_serie.append(df_train[i][x][idx:])
            examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
            examples['suffixes'][equi[x]] = examples['suffixes'][equi[x]] + y_serie if i > 0 else y_serie
    for value in equi.values():
        examples['prefixes'][value]= np.array(examples['prefixes'][value])
    # Reshape times
    examples['prefixes']['times'] = examples['prefixes']['times'].reshape(
            (examples['prefixes']['times'].shape[0],
             examples['prefixes']['times'].shape[1], 1))
    return examples

def create_pref_suf_seq2seq_inter(df_test, ac_index, rl_index, parameters):
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
    df_train = reformat_events(df_test, ac_index, rl_index, columns, parameters)
    max_length = parameters['dim']['time_dim']
    examples = {'prefixes':dict(), 'suffixes':dict()}
    # n-gram definition
    equi = {'ac_index':'activities', 'rl_index':'roles', 'dur_norm':'times'}
    x_inter_dict = dict()
    for i, _ in enumerate(df_train):
        for x in columns:
            serie, y_serie = list(), list() 
            for idx in range(1, len(df_train[i][x])):
                serie.append([0]*(max_length - idx) + df_train[i][x][:idx])
                y_serie.append(df_train[i][x][idx:])
            if x in list(equi.keys()): 
                examples['prefixes'][equi[x]] = examples['prefixes'][equi[x]] + serie if i > 0 else serie
                examples['suffixes'][equi[x]] = examples['suffixes'][equi[x]] + y_serie if i > 0 else y_serie
            else:
                x_inter_dict[x] = x_inter_dict[x] + serie if i > 0 else serie
    for value in equi.values():
        examples['prefixes'][value]= np.array(examples['prefixes'][value])
    # Reshape times
    examples['prefixes']['times'] = examples['prefixes']['times'].reshape(
            (examples['prefixes']['times'].shape[0],
             examples['prefixes']['times'].shape[1], 1))
    # Reshape intercase attributes (prefixes, n-gram size, number of attributes) 
    for key, value in x_inter_dict.items():
        x_inter_dict[key] = np.array(value)
        x_inter_dict[key] = x_inter_dict[key].reshape((x_inter_dict[key].shape[0],
                   x_inter_dict[key].shape[1], 1))        
    examples['prefixes']['inter_attr'] = np.dstack(list(x_inter_dict.values()))
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
                serie.append(ac_index[('end')])
            elif x == 'rl_index':
                serie.append(rl_index[('end')])
            else:
                serie.append(0)
            temp_dict = {**{x: serie},**temp_dict}
        temp_dict = {**{'caseid': key},**temp_dict}
        temp_data.append(temp_dict)
    return temp_data

def dl_measure(prefixes, alias):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
    Returns:
        list: list with measures added.
    """
    results = list()
    for prefix in prefixes:
        suff_log = str([alias[x] for x in prefix['expected']])
        suff_pred = str([alias[x] for x in prefix['predicted']])

        length = np.max([len(suff_log), len(suff_pred)])
        sim = jf.damerau_levenshtein_distance(suff_log,
                                              suff_pred)
        sim = (1-(sim/length))
        results.append({'pref_size': prefix['pref_size'], 'dl': sim})
    return results

def mae_measure(prefixes, parameters):
    """Absolute Error measurement.
    Args:
        prefixes (list): list with predicted remaining-times and expected ones.
    Returns:
        list: list with measures added.
    """
    results = list()
    for prefix in prefixes:
        if parameters['norm_method'] == 'lognorm':
            expected = np.expm1(np.array(prefix['expected']) * parameters['max_dur'])
        else:
            expected = np.rint(np.array(prefix['expected']) * parameters['max_dur'])
        expected = np.sum(expected)
        results.append(abs(expected - prefix['predicted']))        
    return np.mean(results)


def create_alias(quantity):
    """Creates char aliases for a categorical attributes.
    Args:
        quantity (int): number of aliases to create.
    Returns:
        dict: alias for a categorical attributes.
    """
    characters = [chr(i) for i in range(0, quantity)]
    aliases = random.sample(characters, quantity)
    alias = dict()
    for i in range(0, quantity):
        alias[i] = aliases[i]
    return alias

def calculate_by_size(data):
    df = pd.DataFrame.from_dict(data)
    df = df.groupby('pref_size', as_index=False).agg({'dl': 'mean'})
    df = df.reset_index().set_index('pref_size').drop('index', axis=1).transpose()
    df['avg'] = df.mean(numeric_only=True, axis=1)
    return df.to_dict('records')[0]

def define_pred_index(act_pred, parameters):
    index = len(act_pred)
    for x in act_pred[::-1]: 
        if x == 0:
            index -= 1
        else:
            break
    idx = 0
    for x in act_pred[:index]: 
        if parameters['index_ac'][x] == 'end':
            idx += 1
            break
        else:
            idx += 1
    return idx
