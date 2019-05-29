# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import os
import math
import random

from keras.models import load_model

import pandas as pd
import numpy as np

import jellyfish as jf
from support_modules import support as sup


START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()

def predict_suffix_full(timeformat, parameters, is_single_exec=True):
    """Main function of the suffix prediction module.
    Args:
        timeformat (str): event-log date-time format.
        parameters (dict): parameters used in the training step.
        is_single_exec (boolean): generate measurments stand alone or share
                    results with other runing experiments (optional)
    """
    global START_TIMEFORMAT
    global INDEX_AC
    global INDEX_RL
    global DIM
    global TBTW
    global EXP

    START_TIMEFORMAT = timeformat

    output_route = os.path.join('output_files', parameters['folder'])

    # Loading of testing dataframe
    df_test = pd.read_csv(os.path.join(output_route, 'parameters', 'test_log.csv'))
    df_test['start_timestamp'] = pd.to_datetime(df_test['start_timestamp'])
    df_test['end_timestamp'] = pd.to_datetime(df_test['end_timestamp'])
    df_test = df_test.drop(columns=['user'])
    df_test = df_test.rename(index=str, columns={"role": "user"})

    # Loading of parameters from training
    with open(os.path.join(output_route, 'parameters', 'model_parameters.json')) as file:
        data = json.load(file)
        EXP = {k: v for k, v in data['exp_desc'].items()}
        print(EXP)
        DIM['samples'] = int(data['dim']['samples'])
        DIM['time_dim'] = int(data['dim']['time_dim'])
        DIM['features'] = int(data['dim']['features'])
        TBTW['max_tbtw'] = float(data['max_tbtw'])
        INDEX_AC = {int(k): v for k, v in data['index_ac'].items()}
        INDEX_RL = {int(k): v for k, v in data['index_rl'].items()}
        file.close()

    if EXP['norm_method'] == 'max':
        max_tbtw = np.max(df_test.tbtw)
        norm = lambda x: x['tbtw']/max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)
    elif EXP['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        df_test['tbtw_log'] = df_test.apply(logit, axis=1)
        max_tbtw = np.max(df_test.tbtw_log)
        norm = lambda x: x['tbtw_log']/max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm, axis=1)

    ac_index = {v: int(k) for k, v in data['index_ac'].items()}
    rl_index = {v: int(k) for k, v in data['index_rl'].items()}

    ac_alias = create_alias(len(INDEX_AC))
    rl_alias = create_alias(len(INDEX_RL))

#   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 10},
                {'imp': 'Arg Max', 'rep': 1}]
#   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))

    for var in variants:
        measurements_ac = list()
        measurements_rl = list()
        measurements_mae = list()

        for i in range(0, var['rep']):
            prefixes = create_pref_suf(df_test, ac_index, rl_index)
            prefixes = predict(model, prefixes, var['imp'], 100)
            prefixes = dl_measure(prefixes, 'ac', ac_alias)
            prefixes = dl_measure(prefixes, 'rl', rl_alias)
            prefixes = ae_measure(prefixes)
            prefixes = pd.DataFrame.from_dict(prefixes)
            prefixes = prefixes.groupby('pref_size', as_index=False).agg({'ac_dl': 'mean','rl_dl': 'mean', 'ae': 'mean'})
            measure_ac = dict()
            measure_rl = dict()
            measure_rem = dict()
            for size in prefixes.pref_size.unique():
                measure_ac[size] = prefixes[prefixes.pref_size==size].ac_dl.iloc[0]
                measure_rl[size] = prefixes[prefixes.pref_size==size].rl_dl.iloc[0]
                measure_rem[size] = prefixes[prefixes.pref_size==size].ae.iloc[0]
            measure_ac['avg'] = prefixes.ac_dl.mean()
            measure_rl['avg'] = prefixes.rl_dl.mean()
            measure_rem['avg'] = prefixes.ae.mean()
            # Save results
            measurements_ac.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **measure_ac,
                                **EXP})
            measurements_rl.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **measure_rl,
                                **EXP})
            measurements_mae.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **measure_rem,
                                **EXP})
        save_results(measurements_ac, 'ac', is_single_exec, parameters)
        save_results(measurements_rl, 'rl', is_single_exec, parameters)
        save_results(measurements_mae, 'mae', is_single_exec, parameters)
    
def save_results(measurements, feature, is_single_exec, parameters):    
    output_route = os.path.join('output_files', parameters['folder'])
    model_name, _ = os.path.splitext(parameters['model_file'])
    if measurements:    
        if is_single_exec:
                sup.create_csv_file_header(measurements, os.path.join(output_route,
                                                                      model_name +'_'+feature+'_full_suff.csv'))
        else:
            if os.path.exists(os.path.join('output_files', 'full_'+feature+'_suffix_measures.csv')):
                sup.create_csv_file(measurements, os.path.join('output_files',
                                                               'full_'+feature+'_suffix_measures.csv'), mode='a')
            else:
                sup.create_csv_file_header(measurements, os.path.join('output_files',
                                                               'full_'+feature+'_suffix_measures.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def predict(model, prefixes, imp, max_trace_size):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        imp (str): method of next event selection.
    """
    # Generation of predictions
    for prefix in prefixes:
        # Activities and roles input shape(1,5)
        x_ac_ngram = np.append(
                np.zeros(DIM['time_dim']),
                np.array(prefix['ac_pref']),
                axis=0)[-DIM['time_dim']:].reshape((1,DIM['time_dim']))
                
        x_rl_ngram = np.append(
                np.zeros(DIM['time_dim']),
                np.array(prefix['rl_pref']),
                axis=0)[-DIM['time_dim']:].reshape((1,DIM['time_dim']))

        # times input shape(1,5,1)
        x_t_ngram = np.array([np.append(
                np.zeros(DIM['time_dim']),
                np.array(prefix['t_pref']),
                axis=0)[-DIM['time_dim']:].reshape((DIM['time_dim'], 1))])
        acum_tbtw = 0
        ac_suf, rl_suf = list(), list()
        for _  in range(1, max_trace_size):
            predictions = model.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
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
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            ac_suf.append(pos)
            rl_suf.append(pos1)
            if EXP['norm_method'] == 'lognorm':
                acum_tbtw += math.expm1(predictions[2][0][0] * TBTW['max_tbtw'])
            else:
                acum_tbtw += np.rint(predictions[2][0][0] * TBTW['max_tbtw'])
            if INDEX_AC[pos] == 'end':
                break
        prefix['ac_suff_pred'] = ac_suf
        prefix['rl_suff_pred'] = rl_suf
        prefix['rem_time_pred'] = acum_tbtw
    sup.print_done_task()
    return prefixes


# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test, ac_index, rl_index):
    """Extraction of prefixes and expected suffixes from event log.
    Args:
        df_test (dataframe): testing dataframe in pandas format.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
        pref_size (int): size of the prefixes to extract.
    Returns:
        list: list of prefixes and expected sufixes.
    """
    prefixes = list()
    cases = df_test.caseid.unique()
    for case in cases:
        trace = df_test[df_test.caseid == case].to_dict('records')
        ac_pref = list()
        rl_pref = list()
        t_pref = list()
        for i in range(0, len(trace)-1):
            ac_pref.append(trace[i]['ac_index'])
            rl_pref.append(trace[i]['rl_index'])
            t_pref.append(trace[i]['tbtw_norm'])
            prefixes.append(dict(ac_pref=ac_pref.copy(),
                                 ac_suff=[x['ac_index'] for x in trace[i + 1:]],
                                 rl_pref=rl_pref.copy(),
                                 rl_suff=[x['rl_index'] for x in trace[i + 1:]],
                                 t_pref=t_pref.copy(),
#                                 rem_time=(trace[-1]['end_timestamp'] - trace[i + 1]['start_timestamp']).total_seconds(),
                                 rem_time=[x['tbtw'] for x in trace[i + 1:]],
                                 pref_size=i + 1))
    for x in prefixes:
        x['ac_suff'].append(ac_index['end'])
        x['rl_suff'].append(rl_index['end'])
        x['rem_time'].append(0)
    return prefixes

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

def dl_measure(prefixes, feature, alias):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
        feature (str): categorical attribute to measure.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        suff_log = str([alias[x] for x in prefix[feature + '_suff']])
        suff_pred = str([alias[x] for x in prefix[feature + '_suff_pred']])
        
        length = np.max([len(suff_log), len(suff_pred)])
        sim = jf.damerau_levenshtein_distance(suff_log,
                                              suff_pred)
        sim = (1-(sim/length))
        prefix[feature + '_dl'] = sim
    return prefixes

def ae_measure(prefixes):
    """Absolute Error measurement.
    Args:
        prefixes (list): list with predicted remaining-times and expected ones.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        rem_log = np.sum(prefix['rem_time'])
#        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
        prefix['ae'] = abs(rem_log - prefix['rem_time_pred'])
    return prefixes
