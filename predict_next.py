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

def predict_next(timeformat, parameters, is_single_exec=True):
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
    model_name, _ = os.path.splitext(parameters['model_file'])
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

    ac_alias = create_alias(len(INDEX_AC))
    rl_alias = create_alias(len(INDEX_RL))

#   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 15},
                {'imp': 'Arg Max', 'rep': 1}]
#   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))

    for var in variants:
        measurements = list()
        for i in range(0, var['rep']):

            prefixes = create_pref_suf(df_test, ac_alias, rl_alias)
            prefixes = predict(model, prefixes, ac_alias, rl_alias, var['imp'])
            
            accuracy = (np.sum([x['ac_true'] for x in prefixes])/len(prefixes))

            if is_single_exec:
                sup.create_csv_file_header(prefixes, os.path.join(output_route,
                                                                      model_name +'_rep_'+str(i)+'_next.csv'))
            
            # Save results
            measurements.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp']), **{'accuracy': accuracy},
                                **EXP})
        if measurements:    
            if is_single_exec:
                    sup.create_csv_file_header(measurements, os.path.join(output_route,
                                                                          model_name +'_next.csv'))
            else:
                if os.path.exists(os.path.join('output_files', 'next_event_measures.csv')):
                    sup.create_csv_file(measurements, os.path.join('output_files',
                                                                   'next_event_measures.csv'), mode='a')
                else:
                    sup.create_csv_file_header(measurements, os.path.join('output_files',
                                                                              'next_event_measures.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def predict(model, prefixes, ac_alias, rl_alias, imp):
    """Generate business process suffixes using a keras trained model.
    Args:
        model (keras model): keras trained model.
        prefixes (list): list of prefixes.
        ac_index (dict): index of activities.
        rl_index (dict): index of roles.
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
        if pos == prefix['ac_next']:
            prefix['ac_true'] = 1
        else:
            prefix['ac_true'] = 0
        # Roles accuracy evaluation
        if pos1 == prefix['rl_next']:
            prefix['rl_true'] = 1
        else:
            prefix['rl_true'] = 0
    sup.print_done_task()
    return prefixes


# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test, ac_alias, rl_alias):
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
        trace = df_test[df_test.caseid == case]
        ac_pref = list()
        rl_pref = list()
        t_pref = list()
        for i in range(0, len(trace)-1):
            ac_pref.append(trace.iloc[i]['ac_index'])
            rl_pref.append(trace.iloc[i]['rl_index'])
            t_pref.append(trace.iloc[i]['tbtw_norm'])
            prefixes.append(dict(ac_pref=ac_pref.copy(),
                                 ac_next=trace.iloc[i + 1]['ac_index'],
                                 rl_pref=rl_pref.copy(),
                                 rl_next=trace.iloc[i + 1]['rl_index'],
                                 t_pref=t_pref.copy()))
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

def dl_measure(prefixes, feature):
    """Demerau-Levinstain distance measurement.
    Args:
        prefixes (list): list with predicted and expected suffixes.
        feature (str): categorical attribute to measure.
    Returns:
        list: list with measures added.
    """
    for prefix in prefixes:
        length = np.max([len(prefix[feature + '_suf']), len(prefix[feature + '_suf_pred'])])
        sim = jf.damerau_levenshtein_distance(prefix[feature + '_suf'],
                                              prefix[feature + '_suf_pred'])
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
        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
    return prefixes
