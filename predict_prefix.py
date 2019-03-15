# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:16:15 2019

@author: Manuel Camargo
"""
import json
import os

from keras.models import load_model

import pandas as pd
import numpy as np
import math

import jellyfish as jf
import support as sup

import random

START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()

def predict_prefix(st, folder, model_file, is_single_exec=True):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    global START_TIMEFORMAT
    global INDEX_AC
    global INDEX_RL
    global DIM
    global TBTW
    global EXP

    START_TIMEFORMAT = st

    imp = 'Arg Max'
    max_trace_size = 200

    output_route = os.path.join('output_files', folder)
    model_name, _ = os.path.splitext(model_file)
    # Loading of testing dataframe
    df_test = pd.read_csv(os.path.join(output_route, 'parameters', 'test_log.csv'))
    df_test['start_timestamp'] = pd.to_datetime(df_test['start_timestamp'])
    df_test['end_timestamp'] = pd.to_datetime(df_test['end_timestamp'])
    df_test = df_test.drop(columns=['user'])
    df_test = df_test.rename(index=str, columns={"role": "user"})

    # Loading of parameters from training
    with open(os.path.join(output_route, 'parameters', 'model_parameters.json')) as f:
        data = json.load(f)
        EXP = {k: v for k, v in data['exp_desc'].items()}
        print(EXP)
        DIM['samples'] = int(data['dim']['samples'])
        DIM['time_dim'] = int(data['dim']['time_dim'])
        DIM['features'] = int(data['dim']['features'])
        if EXP['norm_method'] == 'activity':
            TBTW['max_tbtw'] = {int(k): int(v) for k, v in data['max_tbtw'].items()}
        else:
            TBTW['max_tbtw'] = float(data['max_tbtw'])
        INDEX_AC = {int(k): v for k, v in data['index_ac'].items()}
        INDEX_RL = {int(k): v for k, v in data['index_rl'].items()}
        f.close()


    if EXP['norm_method'] == 'max':
        max_tbtw = np.max(df_test.tbtw)
        norm = lambda x: x['tbtw']/max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm,axis=1)
    elif EXP['norm_method'] == 'lognorm':
        logit = lambda x: math.log1p(x['tbtw'])
        df_test['tbtw_log'] = df_test.apply(logit,axis=1)
        max_tbtw = np.max(df_test.tbtw_log)
        norm = lambda x: x['tbtw_log']/max_tbtw
        df_test['tbtw_norm'] = df_test.apply(norm,axis=1)        

    ac_alias = create_alias(len(INDEX_AC))
    rl_alias = create_alias(len(INDEX_RL))
    
    args = dict(df_test=df_test, ac_alias=ac_alias,rl_alias=rl_alias,
               output_route=output_route,model_file=model_file,
               imp=imp,max_trace_size=max_trace_size)
    rep = 1
    for r in range(0, rep):    
        results = execute_experiments([2, 10, 20], args)
        # Save results
        measurements=list()
        measurements.append({**dict(model= os.path.join(output_route, model_file),
                                    implementation = imp), **results,
                                    **EXP})
        if is_single_exec: 
            sup.create_csv_file_header(measurements, os.path.join('output_files', model_name +'_sufix.csv'))
        else:
            if os.path.exists(os.path.join('output_files', 'sufix_measures.csv')):
                sup.create_csv_file(measurements, os.path.join('output_files', 'sufix_measures.csv'), mode='a')
            else:
                sup.create_csv_file_header(measurements, os.path.join('output_files', 'sufix_measures.csv'))


# =============================================================================
# Predic traces
# =============================================================================

def predict(model, prefixes, ac_alias, rl_alias, imp, max_trace_size):
    # Generation of predictions
    for prefix in prefixes:
        X_trace = list()
        X_ac_ngram = np.array([prefix['ac_pref']])
        X_rl_ngram = np.array([prefix['rl_pref']])
        X_t_ngram = np.array([prefix['t_pref']])
        
        acum_tbtw = 0           
        ac_suf, rl_suf = '', ''
        for i  in range(1, max_trace_size):
            y = model.predict([X_ac_ngram, X_rl_ngram, X_t_ngram])
            if imp == 'Random Choice':
                # Use this to get a random choice following as PDF the predictions
                pos = np.random.choice(np.arange(0, len(y[0][0])), p=y[0][0])
                pos1 = np.random.choice(np.arange(0, len(y[1][0])), p=y[1][0])
               
            elif imp == 'Arg Max':
                # Use this to get the max prediction
                pos = np.argmax(y[0][0])
                pos1 = np.argmax(y[1][0])
            X_trace.append([pos, pos1, y[2][0][0]])
           # Add prediction to n_gram
            X_ac_ngram = np.append(X_ac_ngram, [[pos]], axis=1)
            X_ac_ngram = np.delete(X_ac_ngram, 0, 1)
            X_rl_ngram = np.append(X_rl_ngram, [[pos1]], axis=1)
            X_rl_ngram = np.delete(X_rl_ngram, 0, 1)
            X_t_ngram = np.append(X_t_ngram, [y[2]], axis=1)
            X_t_ngram = np.delete(X_t_ngram, 0, 1)
            # Stop if the next prediction is the end of the trace
            # otherwise until the defined max_size
            if INDEX_AC[pos] == 'end':
                break
            else:
                ac_suf += ac_alias[pos]
                rl_suf += rl_alias[pos1]
                if EXP['norm_method'] == 'lognorm':
                    acum_tbtw += math.expm1(y[2][0][0] * TBTW['max_tbtw'])
                else:
                    acum_tbtw += np.rint(y[2][0][0] * TBTW['max_tbtw'])
                
        prefix['ac_suf_pred'] = ac_suf
        prefix['rl_suf_pred'] = rl_suf
        prefix['rem_time_pred'] = acum_tbtw
        # sup.print_progress((((case+1) / num_cases)* 100), 'Generating process traces ')
        # case += 1
    sup.print_done_task()
    return prefixes
    
# =============================================================================
# Generate experiments
# =============================================================================
def execute_experiments(prefix_sizes, args):
    results = dict()
    model = load_model(os.path.join(args['output_route'], args['model_file']))
    for size in prefix_sizes:
        prefixes = create_pref_suf(args['df_test'], args['ac_alias'], args['rl_alias'], size)
        prefixes = predict(model, prefixes, args['ac_alias'], 
                           args['rl_alias'], args['imp'], args['max_trace_size'])
        prefixes = dl_measure(prefixes, 'ac')
        prefixes = dl_measure(prefixes, 'rl')
        prefixes = ae_measure(prefixes)
        # Calculate metrics
        dl_task = np.mean([x['ac_dl'] for x in prefixes])
        dl_user = np.mean([x['rl_dl'] for x in prefixes])
        mae = np.mean([x['ae'] for x in prefixes])
        # Print reresults
        print('DL task distance pref '+ str(size) +':', dl_task, sep=' ')
        print('DL role distance pref '+ str(size) +':', dl_user, sep=' ')
        print('MAE pref '+ str(size) +':', mae, sep=' ')
        print('----------')
        results['dl_ac_'+str(size)] = dl_task
        results['dl_rl_'+str(size)] = dl_user
        results['mae_'+str(size)] = mae
    return results

# =============================================================================
# Reformat
# =============================================================================
def create_pref_suf(df_test, ac_alias, rl_alias, pref_size):
    splits = list()
    cases = df_test.caseid.unique()
    for case in cases:   
        traces = df_test[df_test.caseid==case]
        if len(traces) > pref_size:
            dictionary = dict(ac_pref=list(), rl_pref=list(), t_pref=list(), 
                              ac_suf='', rl_suf='', rem_time=0)
            for i in range(0, len(traces)):
                if i < pref_size:
                    dictionary['ac_pref'].append(traces.iloc[i]['ac_index'])
                    dictionary['rl_pref'].append(traces.iloc[i]['rl_index'])
                    dictionary['t_pref'].append(traces.iloc[i]['tbtw_norm'])
                else:
                    dictionary['ac_suf'] += ac_alias[traces.iloc[i]['ac_index']]
                    dictionary['rl_suf'] += rl_alias[traces.iloc[i]['rl_index']]
                    dictionary['rem_time'] += traces.iloc[i]['tbtw']

            x_ngram = np.zeros((1, (DIM['time_dim'])), dtype=np.float32)
            dictionary['ac_pref'] = np.append( x_ngram, dictionary['ac_pref'])
            dictionary['rl_pref'] = np.append( x_ngram, dictionary['rl_pref'])
            dictionary['t_pref'] = np.append( x_ngram, dictionary['t_pref'])
            index = [ x for x in range(0, ((DIM['time_dim'] + pref_size) - DIM['time_dim']))]
            dictionary['ac_pref'] = np.delete(dictionary['ac_pref'], index, 0)
            dictionary['rl_pref'] = np.delete(dictionary['rl_pref'], index, 0)
            dictionary['t_pref'] = np.delete(dictionary['t_pref'], index, 0)
            dictionary['t_pref'] = dictionary['t_pref'].reshape((dictionary['t_pref'].shape[0], 1))
    
            splits.append(dictionary)
    return splits

def create_alias(quantity):
    characters = [chr(i) for i in range(0, quantity)]
    aliases = random.sample(characters, quantity)
    alias = dict()
    for i in range(0, quantity):
        alias[i] = aliases[i]
    return alias

def dl_measure(prefixes, feature):
    for prefix in prefixes:
        length=np.max([len(prefix[feature + '_suf']), len(prefix[feature + '_suf_pred'])])
        sim = jf.damerau_levenshtein_distance(prefix[feature + '_suf'], 
                                              prefix[feature + '_suf_pred'])
        sim = (1-(sim/length))
        prefix[feature + '_dl'] = sim
    return prefixes

def ae_measure(prefixes):
    for prefix in prefixes:
        prefix['ae'] = abs(prefix['rem_time'] - prefix['rem_time_pred'])
    return prefixes