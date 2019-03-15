# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:36:01 2018

@author: Manuel Camargo
"""
import json
import datetime
import os

from keras.models import load_model

import pandas as pd
import numpy as np
import math

from support_modules.analyzers import generalization as gen

from support_modules import  support as sup


from datetime import timedelta

START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()

def predict(st, folder, model_file, is_single_exec=True):
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
#        ac_index = {(v[0], v[1]): int(k) for k, v in data['index_ac'].items()}
        f.close()

    num_cases = len(df_test.caseid.unique())
    variants = [{'imp': 'Random Choice', 'rep': 15},
                       {'imp': 'Arg Max', 'rep': 1}]
    max_trace_size = 200
    # Generation of predictions
    model = load_model(os.path.join(output_route, model_file))
    df_test_log = df_test.to_dict('records')

    for var in variants:
        imp = var['imp']
        for r in range(0, var['rep']):
            generated_event_log = generate_traces(model, imp, num_cases, max_trace_size)
            
            sim_task = gen.gen_mesurement(df_test_log, generated_event_log, 'task')
            sim_role = gen.gen_mesurement(df_test_log, generated_event_log, 'user')
        
            if is_single_exec:
                sup.create_csv_file_header(sim_task, os.path.join(output_route, model_name +'_similarity.csv'))
                sup.create_csv_file_header(generated_event_log,
                                           os.path.join(output_route, model_name +'_log.csv'))
    
            dl_task = np.mean([x['sim_score'] for x in sim_task])
            dl_user = np.mean([x['sim_score'] for x in sim_role])
            dl_t = np.mean([x['sim_score_t'] for x in sim_task])
            mae = np.mean([x['abs_err'] for x in sim_task])
        
            print('Demerau-Levinstein task distance:', dl_task, sep=' ')
            print('Demerau-Levinstein role distance:', dl_user, sep=' ')
            print('Demerau-Levinstein task penalized:', dl_t, sep=' ')
            print('MAE:', mae, sep=' ')
            # Save results
            measurements=list()
            measurements.append({**dict(model= os.path.join(output_route, model_file),
                                        implementation = imp, dl_task=dl_task, 
                                        dl_user=dl_user, mae=mae, dlt=dl_t), **EXP})
            if is_single_exec: 
                sup.create_csv_file_header(measurements, os.path.join('output_files', model_name +'_measures.csv'))
            else:
                if os.path.exists(os.path.join('output_files', 'total_measures.csv')):
                    sup.create_csv_file(measurements, os.path.join('output_files', 'total_measures.csv'), mode='a')
                else:
                    sup.create_csv_file_header(measurements, os.path.join('output_files', 'total_measures.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def generate_traces(model, imp, num_cases, max_trace_size):
    generated_event_log = list()
    for case in range(0, num_cases):
        X_trace = list()
        X_ac_ngram = np.zeros((1, DIM['time_dim']), dtype=np.float32)
        X_rl_ngram = np.zeros((1, DIM['time_dim']), dtype=np.float32)
        X_t_ngram = np.zeros((1, DIM['time_dim'], 1), dtype=np.float32)
        
        #X_ngram[0, DIM['time_dim'] - 1] = 0
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
#            # Add prediction to n_gram
            X_ac_ngram = np.append(X_ac_ngram, [[pos]], axis=1)
            X_ac_ngram = np.delete(X_ac_ngram, 0, 1)
            X_rl_ngram = np.append(X_rl_ngram, [[pos1]], axis=1)
            X_rl_ngram = np.delete(X_rl_ngram, 0, 1)
            X_t_ngram = np.append(X_t_ngram, [y[2]], axis=1)
            X_t_ngram = np.delete(X_t_ngram, 0, 1)

#            # Stop if the next prediction is the end of the trace
#            # otherwise until the defined max_size
            if INDEX_AC[pos] == 'end':
                break
        generated_event_log.extend(decode_trace(X_trace, case))
        # sup.print_progress((((case+1) / num_cases)* 100), 'Generating process traces ')
    sup.print_done_task()
    return generated_event_log

# =============================================================================
# Decoding
# =============================================================================

def decode_trace(trace, case):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    log_trace = list()
    for i, _ in enumerate(trace):
        event = trace[i]
        if INDEX_AC[event[0]] != 'end':
            if EXP['norm_method'] == 'activity':
                tbtw = (event[2] * TBTW['max_tbtw'][event[0]])
            elif EXP['norm_method'] == 'lognorm':
                tbtw = math.expm1(event[2] * TBTW['max_tbtw'])
            else:
                tbtw = np.rint(event[2] * TBTW['max_tbtw'])
            if i == 0:
                now = datetime.datetime.now()
                now.strftime(START_TIMEFORMAT)
                time = now
            else:
                time = log_trace[i-1]['start_timestamp'] + timedelta(seconds=tbtw) 
            log_trace.append(dict(caseid=case,
                                  task=INDEX_AC[event[0]],
                                  user=INDEX_RL[event[1]],
                                  start_timestamp=time,
                                  tbtw_raw=event[2],
                                  tbtw=tbtw))
    return log_trace


def calculate_relative_times(df):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    df['tbtw'] = 0
    # Multitasking time
    cases = df.caseid.unique()
    for case in cases:
        trace = df[df.caseid == case].sort_values('start_timestamp', ascending=True)
        for i in range(1, len(trace)):
            row_num = trace.iloc[i].name
            tbtw = (trace.iloc[i].end_timestamp - trace.iloc[i - 1].end_timestamp).seconds
            df.iat[int(row_num), int(df.columns.get_loc('tbtw'))] = tbtw
    return df
