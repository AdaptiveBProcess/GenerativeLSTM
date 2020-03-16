# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:36:01 2018

@author: Manuel Camargo
"""
import json
import os
import math
import datetime
from datetime import timedelta

from keras.models import load_model

import pandas as pd
import numpy as np

from support_modules.analyzers import generalization as gen
from support_modules import support as sup

START_TIMEFORMAT = ''
INDEX_AC = None
INDEX_RL = None
DIM = dict()
TBTW = dict()
EXP = dict()

def predict(timeformat, parameters, is_single_exec=True):
    """Main function of the event log generation module.
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

#   Next event selection method and numbers of repetitions
    variants = [{'imp': 'Random Choice', 'rep': 1},
                {'imp': 'Arg Max', 'rep': 0}]
#   Generation of predictions
    model = load_model(os.path.join(output_route, parameters['model_file']))
    df_test_log = df_test.to_dict('records')

    for var in variants:
        for _ in range(0, var['rep']):
            generated_event_log = generate_traces(model, var['imp'],
                                                  len(df_test.caseid.unique()),
                                                  200)
            sim_task = gen.gen_mesurement(df_test_log,
                                          generated_event_log, 'task')
            sim_role = gen.gen_mesurement(df_test_log,
                                          generated_event_log, 'user')
            if is_single_exec:
                sup.create_csv_file_header(sim_task,
                                           os.path.join(output_route,
                                                        model_name +'_similarity.csv'))
                sup.create_csv_file_header(generated_event_log,
                                           os.path.join(output_route, model_name +'_log.csv'))

            # Save results
            measurements = list()
            measurements.append({**dict(model=os.path.join(output_route, parameters['model_file']),
                                        implementation=var['imp'],
                                        dl_task=np.mean([x['sim_score'] for x in sim_task]),
                                        dl_user=np.mean([x['sim_score'] for x in sim_role]),
                                        mae=np.mean([x['abs_err'] for x in sim_task]),
                                        dlt=np.mean([x['sim_score_t'] for x in sim_task])),
                                 **EXP})
            if is_single_exec:
                sup.create_csv_file_header(measurements,
                                           os.path.join('output_files',
                                                        model_name +'_measures.csv'))
            else:
                if os.path.exists(os.path.join('output_files', 'total_measures.csv')):
                    sup.create_csv_file(measurements,
                                        os.path.join('output_files',
                                                     'total_measures.csv'), mode='a')
                else:
                    sup.create_csv_file_header(measurements,
                                               os.path.join('output_files', 'total_measures.csv'))

# =============================================================================
# Predic traces
# =============================================================================

def generate_traces(model, imp, num_cases, max_trace_size):
    """Generate business process traces using a keras trained model.
    Args:
        model (keras model): keras trained model.
        imp (str): method of next event selection.
        num_cases (int): number of traces to generate.
        max_trace_size (int): max size of the trace
    """
    sup.print_performed_task('Generating traces')
    generated_event_log = list()
    for case in range(0, num_cases):
        x_trace = list()
        x_ac_ngram = np.zeros((1, DIM['time_dim']), dtype=np.float32)
        x_rl_ngram = np.zeros((1, DIM['time_dim']), dtype=np.float32)
        x_t_ngram = np.zeros((1, DIM['time_dim'], 1), dtype=np.float32)

        for _ in range(1, max_trace_size):
            predictions = model.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
            if imp == 'Random Choice':
                # Use this to get a random choice following as PDF the predictions
                pos = np.random.choice(np.arange(0, len(predictions[0][0])), p=predictions[0][0])
                pos1 = np.random.choice(np.arange(0, len(predictions[1][0])), p=predictions[1][0])
            elif imp == 'Arg Max':
                # Use this to get the max prediction
                pos = np.argmax(predictions[0][0])
                pos1 = np.argmax(predictions[1][0])
            x_trace.append([pos, pos1, predictions[2][0][0]])
#            # Add prediction to n_gram
            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
            x_t_ngram = np.append(x_t_ngram, [predictions[2]], axis=1)
            x_t_ngram = np.delete(x_t_ngram, 0, 1)

#            # Stop if the next prediction is the end of the trace
#            # otherwise until the defined max_size
            if INDEX_AC[pos] == 'end':
                break
        generated_event_log.extend(decode_trace(x_trace, case))
    sup.print_done_task()
    return generated_event_log

# =============================================================================
# Decoding
# =============================================================================

def decode_trace(trace, case):
    """Example function with types documented in the docstring.
    Args:
        trace (list): trace of predicted events.
        case (int): case number.
    Returns:
        list: predicted business trace decoded.
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
