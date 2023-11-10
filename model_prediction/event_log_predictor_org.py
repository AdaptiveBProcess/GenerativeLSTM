# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:11:17 2020

@author: Manuel Camargo
"""
import numpy as np
import math
from tqdm import tqdm
import time
import traceback
import itertools

import multiprocessing
from multiprocessing import Pool
from tensorflow.keras.models import load_model
import keras.utils as ku

from datetime import timedelta

class EventLogPredictor():

    def __init__(self):
        """constructor"""
        self.imp = 'Arg Max'
        self.max_trace_size = 0

    def predict(self, params, model_path, examples, imp, vectorizer):
        self.model_path = model_path
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        self.params = params
        self.vectorizer = vectorizer
        predictor = self._get_predictor(params['model_type'])
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        if model_type in ['shared_cat_cx', 'concatenated_cx',
                          'shared_cat_gru_cx', 'concatenated_gru_cx']:
            return self._generate_inter_parallel
        else:
            return self._generate_traces_parallel

    def _generate_inter_parallel(self, parms, vectorizer):
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        cpu_count = multiprocessing.cpu_count()
        num_digits = len(str(parms['num_cases']))
        cases = ['Case'+str(num).zfill(num_digits) 
                 for num in range(0, parms['num_cases'])]
        b_size = math.ceil(len(cases)/(cpu_count*2))
        chunks = [cases[x:x+b_size] for x in range(0, len(cases), b_size)]
        reps = len(chunks)
        pool = Pool(processes=cpu_count)
        # Generate
        args = [(cases, self.params, self.model_path, self.vectorizer)
                for cases in chunks]
        p = pool.map_async(self._generate_inter_batch, args)
        pbar_async(p, 'generating traces:')
        pool.close()
        # Save results
        event_log = list(itertools.chain(*p.get()))
        return event_log

    @staticmethod
    def _generate_inter_batch(args):
        def gen(cases, parms, model_path, vectorizer):
            try:
                s_timestamp = parms['start_time']
                model = load_model(model_path)
                new_batch = list()
                for cid in cases:
                    x_trace = list()
                    x_ac_ngram = np.zeros((1, parms['n_size']), 
                                          dtype=np.float32)
                    x_rl_ngram = np.zeros((1, parms['n_size']), 
                                          dtype=np.float32)
                    x_t_ngram = np.zeros((1, parms['n_size'], 2), 
                                         dtype=np.float32)
                    num_feat = len(parms['additional_columns']) 
                    num_feat += (6 if 'weekday' in 
                                 parms['additional_columns'] else 0)
                    x_inter_ngram = np.zeros((1, parms['n_size'], num_feat), 
                                             dtype=np.float32)
                    pos, pos1 = 0, 0
                    pre_times = [[0.0, 0.0]]
                    x_trace = [dict(caseid=cid,
                                    task=parms['index_ac'][pos],
                                    role=parms['index_rl'][pos1],
                                    start_timestamp=s_timestamp,
                                    end_timestamp=s_timestamp)]
                    i = 1
                    while i < parms['max_trace_size']:
                        daytime = s_timestamp.time()
                        daytime = (
                            daytime.second + daytime.minute*60 + daytime.hour*3600)
                        daytime = daytime / 86400
                        day_dummies = ku.to_categorical(s_timestamp.weekday(), 
                                                        num_classes=7)
                        record =  [daytime] + list(day_dummies)
                        x_inter_ngram = np.append(x_inter_ngram, [[record]], 
                                                  axis=1)
                        x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                        x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                        x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                        x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                        x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                        x_t_ngram = np.append(x_t_ngram, [pre_times], axis=1)
                        x_t_ngram = np.delete(x_t_ngram, 0, 1)
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                        preds = model.predict(inputs)
                        if parms['variant'] == 'Random Choice':
                            # Use this to get a random choice following as PDF
                            pos = np.random.choice(
                                np.arange(0, len(preds[0][0])), p=preds[0][0])
                            pos1 = np.random.choice(
                                np.arange(0, len(preds[1][0])), p=preds[1][0])
                        elif parms['variant'] == 'Arg Max':
                            # Use this to get the max prediction
                            pos = np.argmax(preds[0][0])
                            pos1 = np.argmax(preds[1][0])
                        # Check that the first prediction wont be the end of the trace
                        if (not x_trace) and (parms['index_ac'][pos] == 'end'):
                            continue
                        pred_dur = preds[2][0][0] if preds[2][0][0]>=0 else 0
                        pred_wait = preds[2][0][1] if preds[2][0][1]>=0 else 0
                        pre_times = np.array([[pred_dur, pred_wait]])
                        # rescale durations
                        dur = EventLogPredictor.rescale(
                            pred_dur, 
                            parms['scale_args']['dur'],
                            parms['norm_method'])
                        wait = EventLogPredictor.rescale(
                            pred_wait,
                            parms['scale_args']['wait'],
                            parms['norm_method'])
                        s_timestamp = (x_trace[i-1]['end_timestamp'] +
                                       timedelta(seconds=wait))
                        end_time = (s_timestamp + timedelta(seconds=dur))
                        x_trace.append(dict(
                            caseid=cid,
                            task=parms['index_ac'][pos],
                            role=parms['index_rl'][pos1],
                            start_timestamp=s_timestamp,
                            end_timestamp=end_time))
                        if parms['index_ac'][pos] == 'end':
                            break
                        i += 1
                    new_batch.extend(x_trace)
                return new_batch
            except Exception:
                traceback.print_exc()
        return gen(*args)

    def _generate_traces_parallel(self, parms, vectorizer):
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        cpu_count = multiprocessing.cpu_count()
        num_digits = len(str(parms['num_cases']))
        cases = ['Case'+str(num).zfill(num_digits)
                 for num in range(0, parms['num_cases'])]
        b_size = math.ceil(len(cases)/(cpu_count*2))
        chunks = [cases[x:x+b_size] for x in range(0, len(cases), b_size)]
        reps = len(chunks)
        pool = Pool(processes=cpu_count)
        # Generate
        args = [(cases, parms, 
                 self.model_path, self.vectorizer) for cases in chunks]
        p = pool.map_async(self.generate_trace, args)
        pbar_async(p, 'generating traces:')
        pool.close()
        # Save results
        event_log = list(itertools.chain(*p.get()))
        return event_log


    @staticmethod
    def generate_trace(args):
        def decode_trace(parms, trace, case):
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
                if parms['index_ac'][event[0]] != 'end':
                    if parms['one_timestamp']:
                        tbtw = EventLogPredictor.rescale(event[2],
                                                         parms['scale_args'],
                                                         parms['norm_method'])
                        if i == 0:
                            now = parms['start_time']
                            time = now
                        else:
                            time = (log_trace[i-1]['end_timestamp'] +
                                    timedelta(seconds=tbtw))
                        log_trace.append(dict(caseid=case,
                                              task=parms['index_ac'][event[0]],
                                              role=parms['index_rl'][event[1]],
                                              end_timestamp=time,
                                              tbtw_raw=event[2],
                                              tbtw=tbtw))
                    else:
                        dur = EventLogPredictor.rescale(event[2],
                                                        parms['scale_args']['dur'],
                                                        parms['norm_method'])
                        wait = EventLogPredictor.rescale(event[3],
                                                         parms['scale_args']['wait'],
                                                         parms['norm_method'])
                        if i == 0:
                            now = parms['start_time']
                            start_time = (now + timedelta(seconds=wait))
                        else:
                            start_time = (log_trace[i-1]['end_timestamp'] +
                                          timedelta(seconds=wait))
                        end_time = (start_time + timedelta(seconds=dur))
                        log_trace.append(dict(caseid=case,
                                              task=parms['index_ac'][event[0]],
                                              role=parms['index_rl'][event[1]],
                                              start_timestamp=start_time,
                                              end_timestamp=end_time,
                                              dur_raw=event[2],
                                              dur=dur,
                                              wait_raw=event[3],
                                              wait=wait))
                  
            return log_trace
        
        def gen(cases, parms, model_path, vectorizer):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            try:
                model = load_model(model_path)
                generated_event_log = list()
                for case in cases:
                    x_trace = list()
                    x_ac_ngram = np.zeros(
                        (1, parms['n_size']), dtype=np.float32)
                    x_rl_ngram = np.zeros(
                        (1, parms['n_size']), dtype=np.float32)
                    if parms['one_timestamp']:
                        x_t_ngram = np.zeros(
                            (1, parms['n_size'], 1), dtype=np.float32)
                    else:
                        x_t_ngram = np.zeros(
                            (1, parms['n_size'], 2), dtype=np.float32)
                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
                        x_inter_ngram = np.zeros(
                            (1, parms['n_size'], 
                             len(parms['additional_columns'])),
                            dtype=np.float32)
                        inputs = [x_ac_ngram, x_rl_ngram, 
                                  x_t_ngram, x_inter_ngram]
                    i = 1
                    while i < parms['max_trace_size']:
                        predictions = model.predict(inputs)
                        if parms['variant'] == 'Random Choice':
                            # Use this to get a random choice following as PDF
                            pos = np.random.choice(
                                np.arange(0, len(predictions[0][0])),
                                p=predictions[0][0])
                            pos1 = np.random.choice(
                                np.arange(0, len(predictions[1][0])),
                                p=predictions[1][0])
                        elif parms['variant'] == 'Arg Max':
                            # Use this to get the max prediction
                            pos = np.argmax(predictions[0][0])
                            pos1 = np.argmax(predictions[1][0])
                        # Check that the first prediction wont be the end of the trace
                        if (not x_trace) and (parms['index_ac'][pos] == 'end'):
                            continue
                        else:
                            if parms['one_timestamp']:
                                x_trace.append([pos, pos1, predictions[2][0][0]])
                            else:
                                pred_dur = (predictions[2][0][0] 
                                            if predictions[2][0][0]>=0 else 0) 
                                pred_wait = (predictions[2][0][1] 
                                            if predictions[2][0][1]>=0 else 0)
                                x_trace.append([pos, pos1,
                                                pred_dur,
                                                pred_wait])
                                pre_times = np.array([[pred_dur, pred_wait]])
                            # Add prediction to n_gram
                            x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                            x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                            x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                            x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                            # x_t_ngram = np.append(x_t_ngram, [predictions[2]], 
                            #                       axis=1)
                            x_t_ngram = np.append(x_t_ngram, [pre_times], axis=1)
                            x_t_ngram = np.delete(x_t_ngram, 0, 1)
                            if vectorizer in ['basic']:
                                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                            elif vectorizer in ['inter']:
                                x_inter_ngram = np.append(x_inter_ngram, 
                                                          [predictions[3]], 
                                                          axis=1)
                                x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                                inputs = [x_ac_ngram, x_rl_ngram, 
                                          x_t_ngram, x_inter_ngram]
                            # Stop if the next prediction is the end of the trace
                            # otherwise until the defined max_size
                            if parms['index_ac'][pos] == 'end':
                                break
                            i += 1
                    generated_event_log.extend(decode_trace(parms, x_trace, case))
                return generated_event_log
            except Exception:
                traceback.print_exc()
        return gen(*args)

    @staticmethod
    def rescale(value, scale_args, norm_method):
        if norm_method == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif norm_method == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif norm_method == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            value = (value * std) + mean
        elif norm_method == 'max':
            max_value = scale_args['max_value']
            value = np.rint(value * max_value)
        elif norm_method is None:
            value = value
        else:
            raise ValueError(norm_method)
        return value
