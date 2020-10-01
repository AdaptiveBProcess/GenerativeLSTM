# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:11:17 2020

@author: Manuel Camargo
"""
import numpy as np
import math
import datetime

from datetime import timedelta

from support_modules import support as sup

class EventLogPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.imp = 'arg_max'
        self.max_trace_size = 0

    def predict(self, params, model, examples, imp, vectorizer):
        self.model = model
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        self.params = params
        self.vectorizer = vectorizer
        predictor = self._get_predictor(params['model_type'])
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        return self._predict_event_log_shared_cat

    def _predict_event_log_shared_cat(self, parms, vectorizer):
        """Generate business process traces using a keras trained model.
        Args:
            model (keras model): keras trained model.
            imp (str): method of next event selection.
            num_cases (int): number of traces to generate.
            max_trace_size (int): max size of the trace
        """
        sup.print_performed_task('Generating traces')
        generated_event_log = list()
        for case in range(0, parms['num_cases']):
            x_trace = list()
            x_ac_ngram = np.zeros(
                (1, parms['dim']['time_dim']), dtype=np.float32)
            x_rl_ngram = np.zeros(
                (1, parms['dim']['time_dim']), dtype=np.float32)
            if parms['one_timestamp']:
                x_t_ngram = np.zeros(
                    (1, parms['dim']['time_dim'], 1), dtype=np.float32)
            else:
                x_t_ngram = np.zeros(
                    (1, parms['dim']['time_dim'], 2), dtype=np.float32)
            # TODO: add intercase support
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
                x_inter_ngram = np.zeros(
                    (1, parms['dim']['time_dim'], len(parms['additional_columns'])), dtype=np.float32)
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            i = 1
            # for _ in range(1, self.max_trace_size):
            while i < self.max_trace_size:
                # predictions = self.model.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
                predictions = self.model.predict(inputs)
                if self.imp == 'random_choice':
                    # Use this to get a random choice following as PDF
                    pos = np.random.choice(
                        np.arange(0, len(predictions[0][0])),
                        p=predictions[0][0])
                    pos1 = np.random.choice(
                        np.arange(0, len(predictions[1][0])),
                        p=predictions[1][0])
                elif self.imp == 'arg_max':
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
                        x_trace.append([pos, pos1,
                                        predictions[2][0][0],
                                        predictions[2][0][1]])
        #            # Add prediction to n_gram
                    x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                    x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                    x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                    x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                    x_t_ngram = np.append(x_t_ngram, [predictions[2]], axis=1)
                    x_t_ngram = np.delete(x_t_ngram, 0, 1)
                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
                        x_inter_ngram = np.append(x_inter_ngram, [predictions[3]], axis=1)
                        x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
        #            # Stop if the next prediction is the end of the trace
        #            # otherwise until the defined max_size
                    if parms['index_ac'][pos] == 'end':
                        break
                    i += 1
            generated_event_log.extend(self.decode_trace(parms, x_trace, case))
        sup.print_done_task()
        return generated_event_log

    @staticmethod
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
                        now = datetime.datetime.now()
                        now.strftime(parms['read_options']['timeformat'])
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
                        now = datetime.datetime.now()
                        now.strftime(parms['read_options']['timeformat'])
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
