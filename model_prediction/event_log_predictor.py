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
        self.imp = 'Arg Max'
        self.max_trace_size = 0

    def predict(self, params, model, examples, imp):
        self.model = model
        self.max_trace_size = params['max_trace_size']
        print(params)
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        return predictor(params)

    def _get_predictor(self, model_type):
        if model_type in ['shared_cat', 'shared_cat_inter']:
            return self._predict_event_log_shared_cat
        else:
            raise ValueError(model_type)

    def _predict_event_log_shared_cat(self, parms):
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
            x_t_ngram = np.zeros(
                (1, parms['dim']['time_dim'], 1), dtype=np.float32)

            for _ in range(1, self.max_trace_size):
                predictions = self.model.predict([x_ac_ngram, x_rl_ngram, x_t_ngram])
                if self.imp == 'Random Choice':
                    # Use this to get a random choice following as PDF
                    pos = np.random.choice(
                        np.arange(0, len(predictions[0][0])),
                        p=predictions[0][0])
                    pos1 = np.random.choice(
                        np.arange(0, len(predictions[1][0])),
                        p=predictions[1][0])
                elif self.imp == 'Arg Max':
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
                if parms['index_ac'][pos] == 'end':
                    break
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
                if parms['norm_method'] == 'activity':
                    tbtw = (event[2] * parms['max_dur'][event[0]])
                elif parms['norm_method'] == 'lognorm':
                    tbtw = math.expm1(event[2] * parms['max_dur'])
                else:
                    tbtw = np.rint(event[2] * parms['max_dur'])
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
        return log_trace
