# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:24:38 2020

@author: Manuel Camargo
"""
from model_prediction import next_event_samples_creator as nesc
from model_prediction import suffix_samples_creator as ssc


from model_prediction import next_event_predictor as nep
from model_prediction import suffix_predictor as sp
from model_prediction import event_log_predictor as elp


class SamplesCreator:
    def create(self, predictor, activity):
        sampler = self._get_samples_creator(activity)
        predictor.sampling(sampler)

    def _get_samples_creator(self, activity):
        if activity == 'predict_next':
            return nesc.NextEventSamplesCreator()
        elif activity == 'pred_sfx':
            return ssc.SuffixSamplesCreator()
        else:
            raise ValueError(activity)


class PredictionTasksExecutioner:
    def predict(self, predictor, activity):
        executioner = self._get_predictor(activity)
        predictor.predict(executioner)

    def _get_predictor(self, activity):
        if activity == 'predict_next':
            return nep.NextEventPredictor()
        elif activity == 'pred_sfx':
            return sp.SuffixPredictor()
        elif activity == 'pred_log':
            return elp.EventLogPredictor()
        else:
            raise ValueError(activity)
