# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""
import os
import json

import pandas as pd

from keras.models import load_model

from support_modules.readers import log_reader as lr
from support_modules import nn_support as nsup
from support_modules import support as sup


from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev


class ModelPredictor():
    """
    This is the man class encharged of the model training
    """

    def __init__(self, parms):
        self.parms = parms
        self.output_route = os.path.join('output_files', parms['folder'])
        self.model_name, _ = os.path.splitext(parms['model_file'])
        self.model = load_model(os.path.join(self.output_route,
                                             parms['model_file']))

        self.log = self.load_log_test(self.output_route, parms)
        self.ac_index = dict()  # TODO Evaluar si se dejan aca o en parms
        self.rl_index = dict()  # TODO Evaluar si se dejan aca o en parms

        self.samples = dict()
        self.predictions = None
        self.run_num = 0

        self.execute_predictive_task()

    def execute_predictive_task(self):
        # load parameters
        self.load_parameters()
        print(self.parms)
        # scale features
        self.log = nsup.scale_feature(self.log, 'dur',
                                      self.parms['norm_method'])
        # create examples for next event and suffix
        if self.parms['activity'] == 'pred_log':
            self.parms['num_cases'] = len(self.log.caseid.unique())
        else:
            sampler = it.SamplesCreator()
            sampler.create(self, self.parms['activity'])
        # predict
        for variant in self.parms['variants']:
            self.imp = variant['imp']
            self.run_num = 0
            for i in range(0, variant['rep']):
                self.predict_values()
                self.run_num += 1
        # export predictions
        self.export_predictions()
        # assesment
        evaluator = EvaluateTask()
        if self.parms['activity'] == 'pred_log':
            data = self.append_sources(self.log, self.predictions)
            data['caseid'] = data['caseid'].astype(str)
            evaluator.evaluate(self.parms, data)
            # data.to_csv('test_data.csv')
        else:
            evaluator.evaluate(self.parms, self.predictions)

    def predict_values(self):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'])

    @staticmethod
    def load_log_test(output_route, parms):
        parms['read_options']['filter_d_attrib'] = False
        df_test = lr.LogReader(
            os.path.join(output_route, 'parameters', 'test_log.csv'),
            parms['read_options'])
        df_test = pd.DataFrame(df_test.data)
        df_test = df_test[~df_test.task.isin(['Start', 'End'])]
        return df_test

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            self.parms = {**self.parms, **{k: v for k, v in data.items()}}
            self.parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            self.parms['max_dur'] = float(data['max_dur'])
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}

    def sampling(self, sampler):
        self.samples = sampler.create_samples(self.parms,
                                              self.log,
                                              self.ac_index,
                                              self.rl_index)

    def predict(self, executioner):
        results = executioner.predict(self.parms,
                                      self.model,
                                      self.samples,
                                      self.imp)
        results = pd.DataFrame(results)
        results['run_num'] = self.run_num
        results['implementation'] = self.imp
        if self.predictions is None:
            self.predictions = results
        else:
            self.predictions = self.predictions.append(results,
                                                       ignore_index=True)

    def export_predictions(self):
        output_folder = os.path.join(self.output_route, 'results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filename = self.model_name + '_' + self.parms['activity'] + '.csv'
        self.predictions.to_csv(os.path.join(output_folder, filename),
                                index=False)

    @staticmethod
    def append_sources(source_log, source_predictions):
        log = source_log.copy()
        log = log[['caseid', 'task', 'end_timestamp', 'role']]
        log['run_num'] = 0
        log['implementation'] = 'log'
        predictions = source_predictions.copy()
        predictions = predictions.drop(columns=['tbtw_raw', 'tbtw'])
        return log.append(predictions, ignore_index=True)


class EvaluateTask():

    def evaluate(self, parms, data):
        sampler = self._get_evaluator(parms['activity'])
        return sampler(data, parms)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        elif activity == 'pred_sfx':
            return self._evaluate_pred_sfx
        elif activity == 'pred_log':
            return self._evaluate_predict_log
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator()
        ac_sim = evaluator.measure('accuracy', data, 'ac')
        rl_sim = evaluator.measure('accuracy', data, 'rl')
        # tm_mae = evaluator.measure('mae_suffix', data, 'tm')
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        # tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        # self.save_results(tm_mae, 'tm', parms)

    def _evaluate_pred_sfx(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator()
        ac_sim = evaluator.measure('similarity', data, 'ac')
        rl_sim = evaluator.measure('similarity', data, 'rl')
        tm_mae = evaluator.measure('mae_suffix', data, 'tm')
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        self.save_results(tm_mae, 'tm', parms)

    def _evaluate_predict_log(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator()
        dl = evaluator.measure('dl', data)
        els = evaluator.measure('els', data)
        mae = evaluator.measure('mae_log', data)
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(dl), ignore_index=True)
        dl = pd.concat([dl, exp_desc], axis=1).to_dict('records')
        els = pd.concat([els, exp_desc], axis=1).to_dict('records')
        mae = pd.concat([mae, exp_desc], axis=1).to_dict('records')
        self.save_results(dl, 'ac', parms)
        self.save_results(els, 'rl', parms)
        self.save_results(mae, 'tm', parms)

    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        return exp_desc

    @staticmethod
    def save_results(measurements, feature, parms):
        if measurements:
            if parms['is_single_exec']:
                output_route = os.path.join('output_files',
                                            parms['folder'],
                                            'results')
                model_name, _ = os.path.splitext(parms['model_file'])
                sup.create_csv_file_header(
                    measurements,
                    os.path.join(
                        output_route,
                        model_name+'_'+feature+'_'+parms['activity']+'.csv'))
            else:
                if os.path.exists(os.path.join(
                        'output_files', feature+'_'+parms['activity']+'.csv')):
                    sup.create_csv_file(
                        measurements,
                        os.path.join('output_files',
                                     feature+'_'+parms['activity']+'.csv'),
                        mode='a')
                else:
                    sup.create_csv_file_header(
                        measurements,
                        os.path.join('output_files',
                                     feature+'_'+parms['activity']+'.csv'))
