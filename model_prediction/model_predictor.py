# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""
import os
import json

import pandas as pd
import numpy as np
import configparser as cp

from tensorflow.keras.models import load_model

from support_modules.readers import log_reader as lr
from support_modules import support as sup

from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev


class ModelPredictor():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.parms = parms
        # load parameters
        self.load_parameters()
        self.model_name, _ = os.path.splitext(parms['model_file'])
        self.model = load_model(os.path.join(self.output_route,
                                             parms['model_file']))

        self.log = self.load_log_test(self.output_route, self.parms)

        self.samples = dict()
        self.predictions = None
        self.run_num = 0

        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        print(self.model_def)
        self.parms['additional_columns'] = self.model_def['additional_columns']
        self.acc = self.execute_predictive_task()

    def execute_predictive_task(self):
        # create examples for next event and suffix
        if self.parms['activity'] == 'pred_log':
            self.parms['num_cases'] = len(self.log.caseid.unique())
        else:
            sampler = it.SamplesCreator()
            sampler.create(self, self.parms['activity'])
        # predict
        self.imp = self.parms ['variant']
        self.run_num = 0
        for i in range(0, self.parms['rep']):
            self.predict_values()
            self.run_num += 1
        # export predictions
        self.export_predictions()
        # assesment
        evaluator = EvaluateTask()
        if self.parms['activity'] == 'pred_log':
            data = self.append_sources(self.log, self.predictions,
                                        self.parms['one_timestamp'])
            data['caseid'] = data['caseid'].astype(str)
            return evaluator.evaluate(self.parms, data)
        else:
            return evaluator.evaluate(self.parms, self.predictions) 

    def predict_values(self):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'])

    @staticmethod
    def load_log_test(output_route, parms):
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
            if self.parms['one_timestamp']:
                self.parms['scale_args'] = {
                    k: float(v) for k, v in data['scale_args'].items()}
            else:
                for key in data['scale_args'].keys():
                    self.parms['scale_args'][key] = {
                        k: float(v) for k, v in data['scale_args'][key].items()}
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}

    def sampling(self, sampler):
        sampler.register_sampler(self.parms['model_type'],
                                 self.model_def['vectorizer'])
        self.samples = sampler.create_samples(
            self.parms, self.log, self.ac_index,
            self.rl_index, self.model_def['additional_columns'])

    def predict(self, executioner):
        results = executioner.predict(self.parms,
                                      self.model,
                                      self.samples,
                                      self.imp,
                                      self.model_def['vectorizer'])
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
    def append_sources(source_log, source_predictions, one_timestamp):
        log = source_log.copy()
        columns = ['caseid', 'task', 'end_timestamp', 'role']
        if not one_timestamp:
            columns += ['start_timestamp']
        log = log[columns]
        log['run_num'] = 0
        log['implementation'] = 'log'
        predictions = source_predictions.copy()
        columns = log.columns
        predictions = predictions[columns]
        return log.append(predictions, ignore_index=True)

    @staticmethod
    def scale_feature(log, feature, parms, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        method = parms['norm_method']
        scale_args = parms['scale_args']
        if method == 'lognorm':
            log[feature + '_log'] = np.log1p(log[feature])
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature+'_log'], min_value), (max_value - min_value))
            log = log.drop((feature + '_log'), axis=1)
        elif method == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature], min_value), (max_value - min_value))
        elif method == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                               std)
        elif method == 'max':
            max_value = scale_args['max_value']
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
        elif method is None:
            log[feature+'_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log

    def read_model_definition(self, model_type):
        Config = cp.ConfigParser(interpolation=None)
        Config.read('models_spec.ini')
        #File name with extension
        self.model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type,'additional_columns'), dtype='str')
        self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')
        
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
        evaluator = ev.Evaluator(parms['one_timestamp'])
        ac_sim = evaluator.measure('accuracy', data, 'ac')
        rl_sim = evaluator.measure('accuracy', data, 'rl')
        mean_ac = ac_sim.accuracy.mean()
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_next', data, 'tm')
            tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_next', data, 'dur')
            wait_mae = evaluator.measure('mae_next', data, 'wait')
            dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(dur_mae, 'dur', parms)
            self.save_results(wait_mae, 'wait', parms)
        return mean_ac

    def _evaluate_pred_sfx(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'])
        ac_sim = evaluator.measure('similarity', data, 'ac')
        rl_sim = evaluator.measure('similarity', data, 'rl')
        mean_sim = ac_sim['mean'].mean()
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_suffix', data, 'tm')
            tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_suffix', data, 'dur')
            wait_mae = evaluator.measure('mae_suffix', data, 'wait')
            dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(dur_mae, 'dur', parms)
            self.save_results(wait_mae, 'wait', parms)
        return mean_sim

    def _evaluate_predict_log(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'])
        dl = evaluator.measure('dl', data)
        els = evaluator.measure('els', data)
        mean_els = els.els.mean()
        mae = evaluator.measure('mae_log', data)
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc]*len(dl), ignore_index=True)
        # exp_desc = pd.concat([exp_desc]*len(els), ignore_index=True)
        dl = pd.concat([dl, exp_desc], axis=1).to_dict('records')
        els = pd.concat([els, exp_desc], axis=1).to_dict('records')
        mae = pd.concat([mae, exp_desc], axis=1).to_dict('records')
        self.save_results(dl, 'dl', parms)
        self.save_results(els, 'els', parms)
        self.save_results(mae, 'mae', parms)
        return mean_els

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
