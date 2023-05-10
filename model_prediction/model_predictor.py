# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""
import os
import json
import copy

import pandas as pd
import numpy as np
import configparser as cp

import readers.log_reader as lr
import utils.support as sup

from model_training import features_manager as feat
from model_prediction import interfaces as it
import analyzers.sim_evaluator as ev
from support_modules import traces_evaluation as te

class ModelPredictor():
    """
    This is the man class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.parms = parms
        # load parameters
        self.load_parameters()
        self.model_name = os.path.join(self.output_route, parms['model_file'])
        self.log = self.load_log_test(self.output_route, self.parms)

        self.samples = dict()
        self.predictions = None
        self.sim_values = list()

        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        self.parms['additional_columns'] = self.model_def['additional_columns']
        self.acc = self.execute_predictive_task()

    def execute_predictive_task(self):
        # create examples for next event and suffix
        if self.parms['activity'] == 'pred_log':
            #self.parms['num_cases'] = len(self.log.caseid.unique())
            self.parms['len_log'] = len(self.log.caseid.unique())
            self.parms['num_cases'] = 150
            self.parms['start_time'] = self.log.start_timestamp.min()
        else:
            feat_mannager = feat.FeaturesMannager(self.parms)
            feat_mannager.register_scaler(self.parms['model_type'],
                                          self.model_def['vectorizer'])
            self.log, _ = feat_mannager.calculate(
                self.log, self.parms['additional_columns'])
            sampler = it.SamplesCreator()
            sampler.create(self, self.parms['activity'])
        # predict
        self.imp = self.parms['variant']

        org_log_path = os.path.join('output_files', self.parms['folder'], 'parameters', '{}_ASIS.csv'.format(self.parms['log_name']))
        df_org = pd.read_csv(org_log_path)
        df_org['start_timestamp'] = pd.to_datetime(df_org['start_timestamp'])
        df_org['end_timestamp'] = pd.to_datetime(df_org['end_timestamp'])

        self.parms['ac_index'] = self.index_ac = {self.parms['index_ac'][key]:key for key in self.parms['index_ac'].keys()}
        self.parms['rules'] = te.extract_rules()

        self.parms['traces_gen_path'] = os.path.join('output_files', self.parms['folder'], 'parameters', 'traces_generated')
        if not os.path.exists(self.parms['traces_gen_path']):
            os.mkdir(self.parms['traces_gen_path'])

        gs = te.GenerateStats(df_org, self.parms['ac_index'], self.parms['rules']['path'], self.parms['rules']['rule'])        
        self.parms['pos_cases_org'], self.parms['total_cases_org'] = gs.get_stats()

        if self.parms['rules']['variation'] == '+':
            self.parms['new_prop_cases'] = (self.parms['pos_cases_org']/self.parms['total_cases_org']) + self.parms['rules']['prop_variation']
        elif self.parms['rules']['variation'] == '-':
            self.parms['new_prop_cases'] = (self.parms['pos_cases_org']/self.parms['total_cases_org']) - self.parms['rules']['prop_variation']

        for run_num in range(0, self.parms['rep']):

            #Modificar
            self.predict_values(run_num)
    
            # export predictions
            self.export_predictions(run_num)
            # assesment
            evaluator = EvaluateTask()
            if self.parms['activity'] == 'pred_log':
                self.sim_values.extend(
                    evaluator.evaluate(self.parms,
                                       self.log,
                                    self.predictions,
                                    run_num))
            else:
                evaluator.evaluate(self.predictions, self.parms)
        self._export_results(self.output_route)

    def predict_values(self, run_num):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'], run_num)

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
            parms = {k: v for k, v in data.items()}
            parms.pop('rep', None)
            self.parms = {**self.parms, **parms}
            if 'dim' in data.keys():
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


    def predict(self, executioner, run_num):
        
        results = executioner.predict(self.parms,
                                      self.model_name,
                                      self.samples,
                                      self.imp,
                                      self.model_def['vectorizer'])
        results = pd.DataFrame(results)
        self.predictions = results

    def export_predictions(self, r_num):
        # output_folder = os.path.join(self.output_route, 'results')
        if not os.path.exists(self.output_route):
            os.makedirs(self.output_route)

        df_traces_generated, files_gen = te.get_stats_log_traces(self.parms['traces_gen_path'])
        cols = ['caseid', 'task', 'role', 'start_timestamp','end_timestamp']

        if self.parms['include_org_log']:
            log_filtered = pd.DataFrame(data=[], columns=cols)
            for caseid in self.log['caseid'].drop_duplicates():
                log_tmp = self.log[self.log['caseid']==caseid]
                if te.evaluate_condition(log_tmp, self.ac_index, self.parms['rules']['path'], self.parms['rules']['rule']):
                    log_filtered = pd.concat([log_filtered, log_tmp])

            final_log = pd.concat([log_filtered, df_traces_generated[cols]])
        else:
            final_log = df_traces_generated[cols]

        final_log['start_timestamp'] = pd.to_datetime(final_log['start_timestamp']).dt.strftime(self.parms['read_options']['timeformat'])
        final_log['end_timestamp'] = pd.to_datetime(final_log['end_timestamp']).dt.strftime(self.parms['read_options']['timeformat'])
        final_log = final_log.rename({'role':'user'}, axis=1)

        final_log.to_csv(os.path.join('output_files', self.parms['folder'], 'parameters', '{}_TOBE.csv'.format(self.parms['log_name'])), index=False)

        if len(files_gen)>0:
            for file_gen in files_gen:
                os.remove(file_gen)

        self.predictions.to_csv(
            os.path.join(
                self.output_route, 'gen_'+ 
                self.parms['model_file'].split('.')[0]+'_'+str(r_num+1)+'.csv'), 
            index=False)

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

    def _export_results(self, output_path) -> None:
        # Save results
        pd.DataFrame(self.sim_values).to_csv(
            os.path.join(self.output_route, sup.file_id(prefix='SE_')), 
            index=False)
        # Save logs        
        log_test = self.log[~self.log.task.isin(['Start', 'End'])]
        log_test.to_csv(
            os.path.join(self.output_route, 'tst_'+
                         self.parms['model_file'].split('.')[0]+'.csv'), 
            index=False)
        
class EvaluateTask():

    def evaluate(self, parms, log, predictions, rep_num):
        sampler = self._get_evaluator(parms['activity'])
        return sampler(parms, log, predictions, rep_num)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        elif activity == 'pred_sfx':
            return self._evaluate_pred_sfx
        elif activity == 'pred_log':
            return self._evaluate_predict_log
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms, rep_num):
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

    def _evaluate_pred_sfx(self, data, parms, rep_num):
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

    @staticmethod
    def _evaluate_predict_log(parms, log, sim_log, rep_num):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        sim_values = list()
        log = copy.deepcopy(log)
        log = log[~log['task'].isin(['Start', 'End', 'start', 'end'])]
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']
        sim_log = sim_log[~sim_log['task'].isin(['Start', 'End', 'start', 'end'])]
        evaluator = ev.SimilarityEvaluator(log, sim_log, parms)
        metrics = ['tsd', 'day_hour_emd', 'log_mae', 'dl', 'mae']
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
        return sim_values

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
