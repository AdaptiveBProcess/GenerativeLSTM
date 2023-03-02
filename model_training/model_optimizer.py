# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:48:57 2020

@author: Manuel Camargo
"""
import os
import copy
import ast
import traceback
import pandas as pd
import configparser as cp
from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

import utils.support as sup
import readers.log_splitter as ls

import tensorflow as tf
from model_training import samples_creator as sc
from model_training import model_loader as mload
from model_training import features_manager as feat


class ModelOptimizer():
    """
    Hyperparameter-optimizer class
    """
    class Decorators(object):

        @classmethod
        def safe_exec(cls, method):
            """
            Decorator to safe execute methods and return the state
            ----------
            method : Any method.
            Returns
            -------
            dict : execution status
            """
            def safety_check(*args, **kw):
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        response['status'] = STATUS_FAIL
                return response
            return safety_check
        
    def __init__(self, parms, log, ac_index, ac_weights, rl_index, rl_weights):
        """constructor"""
        self.space = self.define_search_space(parms)
        self.log = copy.deepcopy(log)
        self.ac_index = ac_index
        self.ac_weights = ac_weights
        self.rl_index = rl_index
        self.rl_weights = rl_weights
        
        # Load settings
        self.parms = parms
        self.temp_output = parms['output']
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.file_name = os.path.join(self.temp_output, 
                                      sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        # Trials object to track progress
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_params = dict()
        self.best_loss = 1
        
    @staticmethod
    def define_search_space(parms):
        space = {'model_type': hp.choice('model_type', parms['model_type']),
                 'n_size': hp.choice('n_size', parms['n_size']),
                 'l_size': hp.choice('l_size', parms['l_size']),
                 'lstm_act': hp.choice('lstm_act', parms['lstm_act']),
                 'dense_act': hp.choice('dense_act', parms['dense_act']),
                 'norm_method': hp.choice('norm_method', parms['norm_method']),
                 'optim': hp.choice('optim', parms['optim']),
                 'imp': parms['imp'], 'file': parms['file_name'],
                 'batch_size': parms['batch_size'], 'epochs': parms['epochs'],
                 'one_timestamp': parms['one_timestamp']}
        return space

    def execute_trials(self):
        def exec_pipeline(trial_stg):
            print(trial_stg)
            status = STATUS_OK
            # Path redefinition
            rsp = self._temp_path_redef(trial_stg, status=status)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg
            # Model definition
            model_def = self.read_model_definition(trial_stg['model_type'])
            # Scale values
            log, trial_stg = self._scale_values(self.log, trial_stg, model_def)
            # split validation
            log_valdn, log_train = self.split_timeline(0.8, log, trial_stg['one_timestamp'])
            print('train split size:', len(log_train))
            print('valdn split size:', len(log_valdn))
            # Vectorize input
            vectorizer = sc.SequencesCreator(
                self.parms['read_options']['one_timestamp'], 
                self.ac_index, self.rl_index)
            vectorizer.register_vectorizer(trial_stg['model_type'],
                                           model_def['vectorizer'])
            train_vec = vectorizer.vectorize(trial_stg['model_type'],
                                             log_train,
                                             trial_stg,
                                             model_def['additional_columns'])
            valdn_vec = vectorizer.vectorize(trial_stg['model_type'],
                                             log_valdn,
                                             trial_stg,
                                             model_def['additional_columns'])
            # Train
            m_loader = mload.ModelLoader(trial_stg)
            m_loader.register_model(trial_stg['model_type'],
                                    model_def['trainer'])
            tf.compat.v1.reset_default_graph()
            model = m_loader.train(trial_stg['model_type'],
                                   train_vec, 
                                   valdn_vec,
                                   self.ac_weights,
                                   self.rl_weights,
                                   trial_stg['output'])
            # evaluation
            x_input = {'ac_input': valdn_vec['prefixes']['activities'],
                       'rl_input': valdn_vec['prefixes']['roles'],
                       't_input': valdn_vec['prefixes']['times']}
            if trial_stg['model_type'] in ['shared_cat_cx', 
                                           'concatenated_cx',
                                           'shared_cat_gru_cx', 
                                           'concatenated_gru_cx']:
                x_input['inter_input']= valdn_vec['prefixes']['inter_attr']
            acc = model.evaluate(
                x=x_input,
                y={'act_output': valdn_vec['next_evt']['activities'],
                   'role_output': valdn_vec['next_evt']['roles'],
                   'time_output': valdn_vec['next_evt']['times']},
                return_dict=True)
            rsp = self._define_response(trial_stg, status, acc['loss'])
            print("-- End of trial --")
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.parms['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        # Save results
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                       .sort_values('loss', ascending=True))
            result = results[results.status == 'ok'].head(1).iloc[0]
            self.best_output = result.output
            self.best_loss = result.loss
            self.best_params = {k: self.parms[k][v] for k, v in best.items()}
            opt_res = pd.read_csv(self.file_name)
            opt_res = opt_res[opt_res.output == result.output].iloc[0]
            self.best_params['scale_args'] = ast.literal_eval(opt_res.scale_args)
        except Exception as e:
            print(e)
            pass

    @Decorators.safe_exec
    def _temp_path_redef(self, settings, **kwargs) -> dict:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
        return settings

    @staticmethod
    def _scale_values(log, params, model_def):
        # Features treatment
        inp = feat.FeaturesMannager(params)
        # Register scaler
        inp.register_scaler(params['model_type'], model_def['scaler'])
        # Scale features
        log, params['scale_args'] = inp.calculate(
            log, model_def['additional_columns'])
        return log, params

    def _define_response(self, parms, status, loss, **kwargs) -> dict:
        print(loss)
        response = dict()
        measurements = list()
        data = {'n_size': parms['n_size'],
                'l_size': parms['l_size'],
                'lstm_act': parms['lstm_act'],
                'dense_act': parms['dense_act'],
                'optim': parms['optim'],
                'scale_args': parms['scale_args'],
                'output': parms['output']}
        response['output'] = parms['output']
        if status == STATUS_OK:
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL
            measurements.append({**{'loss': loss,
                                    'sim_metric': 'val_loss', 
                                    'status': response['status']},
                                 **data})
        else:
            response['status'] = status
            measurements.append({**{'loss': 1,
                                    'sim_metric': 'val_loss',
                                    'status': response['status']},
                                 **data})
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response

    @staticmethod
    def split_timeline(size: float, log: pd.DataFrame, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(log)
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(log)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        log_valdn = (valdn.sort_values(key, ascending=True).reset_index(drop=True))
        log_train = (train.sort_values(key, ascending=True).reset_index(drop=True))
        return log_valdn, log_train

    @staticmethod
    def read_model_definition(model_type):
        model_def = dict()
        config = cp.ConfigParser(interpolation=None)
        config.read('models_spec.ini')
        #  File name with extension
        model_def['additional_columns'] = sup.reduce_list(config.get(model_type, 'additional_columns'), dtype='str')
        model_def['scaler'] = config.get(model_type, 'scaler')
        model_def['vectorizer'] = config.get(model_type, 'vectorizer')
        model_def['trainer'] = config.get(model_type, 'trainer')
        return model_def
