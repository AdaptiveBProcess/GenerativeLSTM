# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:48:09 2020

@author: Manuel Camargo
"""
import os

# import pandas as pd
# import numpy as np

# from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL
from hyperopt import tpe, hp, fmin, STATUS_OK, STATUS_FAIL

from hyperopt.mongoexp import MongoTrials


class LSTMOptimizer():
    """
    Hyperparameter-optimizer class
    """

    def __init__(self):
        """constructor"""
        self.space = self.define_search_space()
        # self.settings = settings
        # self.args = args
        # Trials object to track progress
        
        self.bayes_trials = MongoTrials('mongo://localhost:27017/tpe_mongo/jobs', exp_key='exp1')

        # self.bayes_trials = Trials()

    @staticmethod
    def define_search_space():
        space = {'par1': hp.uniform('par1', 0, 1),
                 'par2': hp.uniform('par2', 0, 1)}
                    # 'alg_manag': hp.choice('alg_manag',
                    #                        ['replacement',
                    #                         'repair',
                    #                         'removal']),
                    # 'rp_similarity': hp.uniform('rp_similarity',
                    #                             args['rp_similarity'][0],
                    #                             args['rp_similarity'][1]),
                    # 'gate_management': hp.choice('gate_management',
                    #                              args['gate_management'])},
                 # **settings}
        return space

    def execute_trials(self):
        # create a new instance of Simod
        # def exec_simod(instance_settings):
        #     simod = Simod(instance_settings)
        #     simod.execute_pipeline(self.settings['exec_mode'])
        #     return simod.response
        
        def exec_simod(instance_settings):
            test = instance_settings['par1']
            return test
        # Optimize
        best = fmin(fn=exec_simod,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=3,
                    trials=self.bayes_trials,
                    show_progressbar=False)
        print(best)
        
opt = LSTMOptimizer()
opt.execute_trials()