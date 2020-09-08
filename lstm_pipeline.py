# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""

import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size', '-e': 'splits', '-g': 'sub_group',
              '-v': 'variant', '-r': 'rep'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


# --setup--
def main(argv):
    """Main aplication method"""
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32 # Usually 32/64/128/256
    parameters['epochs'] = 200
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Event-log parameters
        parameters['file_name'] = 'inter_Production_training.csv'
        # Event-log reading parameters
        parameters['one_timestamp'] = False  # Only one timestamp in the log
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # Specific model training parameters
        parameters['imp'] = 1  # keras lstm implementation 1 cpu,2 gpu
        parameters['lstm_act'] = 'relu'  # optimization function Keras
        parameters['dense_act'] = None  # optimization function Keras
        parameters['optim'] = 'Adam'  # optimization function Keras
        parameters['norm_method'] = 'max'  # max, lognorm
        # Model types --> shared_cat, shared_cat_inter, specialized, concatenated
        parameters['model_type'] = 'concatenated_inter'
        parameters['n_size'] = 10  # n-gram size
        parameters['l_size'] = 50  # LSTM layer sizes
        parameters['is_single_exec'] = False  # single or batch execution
        # variants and repetitions to be tested Random Choice, Arg Max
        parameters['variant'] = 'Random Choice'
        parameters['rep'] = 2
        if parameters['model_type'] == 'simple_gan':
            parameters['gan_pretrain'] = False
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:e:v:r:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size=', 'splits=', 'sub_group=',
                 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size',
                             'max_trace_size','splits', 'rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                              'ns_include': True}
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Train models
    print(parameters)
    trainer = tr.ModelTrainer(parameters)
    parameters = dict()
    # Clean parameters and define validation experiment
    parameters['folder'] = trainer.output
    parameters['model_file'] = trainer.model
    parameters['activity'] = 'pred_log'
    print(parameters['folder'])
    print(parameters['model_file'])
    predictor = pr.ModelPredictor(parameters)
    print(predictor.acc)

if __name__ == "__main__":
    main(sys.argv[1:])
