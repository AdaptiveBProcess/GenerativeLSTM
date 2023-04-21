# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_training import model_trainer as tr


# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file_name', '-m': 'model_family',
              '-e': 'max_eval', '-o': 'opt_method'}
    return switch.get(opt)


def main(argv):
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parameters['one_timestamp'] = False  # Only one timestamp in the log
    parameters['read_options'] = {
        'timeformat': '%Y/%m/%d %H:%M:%S',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp']}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        parameters['file_name'] = 'Production.csv'
        parameters['model_family'] = 'lstm'
        parameters['opt_method'] = 'bayesian'  # 'rand_hpc', 'bayesian'
        parameters['max_eval'] = 1
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:m:e:o:",
                                    ['file_name=', 'model_family=',
                                     'max_eval=', 'opt_method='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['max_eval']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32  # Usually 32/64/128/256
    parameters['norm_method'] = ['max', 'lognorm']
    parameters['imp'] = 1
    parameters['epochs'] = 200
    parameters['n_size'] = [5, 10, 15]
    parameters['l_size'] = [50, 100]
    parameters['lstm_act'] = ['selu', 'tanh']
    if parameters['model_family'] == 'lstm':
        parameters['model_type'] = ['shared_cat', 'concatenated']
    elif parameters['model_family'] == 'gru':
        parameters['model_type'] = ['shared_cat_gru', 'concatenated_gru']
    elif parameters['model_family'] == 'lstm_cx':
        parameters['model_type'] = ['shared_cat_cx', 'concatenated_cx']
    elif parameters['model_family'] == 'gru_cx':
        parameters['model_type'] = ['shared_cat_gru_cx', 'concatenated_gru_cx']
    parameters['dense_act'] = ['linear']
    parameters['optim'] = ['Nadam']

    if parameters['model_type'] == 'simple_gan':
        parameters['gan_pretrain'] = False
    parameters.pop('model_family', None)
    # Train models
    tr.ModelTrainer(parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
