# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr

# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file_name'}
    try:
        return switch[opt]
    except Exception as e:
        print(e.message)
        raise Exception('Invalid option ' + opt)


def main(argv):
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parameters['one_timestamp'] = False  # Only one timestamp in the log
    parameters['read_options'] = {
        'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp']}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # Event-log filename
        parameters['file_name'] = 'PurchasingExample.xes'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt( argv, "h:f:", ['file_name='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32 # Usually 32/64/128/256
    parameters['norm_method'] = ['max', 'lognorm']
    # # Model types --> shared_cat, shared_cat_inter, shared_cat_rd
    # # cnn_lstm_inter, simple_gan
    parameters['model_type'] = ['shared_cat', 'specialized', 'concatenated']
    # parameters['model_type'] = 'shared_cat'
    parameters['imp'] = 1
    parameters['max_eval'] = 2
    parameters['batch_size'] = 32 # Usually 32/64/128/256
    parameters['epochs'] = 2
    parameters['n_size'] = [5]
    parameters['l_size'] = [50] 
    parameters['lstm_act'] = ['selu', 'relu', 'tanh']
    parameters['dense_act'] = ['linear']
    parameters['optim'] = ['Nadam', 'Adam']
    
    if parameters['model_type'] == 'simple_gan':
        parameters['gan_pretrain'] = False
    # Train models
    print(parameters)
    trainer = tr.ModelTrainer(parameters)
    # parameters = dict()
    # # Clean parameters and define validation experiment
    # parameters['folder'] = trainer.output
    # parameters['model_file'] = trainer.model
    # parameters['activity'] = 'pred_log'
    # print(parameters['folder'])
    # print(parameters['model_file'])
    # predictor = pr.ModelPredictor(parameters)
    # print(predictor.acc)

if __name__ == "__main__":
    main(sys.argv[1:])
