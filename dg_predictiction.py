# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:08:25 2021

@author: Manuel Camargo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import getopt

from model_prediction import model_predictor as pr

# =============================================================================
# Main function
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-a': 'activity', '-c': 'folder', 
              '-b': 'model_file', '-v': 'variant', '-r': 'rep'}
    return switch.get(opt)


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
        parameters['activity'] = 'pred_log'
        parameters['folder'] = '20210208_AE2236CA_E88C_4EC9_ABC1_17173FD4DCFF'
        parameters['model_file'] = 'confidential_2000.h5'
        parameters['is_single_exec'] = False  # single or batch execution
        # variants and repetitions to be tested Random Choice, Arg Max
        parameters['variant'] = 'Random Choice'
        parameters['rep'] = 5
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "ho:a:f:c:b:v:r:",
                                    ['one_timestamp=', 'activity=', 'folder=',
                                     'model_file=', 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key in ['rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    print(parameters['folder'])
    print(parameters['model_file'])
    predictor = pr.ModelPredictor(parameters)
    print(predictor.acc)

if __name__ == "__main__":
    main(sys.argv[1:])
