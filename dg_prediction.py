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
import support_functions as sf

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
    parameters['include_org_log'] = False
    parameters['read_options'] = {
        'timeformat': '%Y-%m-%d %H:%M:%S',
        'column_names': column_names,
        'one_timestamp': parameters['one_timestamp'],
        'filter_d_attrib': False}
    # Parameters settled manually or catched by console for batch operations
    if not argv:
        # predict_next, pred_sfx
        parameters['activity'] = 'pred_log'
        parameters['folder'] = '20230508_8A2DA06E_2FB3_43B8_AD6E_7AAE2143FF14'
        parameters['model_file'] = 'Production.h5'
        parameters['log_name'] = parameters['model_file'].split('.')[0]
        parameters['is_single_exec'] = False  # single or batch execution
        # variants and repetitions to be tested Random Choice, Arg Max, Rules Based Random Choice, Rules Based Arg Max
        parameters['variant'] = 'Rules Based Random Choice'
        parameters['rep'] = 1
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
    pr.ModelPredictor(parameters)

if __name__ == "__main__":
    main(sys.argv[1:])
