# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr
from intercase_feat import intercase_feat_extraction as itf


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size', '-e': 'splits', '-g': 'sub_group'}
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
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Event-log reading parameters
        parameters['one_timestamp'] = True  # Only one timestamp in the log
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next, inter_case
        parameters['activity'] = 'training'
        # General training parameters
        if parameters['activity'] in ['emb_training', 'training']:
            # Event-log parameters
            parameters['file_name'] = 'Helpdesk.xes'
            # Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 1  # keras lstm implementation 1 cpu,2 gpu
                parameters['lstm_act'] = 'relu'  # optimization function Keras
                parameters['dense_act'] = None  # optimization function Keras
                parameters['optim'] = 'Adam'  # optimization function Keras
                parameters['norm_method'] = 'max'  # max, lognorm
                # Model types --> shared_cat
                parameters['model_type'] = 'shared_cat'
                parameters['n_size'] = 10  # n-gram size
                parameters['l_size'] = 100  # LSTM layer sizes
                # Generation parameters
        elif parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20200505_181929922800'
            parameters['model_file'] = 'model_shared_cat_inter_72-0.64.h5'
            parameters['is_single_exec'] = False  # single or batch execution
            parameters['max_trace_size'] = 100
            # variants and repetitions to be tested
            parameters['variants'] = [{'imp': 'Random Choice', 'rep': 0},
                                      {'imp': 'Arg Max', 'rep': 1}]
        elif parameters['activity'] == 'inter_case':
            parameters['file_name'] = 'Helpdesk.xes'
            parameters['splits'] = 10
            parameters['sub_group'] = 'inter' # pd, rw, inter
        else:
            raise ValueError(parameters['activity']) 
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:e:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size=', 'splits=', 'sub_group='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size', 'max_trace_size', 'splits']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                              'ns_include': True}
            if parameters['activity'] in ['pred_log', 'pred_sfx',
                                          'predict_next']:
                # variants and repetitions to be tested
                parameters['variants'] = [{'imp': 'Random Choice', 'rep': 10},
                                          {'imp': 'Arg Max', 'rep': 0}]
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
#   Execution
    if parameters['activity'] == 'training':
        print(parameters)
        tr.ModelTrainer(parameters)
    elif parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        print(parameters['folder'])
        print(parameters['model_file'])
        pr.ModelPredictor(parameters)
    elif parameters['activity'] == 'inter_case':
        print(parameters)
        itf.extract_features(parameters)

if __name__ == "__main__":
    main(sys.argv[1:])
