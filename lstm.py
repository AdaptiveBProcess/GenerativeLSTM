# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr
from model_training import embedding_training as em


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size'}
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
    # variants and repetitions to be tested
    parameters['variants'] = [{'imp': 'Random Choice', 'rep': 1},
                              {'imp': 'Arg Max', 'rep': 1}]
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Event-log reading parameters
        parameters['one_timestamp'] = True  # Only one timestamp in the log
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # Type of LSTM task -> emb_training, training, pred_log
        # pred_sfx, predict_next, f_dist
        parameters['activity'] = 'pred_sfx'
        # General training parameters
        if parameters['activity'] in ['emb_training', 'training', 'f_dist']:
            # Event-log parameters
            parameters['file_name'] = 'BPI_Challenge_2013_closed_problems.xes'
            # Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 1  # keras lstm implementation 1 cpu,2 gpu
                parameters['lstm_act'] = 'relu'  # optimization function Keras
                parameters['dense_act'] = None  # optimization function Keras
                parameters['optim'] = 'Adam'  # optimization function Keras
                parameters['norm_method'] = 'lognorm'  # max, lognorm
                # Model types --> shared_cat, shared_cat_inter,
                # seq2seq, seq2seq_inter
                parameters['model_type'] = 'shared_cat'
                parameters['n_size'] = 5  # n-gram size
                parameters['l_size'] = 100  # LSTM layer sizes
                # Generation parameters
        if parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20200327_185514220611'
            parameters['model_file'] = 'model_shared_cat_99-1.00.h5'
            parameters['is_single_exec'] = True  # single or batch execution
            parameters['max_trace_size'] = 100

    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size', 'max_trace_size']:
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
#   Execution
    if parameters['activity'] == 'emb_training':
        if parameters['file_name'] == '' or not parameters['file_name']:
            raise Exception('The file name is missing...')
        print(parameters)
        em.training_model(parameters)
    elif parameters['activity'] == 'training':
        print(parameters)
        tr.ModelTrainer(parameters)
    elif parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        print(parameters['folder'])
        print(parameters['model_file'])
        pr.ModelPredictor(parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
