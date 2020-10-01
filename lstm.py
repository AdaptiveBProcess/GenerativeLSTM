# -*- coding: utf-8 -*-
"""
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
    parameters['one_timestamp'] = True  # Only one timestamp in the log
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32 # Usually 32/64/128/256
    parameters['epochs'] = 2
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next
        parameters['activity'] = 'pred_sfx'
        # Event-log reading parameters
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # General training parameters
        if parameters['activity'] in ['training']:
            # Event-log parameters
            parameters['file_name'] = 'BPI_Challenge_2013_closed_problems.xes'
            # Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 1  # keras lstm implementation 1 cpu,2 gpu
                parameters['lstm_act'] = 'relu'  # optimization function Keras
                parameters['dense_act'] = None  # optimization function Keras
                parameters['optim'] = 'Adam'  # optimization function Keras
                parameters['norm_method'] = 'max'  # max, lognorm
                # Model types --> shared_cat, specialized, concatenated, 
                # shared_cat_gru, specialized_gru, concatenated_gru
                parameters['model_type'] = 'concatenated_gru'
                parameters['n_size'] = 5  # n-gram size
                parameters['l_size'] = 50  # LSTM layer sizes
                # Generation parameters
        elif parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20201001_87C9DEA1_5CB9_4B9B_AA5C_45BD014F833C'
            parameters['model_file'] = 'model_concatenated_gru_02-2.39.h5'
            parameters['is_single_exec'] = False  # single or batch execution
            # variants and repetitions to be tested random_choice, arg_max
            parameters['variant'] = 'random_choice'
            parameters['rep'] = 1
        else:
            raise ValueError(parameters['activity'])
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
#   Execution
    if parameters['activity'] == 'training':
        print(parameters)
        trainer = tr.ModelTrainer(parameters)
        print(trainer.output, trainer.model, sep=' ')
    elif parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        print(parameters)
        print(parameters['folder'])
        print(parameters['model_file'])
        predictor = pr.ModelPredictor(parameters)
        print(predictor.acc)
if __name__ == "__main__":
    main(sys.argv[1:])
