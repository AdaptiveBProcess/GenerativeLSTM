# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import sys
import getopt

from prediction import predict_log as pr
from prediction import predict_suffix_full as px
from prediction import predict_next as nx

from training import model_training as tr
from training import embedding_training as em


from support_modules.intercase_features import feature_engineering as fe


def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-n': 'norm_method', '-f': 'folder',
              '-m': 'model_file', '-t': 'model_type', '-a': 'activity',
              '-e': 'file_name', '-b': 'n_size', '-c': 'l_size',
              '-o': 'optim', '-s': 'one_timestamp'}
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
        # Type of LSTM task -> emb_training, training, pred_log
        # pred_sfx, predict_next, f_dist
        parameters['activity'] = 'training'
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
        if parameters['activity'] in ['pred_log', 'pred_sfx',
                                      'predict_next', 'pred_sfx2']:
            parameters['folder'] = '20191107_221318912825'
            parameters['model_file'] = 'model_seq2seq_101-0.91.h5'
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(argv, "hi:l:d:n:f:m:t:a:e:b:c:o:s:",
                                    ["imp=", "lstm_act=", "dense_act=",
                                     "norm_method=", 'folder=', 'model_file=',
                                     'model_type=', 'activity=', 'file_name=',
                                     'n_size=', 'l_size=', 'optim=',
                                     'one_timestamp='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if key == 'one_timestamp':
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                          'reorder': False}
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
        # tr.training_model(parameters)
    elif parameters['activity'] == 'pred_log':
        print(parameters['folder'])
        print(parameters['model_file'])
        pr.predict(parameters, is_single_exec=True)
    elif parameters['activity'] == 'predict_next':
        print(parameters['folder'])
        print(parameters['model_file'])
        nx.predict_next(parameters, is_single_exec=False)
    elif parameters['activity'] == 'pred_sfx':
        print(parameters['folder'])
        print(parameters['model_file'])
        px.predict_suffix_full(parameters, is_single_exec=False)
    elif parameters['activity'] == 'f_dist':
        print(parameters)
        fe.extract_features(parameters)


if __name__ == "__main__":
    main(sys.argv[1:])
