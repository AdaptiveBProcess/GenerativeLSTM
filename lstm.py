# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import sys
import getopt
import model_training as tr
import embedding_training as em
import predict_log as pr
import predict_suffix_full as px
import predict_next as nx



def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h':'help', '-i':'imp', '-l':'lstm_act',
              '-d':'dense_act', '-n':'norm_method', '-f':'folder',
              '-m':'model_file', '-t':'model_type', '-a':'activity',
              '-e':'file_name', '-b':'n_size', '-c':'l_size', '-o':'optim'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)

# --setup--
def main(argv):
    """Main aplication method"""
    timeformat = '%Y-%m-%dT%H:%M:%S.%f'
    parameters = dict()
#   Parameters setting manual fixed or catched by console for batch operations
    if not argv:
#       Type of LSTM task -> emb_training, training, pred_log, pred_sfx
        parameters['activity'] = 'predict_next'
#       General training parameters
        if parameters['activity'] in ['emb_training', 'training']:
            parameters['file_name'] = 'Helpdesk.xes.gz'
#           Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 1 # keras lstm implementation 1 cpu, 2 gpu
                parameters['lstm_act'] = None # optimization function see keras doc
                parameters['dense_act'] = None # optimization function see keras doc
                parameters['optim'] = 'Nadam' # optimization function see keras doc
                parameters['norm_method'] = 'lognorm' # max, lognorm
                # Model types --> specialized, concatenated, shared_cat, joint, shared
                parameters['model_type'] = 'shared_cat'
                parameters['n_size'] = 5 # n-gram size
                parameters['l_size'] = 100 # LSTM layer sizes
#       Generation parameters
        if parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20190524_163437607527'
            parameters['model_file'] = 'model_rd_100 Nadam_01-257024830.67.h5'

    else:
#       Catch parameters by console
        try:
            opts, _ = getopt.getopt(argv, "hi:l:d:n:f:m:t:a:e:b:c:o:",
                                    ["imp=", "lstmact=", "denseact=", "norm=",
                                     'folder=', 'model=', 'mtype=',
                                     'activity=', 'eventlog=', 'batchsize=',
                                     'cellsize=', 'optimizer='])
            for opt, arg in opts:
                parameters[catch_parameter(opt)] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)

#   Execution
    try:
        if parameters['activity'] == 'emb_training':
            if parameters['file_name'] == '' or not parameters['file_name']:
                raise Exception('The file name is missing...')
            print(parameters)
            em.training_model(parameters, timeformat)
        elif parameters['activity'] == 'training':
            print(parameters)
            tr.training_model(timeformat, parameters)
        elif parameters['activity'] == 'pred_log':
            print(parameters['folder'])
            print(parameters['model_file'])
            pr.predict(timeformat, parameters, is_single_exec=True)
        elif parameters['activity'] == 'predict_next':
            print(parameters['folder'])
            print(parameters['model_file'])
            nx.predict_next(timeformat, parameters, is_single_exec=False)
        elif parameters['activity'] == 'pred_sfx':
            print(parameters['folder'])
            print(parameters['model_file'])
            px.predict_suffix_full(timeformat, parameters, is_single_exec=False)
    except:
        raise Exception('Check the parameters structure...')


if __name__ == "__main__":
    main(sys.argv[1:])
