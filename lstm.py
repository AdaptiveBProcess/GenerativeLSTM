# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import sys, getopt
import n_gram_emb_training as tr
import embedding_training as em
import predict_n_gram_emb as pr

# --setup--
def main(argv):
    """Main aplication method"""
    start_timeformat = '%Y-%m-%dT%H:%M:%S.%f'
    end_timeformat = '%Y-%m-%dT%H:%M:%S.%f'

    imp = 1
    lstm_act = None
    dense_act = None
    norm_method = 'max'
    model_type = ''
    folder = '20190228_155211821977'
    model_file = 'model_rd_100 Nadam_53-0.80.h5'
#    file_name = 'T_BPIC15_5.xes.gz'
#    file_name = 'BPI_Challenge_2012.xes'
    file_name = 'BPI_2012_W_complete.xes.gz'
#    file_name = 'Production.xes.gz'
#    file_name = 'Helpdesk.xes.gz'
    activity = 'predict'
    no_loops = False
    n_size=5
    l_size=100
    optim='Nadam'
    if argv:
        try:
            opts, args = getopt.getopt(argv,"hi:l:d:n:f:m:t:a:e:b:c:o:",["imp=","lstmact=","denseact=","norm=",'folder=','model=','mtype=','activity=','eventlog=','batchsize=','cellsize=','optimizer='])
        except getopt.GetoptError:
            print('test.py -i <implementation> -l <implementation> -d <implementation> -n <implementation>')
            sys.exit(2)
    
        for opt, arg in opts:
            if opt == '-h':
                print('test.py -i <implementation 1-cpu 2-gpu>')
                sys.exit()
            elif opt in ("-i", "--imp"):
                imp = arg
            elif opt in ("-l", "--lstmact"):
                lstm_act = arg
            elif opt in ("-d", "--denseact"):
                dense_act = arg
            elif opt in ("-n", "--norm"):
                norm_method = arg
            elif opt in ("-f", "--folder"):
                folder = arg
            elif opt in ("-m", "--model"):
                model_file = arg
            elif opt in ("-t", "--mtype"):
                model_type = arg
            elif opt in ("-a", "--activity"):
                activity = arg
            elif opt in ("-e", "--eventlog"):
                file_name = arg
            elif opt in ("-b", "--batchsize"):
                n_size = int(arg)
            elif opt in ("-c", "--cellsize"):
                l_size = int(arg)
            elif opt in ("-o", "--optimizer"):
                optim = arg

    if activity == 'emb':
        em.training_model(file_name, no_loops, start_timeformat, end_timeformat)
    elif activity == 'training':
        args = dict(imp=int(imp),lstm_act=lstm_act,
                    dense_act=dense_act,norm_method=norm_method,
                    model_type=model_type, n_size=n_size, l_size=l_size, optim=optim)
        print(args)
        tr.training_model(file_name, no_loops, start_timeformat, end_timeformat, args)
    elif activity == 'predict':
        print(folder)
        print(model_file)
        pr.predict(start_timeformat, folder, model_file, is_single_exec=False)
    

if __name__ == "__main__":
    main(sys.argv[1:])
