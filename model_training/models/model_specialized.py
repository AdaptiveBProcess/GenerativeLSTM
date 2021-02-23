# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    from support_modules.callbacks import time_callback as tc
except:
    from importlib import util
    spec = util.spec_from_file_location(
        'time_callback', 
        os.path.join(os.getcwd(), 'support_modules', 'callbacks', 'time_callback.py'))
    tc = util.module_from_spec(spec)
    spec.loader.exec_module(tc)


def _training_model(train_vec, valdn_vec, ac_weights, rl_weights, output_folder, args, log_path=None):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    print('Build model...')
    print(args)
# =============================================================================
#     Input layer
# =============================================================================
    ac_input = Input(shape=(train_vec['prefixes']['activities'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(train_vec['prefixes']['roles'].shape[1], ), name='rl_input')
    t_input = Input(shape=(train_vec['prefixes']['times'].shape[1],
                           train_vec['prefixes']['times'].shape[2]), name='t_input')

# =============================================================================
#    Embedding layer for categorical attributes        
# =============================================================================
    ac_embedding = Embedding(ac_weights.shape[0],
                            ac_weights.shape[1],
                            weights=[ac_weights],
                            input_length=train_vec['prefixes']['activities'].shape[1],
                            trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                            rl_weights.shape[1],
                            weights=[rl_weights],
                            input_length=train_vec['prefixes']['roles'].shape[1],
                            trainable=False, name='rl_embedding')(rl_input)
# =============================================================================
#    Layer 1
# =============================================================================
    l1_c1 = LSTM(args['l_size'],
                  kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  dropout=0.2,
                  implementation=args['imp'])(ac_embedding)
    
    l1_c2 = LSTM(args['l_size'],
                  kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  dropout=0.2,
                  implementation=args['imp'])(rl_embedding)

    l1_c3 = LSTM(args['l_size'],
                 activation=args['lstm_act'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=args['imp'])(t_input)

# =============================================================================
#    Batch Normalization Layer
# =============================================================================
    batch1 = BatchNormalization()(l1_c1)
    batch2 = BatchNormalization()(l1_c2)
    batch3 = BatchNormalization()(l1_c3)
    
# =============================================================================
# The layer specialized in prediction
# =============================================================================
    l2_c1 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch1)
 
#   The layer specialized in role prediction
    l2_c2 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch2)
    
#   The layer specialized in role prediction
    if args['lstm_act'] is not None:
        l2_3 = LSTM(args['l_size'],
                    activation=args['lstm_act'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch3)
    else:
        l2_3 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch3)
    

    
# =============================================================================
# Output Layer
# =============================================================================
    act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(l2_c1)

    role_output = Dense(rl_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='role_output')(l2_c2)

    if ('dense_act' in args) and (args['dense_act'] is not None):
        time_output = Dense(train_vec['next_evt']['times'].shape[1], activation=args['dense_act'],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    else:
        time_output = Dense(train_vec['next_evt']['times'].shape[1],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)

    model = Model(inputs=[ac_input, rl_input, t_input],
                  outputs=[act_output, role_output, time_output])

    if args['optim'] == 'Nadam':
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif args['optim'] == 'Adam':
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(learning_rate=0.01)

    model.compile(loss={'act_output':'categorical_crossentropy',
                        'role_output':'categorical_crossentropy',
                        'time_output':'mae'}, optimizer=opt)
    
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    if log_path:
        cb = tc.TimingCallback(output_folder, log_path=log_path)
    else:
        cb = tc.TimingCallback(output_folder)

    # Output file
    output_file_path = os.path.join(output_folder, 
                                    os.path.splitext(args['file'])[0]+'.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)

    batch_size = args['batch_size']
    model.fit({'ac_input': train_vec['prefixes']['activities'],
                'rl_input': train_vec['prefixes']['roles'],
                't_input': train_vec['prefixes']['times']},
              {'act_output': train_vec['next_evt']['activities'],
                'role_output': train_vec['next_evt']['roles'],
                'time_output': train_vec['next_evt']['times']},
              validation_data=(
                  {'ac_input': valdn_vec['prefixes']['activities'],
                   'rl_input': valdn_vec['prefixes']['roles'],
                   't_input': valdn_vec['prefixes']['times']},
                  {'act_output': valdn_vec['next_evt']['activities'],
                   'role_output': valdn_vec['next_evt']['roles'],
                   'time_output': valdn_vec['next_evt']['times']}),
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer, cb],
              batch_size=batch_size,
              epochs=args['epochs'])
    return model
