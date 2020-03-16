# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:52:26 2019

@author: Manuel Camargo
"""
import os
#import datetime

from keras.models import Model
from keras.optimizers import Nadam, Adam, SGD, Adagrad
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from support_modules.callbacks import time_callback as tc
from support_modules.callbacks import clean_models_callback as cm

def _training_model(vec, ac_weights, rl_weights, output_folder, args):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    print('Build model...')
# =============================================================================
#     Input layer
# =============================================================================
    ac_input = Input(shape=(vec['encoder_input_data']['activities'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['encoder_input_data']['roles'].shape[1], ), name='rl_input')
    t_input = Input(shape=(vec['encoder_input_data']['times'].shape[1], 1), name='t_input')

# =============================================================================
#    Embedding layer for categorical attributes        
# =============================================================================
    ac_embedding = Embedding(ac_weights.shape[0],
                            ac_weights.shape[1],
                            weights=[ac_weights],
                            input_length=vec['encoder_input_data']['activities'].shape[1],
                            trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                            rl_weights.shape[1],
                            weights=[rl_weights],
                            input_length=vec['encoder_input_data']['roles'].shape[1],
                            trainable=False, name='rl_embedding')(rl_input)

# =============================================================================
#    Encoder
# =============================================================================
    merged = Concatenate(name = 'concatenated', axis = 2)([ac_embedding, rl_embedding])

    l1_c1 = LSTM(args['l_size'],
                  kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  dropout=0.2,
                  implementation=args['imp'])(merged)
    
    l1_c3 = LSTM(args['l_size'],
                 activation=args['lstm_act'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=args['imp'])(t_input)

# =============================================================================
#    Decoder
# =============================================================================
    l2_c1 = LSTM(args['l_size'],
                    return_sequences=True,
                    implementation=args['imp'])(l1_c1)
  
#   The layer specialized in role prediction
    l2_c2 = LSTM(args['l_size'],
                    return_sequences=True,
                    dropout=0.2,
                    implementation=args['imp'])(l1_c1)

    l2_c3 = LSTM(args['l_size'],
                    activation=args['lstm_act'],
                    return_sequences=True,
                    dropout=0.2,
                    implementation=args['imp'])(l1_c3)

    
# =============================================================================
#    Output
# =============================================================================
    act_output = Dense(len(args['index_ac']), activation='softmax',
                       name='act_output')(l2_c1)
    rl_output = Dense(len(args['index_rl']), activation='softmax',
                       name='rl_output')(l2_c2)
    if ('dense_act' in args) and (args['dense_act'] is not None):
        time_output = Dense(1, activation=args['dense_act'],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_c3)
    else:
        time_output = Dense(1,
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_c3)

    
    model = Model(inputs=[ac_input, rl_input, t_input], outputs=[act_output, rl_output, time_output])

    if args['optim'] == 'Nadam':
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif args['optim'] == 'Adam':
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(learning_rate=0.01)

    model.compile(loss={'act_output':'categorical_crossentropy',
                        'rl_output':'categorical_crossentropy',
                        'time_output':'mae'}, metrics=['accuracy'], optimizer=opt)
    
    print(model.summary())    

    early_stopping = EarlyStopping(monitor='val_loss', patience=42)
    cb = tc.TimingCallback(output_folder)
    clean_models = cm.CleanSavedModelsCallback(output_folder, 2) 

    # Output file
    output_file_path = os.path.join(output_folder,
                                    'model_' + str(args['model_type']) +
                                    '_{epoch:02d}-{val_loss:.2f}.h5')
    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=40,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)
    
    batch_size = vec['encoder_input_data']['activities'].shape[1]
    model.fit({'ac_input':vec['encoder_input_data']['activities'],
               'rl_input':vec['encoder_input_data']['roles'],
               't_input':vec['encoder_input_data']['times']},
              {'act_output':vec['decoder_target_data']['activities'],
               'rl_output':vec['decoder_target_data']['roles'],
               'time_output':vec['decoder_target_data']['times']},
              batch_size=batch_size,
              epochs=500,
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer, cb, clean_models])