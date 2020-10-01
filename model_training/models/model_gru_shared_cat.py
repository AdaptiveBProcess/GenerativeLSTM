# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import Dense, GRU, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
    print(args)
# =============================================================================
#     Input layer
# =============================================================================
    ac_input = Input(shape=(vec['prefixes']['activities'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['roles'].shape[1], ), name='rl_input')
    t_input = Input(shape=(vec['prefixes']['times'].shape[1],
                           vec['prefixes']['times'].shape[2]), name='t_input')

# =============================================================================
#    Embedding layer for categorical attributes
# =============================================================================
    ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
                             input_length=vec['prefixes']['activities'].shape[1],
                             trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                             rl_weights.shape[1],
                             weights=[rl_weights],
                             input_length=vec['prefixes']['roles'].shape[1],
                             trainable=False, name='rl_embedding')(rl_input)
# =============================================================================
#    Layer 1
# =============================================================================

    merged = Concatenate(name='concatenated', axis=2)([ac_embedding, rl_embedding])

    l1_c1 = GRU(args['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=args['imp'])(merged)

    l1_c3 = GRU(args['l_size'],
                 activation=args['lstm_act'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=args['imp'])(t_input)

# =============================================================================
#    Batch Normalization Layer
# =============================================================================
    batch1 = BatchNormalization()(l1_c1)
    batch3 = BatchNormalization()(l1_c3)

# =============================================================================
# The layer specialized in prediction
# =============================================================================
    l2_c1 = GRU(args['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=args['imp'])(batch1)

#   The layer specialized in role prediction
    l2_c2 = GRU(args['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=False,
                 dropout=0.2,
                 implementation=args['imp'])(batch1)

#   The layer specialized in role prediction
    l2_3 = GRU(args['l_size'],
                activation=args['lstm_act'],
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
        time_output = Dense(vec['next_evt']['times'].shape[1],
                            activation=args['dense_act'],
                            kernel_initializer='glorot_uniform',
                            name='time_output')(l2_3)
    else:
        time_output = Dense(vec['next_evt']['times'].shape[1],
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

    model.compile(loss={'act_output': 'categorical_crossentropy',
                        'role_output': 'categorical_crossentropy',
                        'time_output': 'mae'}, optimizer=opt)

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
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
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)

    batch_size = args['batch_size']
    model.fit({'ac_input': vec['prefixes']['activities'],
                'rl_input': vec['prefixes']['roles'],
                't_input': vec['prefixes']['times']},
              {'act_output': vec['next_evt']['activities'],
                'role_output': vec['next_evt']['roles'],
                'time_output': vec['next_evt']['times']},
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint,
                          lr_reducer, cb, clean_models],
              batch_size=batch_size,
              epochs=args['epochs'])

