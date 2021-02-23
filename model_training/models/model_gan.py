# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""

import os

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Concatenate, LeakyReLU
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy

from model_training.models import gan_trainer as gtrain

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

    print('Build discriminative model...')
    print(args)

    if args['optim'] == 'Nadam':
        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif args['optim'] == 'Adam':
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(learning_rate=0.01)

    discriminator = Discriminator(ac_weights, rl_weights, args)
    # discriminator.summary()
    
    generator = Generator(ac_weights, rl_weights, args)
    # generator.summary()
    # noise = tf.random.uniform(
    #         shape=(1, args['n_size']),
    #         minval=0, maxval=25,
    #         dtype=tf.int32)

    # generated_image = generator(noise, training=False)
    # decision = discriminator(generated_image)
    # print (decision)
    # gmodel = GAN(discriminator=discriminator,
    #              generator=generator,
    #              ngram=args['n_size'])

    gan = gtrain.GAN(discriminator=discriminator,
                      generator=generator,
                      ngram=args['n_size'])
    
    # # gan.compile(d_optimizer=opt,
    # #             g_optimizer=opt)
    
    batch_size = args['batch_size']
    gan.train(vec['training']['activities'], 2, batch_size=batch_size)
    # gan.fit({'ac_input': vec['training']['activities'],
    #           'rl_input': vec['training']['roles']},
    #         # {'dense_output_2': vec['training']['class']},
    #         {'act_output': vec['training']['class'],
    #           'role_output': vec['training']['class']},
    #         # validation_split=0.2,
    #           # verbose=2,
    #           batch_size=batch_size,
    #           epochs=args['epochs'])
    
    # early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    # cb = tc.TimingCallback(output_folder)
    # clean_models = cm.CleanSavedModelsCallback(output_folder, 2)

    # # Output file
    # output_file_path = os.path.join(output_folder,
    #                                 'model_' + str(args['model_type']) +
    #                                 '_{epoch:02d}-{val_loss:.2f}.h5')

    # # Saving
    # model_checkpoint = ModelCheckpoint(output_file_path,
    #                                     monitor='val_loss',
    #                                     verbose=0,
    #                                     save_best_only=True,
    #                                     save_weights_only=False,
    #                                     mode='auto')
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss',
    #                                 factor=0.5,
    #                                 patience=10,
    #                                 verbose=0,
    #                                 mode='auto',
    #                                 min_delta=0.0001,
    #                                 cooldown=0,
    #                                 min_lr=0)

    # batch_size = int(round(vec['activities'].shape[0]/20))
    # discriminator.fit({'ac_input': vec['activities'],
    #                    'rl_input': vec['roles']},
    #           {'dense_output_2': vec['class']},
    #           validation_split=0.2,
    #           verbose=2,
    #           callbacks=[early_stopping, model_checkpoint,
    #                       lr_reducer, cb, clean_models],
    #           batch_size=batch_size,
    #           shuffle=True,
    #           epochs=100)

# def create_discriminator(ac_weights, rl_weights, args):
#     print('Build discriminative model...')
#     ac_input = Input(shape=(ac_weights.shape[0], ), name='ac_input')
#     # rl_input = Input(shape=(rl_weights.shape[0], ), name='rl_input')
#     ac_embedding = Embedding(ac_weights.shape[0],
#                               ac_weights.shape[1],
#                               weights=[ac_weights],
#                               input_length=ac_weights.shape[0],
#                               trainable=False, name='ac_embedding')(ac_input)

#     # rl_embedding = Embedding(rl_weights.shape[0],
#     #                           rl_weights.shape[1],
#     #                           weights=[rl_weights],
#     #                           input_length=rl_weights.shape[0],
#     #                           trainable=False, name='rl_embedding')(rl_input)
#     # merged = Concatenate(name='concatenated', axis=1)([ac_embedding, rl_embedding])
#     l1_c1 = LSTM(args['l_size'],
#                   kernel_initializer='glorot_uniform',
#                   return_sequences=True,
#                   dropout=0.2,
#                   implementation=args['imp'])(ac_embedding)
#     batch1 = BatchNormalization()(l1_c1)
#     l2_c1 = LSTM(args['l_size'],
#                   kernel_initializer='glorot_uniform',
#                   return_sequences=False,
#                   dropout=0.2,
#                   implementation=args['imp'])(batch1)
#     l3_out1 = Dense(ac_weights.shape[0],
#                     activation='softmax',
#                     kernel_initializer='glorot_uniform',
#                     name='dense_output_1')(l2_c1)
#     l3_out2 = LeakyReLU(0.2)(l3_out1)
#     l3_out3 = Dense(units=1, activation='sigmoid',
#                     name='dense_output_2')(l3_out2)
#     model = Model(inputs=[ac_input],
#                   outputs=[l3_out3],
#                   name='discriminator')

   
#     return model


# def create_generator(ac_weights, rl_weights, args):
#     """Example function with types documented in the docstring.
#     Args:
#         param1 (int): The first parameter.
#         param2 (str): The second parameter.
#     Returns:
#         bool: The return value. True for success, False otherwise.
#     """

#     print('Build generative model...')
#     ac_serie = Input(shape=(args['n_size'], ), name='ac_serie')
#     # rl_serie = Input(shape=(args['n_size'], ), name='rl_serie')

#     ac_embedding = Embedding(ac_weights.shape[0],
#                               ac_weights.shape[1],
#                               weights=[ac_weights],
#                               input_length=args['n_size'],
#                               trainable=False, name='ac_embedding')(ac_serie)

#     # rl_embedding = Embedding(rl_weights.shape[0],
#     #                           rl_weights.shape[1],
#     #                           weights=[rl_weights],
#     #                           input_length=args['n_size'],
#     #                           trainable=False, name='rl_embedding')(rl_serie)

#     # merged = Concatenate(
#     #     name='concatenated', axis=2)([ac_embedding, rl_embedding])

#     l1_c1 = LSTM(args['l_size'],
#                   kernel_initializer='glorot_uniform',
#                   return_sequences=True,
#                   dropout=0.2,
#                   implementation=args['imp'])(ac_embedding)

#     batch1 = BatchNormalization()(l1_c1)

#     l2_c1 = LSTM(args['l_size'],
#                   kernel_initializer='glorot_uniform',
#                   return_sequences=False,
#                   dropout=0.2,
#                   implementation=args['imp'])(batch1)

# #   The layer specialized in role prediction
#     # l2_c2 = LSTM(args['l_size'],
#     #              kernel_initializer='glorot_uniform',
#     #              return_sequences=False,
#     #              dropout=0.2,
#     #              implementation=args['imp'])(batch1)

#     act_output = Dense(ac_weights.shape[0],
#                         activation='softmax',
#                         kernel_initializer='glorot_uniform',
#                         name='act_output')(l2_c1)

#     # role_output = Dense(rl_weights.shape[0],
#     #                     activation='softmax',
#     #                     kernel_initializer='glorot_uniform',
#     #                     name='role_output')(l2_c2)

#     # model = Model(inputs=[ac_input, rl_input],
#     #               outputs=[act_output, role_output])

#     model = Model(inputs=[ac_serie],
#                   outputs=[act_output],
#                   name='generator')
#     return model

class Discriminator(Model):
    def __init__(self, ac_weights, rl_weights, args):
        super(Discriminator, self).__init__()
        print('Build discriminative model...')
        self.model = Sequential(name='discriminator')
    
        self.model.add(Input(shape=(ac_weights.shape[0], ), name='ac_input'))
        self.model.add(Embedding(ac_weights.shape[0],
                                  ac_weights.shape[1],
                                  weights=[ac_weights],
                                  input_length=ac_weights.shape[0],
                                  trainable=False, name='ac_embedding'))
        self.model.add(LSTM(args['l_size'],
                      kernel_initializer='glorot_uniform',
                      return_sequences=True,
                      dropout=0.2,
                      implementation=args['imp']))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(args['l_size'],
                      kernel_initializer='glorot_uniform',
                      return_sequences=False,
                      dropout=0.2,
                      implementation=args['imp']))
        self.model.add(Dense(ac_weights.shape[0],
                        activation='softmax',
                        kernel_initializer='glorot_uniform',
                        name='dense_output_1'))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dense(units=1, activation='sigmoid',
                    name='dense_output_2'))
        self.model.summary()
        
    def call(self, x, training):
        return self.model(x, training)


class Generator(Model):
    def __init__(self, ac_weights, rl_weights, args):
        super(Generator, self).__init__()

        print('Build generative model...')
        self.model = Sequential(name='generator')
        self.model.add(Input(shape=(args['n_size'], ), name='ac_serie'))
        self.model.add(Embedding(ac_weights.shape[0],
                                  ac_weights.shape[1],
                                  weights=[ac_weights],
                                  input_length=args['n_size'],
                                  trainable=False, name='ac_embedding'))
    
        self.model.add(LSTM(args['l_size'],
                      kernel_initializer='glorot_uniform',
                      return_sequences=True,
                      dropout=0.2,
                      implementation=args['imp']))
    
        self.model.add(BatchNormalization())
    
        self.model.add(LSTM(args['l_size'],
                      kernel_initializer='glorot_uniform',
                      return_sequences=False,
                      dropout=0.2,
                      implementation=args['imp']))
    
        self.model.add(Dense(ac_weights.shape[0],
                            activation='softmax',
                            kernel_initializer='glorot_uniform',
                            name='act_output'))
        self.model.summary()
        
    def call(self, x, training):
        return self.model(x, training)