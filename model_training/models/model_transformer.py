# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:51:29 2020

@author: Mauricio Díaz
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
import os
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from support_modules.callbacks import time_callback as tc
from support_modules.callbacks import clean_models_callback as cm

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({'embed_dim': self.embed_dim,
                       'num_heads': self.num_heads,
                       'projection_dim': self.projection_dim,
                       'query_dense': self.query_dense,
                       'key_dense': self.key_dense,
                       'value_dense': self.value_dense,
                       'combine_heads': self.combine_heads})
        
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({'att': self.att,
                       'ffn': self.ffn,
                       'layernorm1': self.layernorm1,
                       'layernorm2': self.layernorm2,
                       'dropout1': self.dropout1,
                       'dropout2': self.dropout2})
        
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
    
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self,maxlen, vocab_size, embed_dim, weight, inputL, train, nam, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, 
                                   output_dim=embed_dim,
                                   weights=weight,
                                   input_length=inputL,
                                   trainable=train)
        self.pos_emb = Embedding(input_dim=maxlen, 
                                 output_dim=embed_dim,
                                 trainable=train)
        
        

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({'token_emb': self.token_emb,
                       'pos_emb': self.pos_emb})
        
        return config
    
    @classmethod
    def from_config(cls,config):
        return cls(**config)
    

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
    maxlen = ac_weights.shape[1] + 1
    if (args['file_name'] == 'BPI_Challenge_2013_closed_problems.xes'
        or args['file_name'] == 'BPI_2012_W_complete.xes'):
        maxlen += 1
        
    ac_embedding = TokenAndPositionEmbedding(maxlen, 
                                             ac_weights.shape[0], 
                                             ac_weights.shape[1], 
                                             [ac_weights],
                                             vec['prefixes']['activities'].shape[1],
                                             False,
                                             'ac_embedding')(ac_input)

    maxlen = rl_weights.shape[1] + 1
    if (args['file_name'] == 'BPI_Challenge_2013_closed_problems.xes'
        or args['file_name'] == 'BPI_2012_W_complete.xes'):
        maxlen += 1
        
    rl_embedding = TokenAndPositionEmbedding(maxlen,
                                             rl_weights.shape[0],
                                             rl_weights.shape[1],
                                             [rl_weights],
                                             vec['prefixes']['roles'].shape[1],
                                             False,
                                             'rl_embedding')(rl_input)
    
# =============================================================================
#
# =============================================================================
    
    transformer_block_Act = TransformerBlock(ac_weights.shape[1],
                                             1,
                                             ac_weights.shape[1])
    transformer_block_Rol = TransformerBlock(rl_weights.shape[1],
                                             1,
                                             rl_weights.shape[1])
    transformer_block_time = TransformerBlock(vec['prefixes']['times'].shape[1],
                                              1,
                                              vec['prefixes']['times'].shape[1])
    
    xAct = transformer_block_Act(ac_embedding)
    xRol = transformer_block_Rol(rl_embedding)
    xTime = transformer_block_time(t_input)

# =============================================================================
#    
# =============================================================================
    
    layer = layers.GlobalAveragePooling1D()
    xAct = layer(xAct)
    xRol = layer(xRol)
    xTime = layer(xTime)
    
# =============================================================================
# 
# =============================================================================
    
    xAct = layers.Dropout(0.1)(xAct)
    xRol = layers.Dropout(0.1)(xRol)
    xTime = layers.Dropout(0.1)(xTime)
    
# =============================================================================
# 
# =============================================================================
    
    xAct = Dense(ac_weights.shape[0], activation="relu")(xAct)
    xRol = Dense(rl_weights.shape[0], activation="relu")(xRol)
    xTime = Dense(vec['next_evt']['times'].shape[1], activation="relu")(xTime)
    
# =============================================================================
# 
# =============================================================================
    
    xAct = layers.Dropout(0.1)(xAct)
    xRol = layers.Dropout(0.1)(xRol)
    xTime = layers.Dropout(0.1)(xTime)
    
# =============================================================================
# Output Layer
# =============================================================================
    act_output = Dense(ac_weights.shape[0], activation='softmax', name='act_output')(xAct)

    role_output = Dense(rl_weights.shape[0], activation='softmax', name='role_output')(xRol)
    
    time_output = Dense(vec['next_evt']['times'].shape[1], activation='softmax', name='time_output')(xTime)


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
                        'time_output': 'mae'}, optimizer=opt, metrics=['accuracy'])

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
    
    
    name = 'transf_model'
    model.save(output_folder+'/'+name)
    
    
    reconstructed_model = keras.models.load_model(output_folder+"/transf_model")
    
    flag = True
    cont = 0
    while cont < len(model.layers):
        x = model.layers[cont]
        y = reconstructed_model.layers[cont]
        cont2 = 0
        while cont2 < len(x.get_weights()):
            a = x.get_weights()[cont2]
            b = y.get_weights()[cont2]
            if not np.array_equal(a,b):
                flag = False
                break
            
            cont2+=1
        cont+=1
    print("Model saved correctly:",flag)
    
            