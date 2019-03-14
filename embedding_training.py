# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
from support_modules.readers import log_reader as lr
from support_modules import role_discovery as rl
import support as sup

import pandas as pd
import os
import nn_support as nsup
import random
import numpy as np
import itertools
import math

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape

def training_model(file_name, no_loops, start_timeformat, end_timeformat):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    input_file_path = os.path.join('input_files', file_name)
    log = lr.LogReader(input_file_path, start_timeformat, end_timeformat, one_timestamp=True)
    _, resource_table = rl.read_resource_pool(log, sim_percentage=0.50)
    # Role discovery
    log_df_resources = pd.DataFrame.from_records(resource_table)
    log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})
    # Dataframe creation
    log_df = pd.DataFrame.from_records(log.data)
    log_df = log_df.merge(log_df_resources, on='user', how='left')
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)

    if no_loops:
        log_df = nsup.reduce_loops(log_df)
    # Index creation
    ac_index = create_index(log_df, 'task')
    ac_index['start'] = 0
    ac_index['end'] = len(ac_index)
    index_ac = {v: k for k, v in ac_index.items()}
    
    rl_index = create_index(log_df, 'role')
    rl_index['start'] = 0
    rl_index['end'] = len(rl_index)
    index_rl = {v: k for k, v in rl_index.items()}
    
    categories = itertools.product(*[list(ac_index.items()), list(rl_index.items())])

    # Define the number of dimensions as the 4th root of the number of categories
    dim_number = math.ceil(len(list(categories))**0.25)

    ac_weights, rl_weights = train_embedded(log_df, ac_index, rl_index, dim_number)
    
    output_file = os.path.join('input_files', 'embedded_matix')
    sup.create_file_from_list(reformat_matrix(index_ac, ac_weights),
                              os.path.join(output_file, 'ac_'+ file_name.split('.')[0]+'.emb'))
    sup.create_file_from_list(reformat_matrix(index_rl, rl_weights),
                              os.path.join(output_file, 'rl_'+ file_name.split('.')[0]+'.emb'))


# =============================================================================
# Pre-processing: embedded dimension
# =============================================================================

def train_embedded(log_df, ac_index, rl_index, dim_number):
    # Iterate through each book
    pairs = list()
    for i in range(0,len(log_df)):
        # Iterate through the links in the book
        pairs.append((ac_index[log_df.iloc[i]['task']], rl_index[log_df.iloc[i]['role']]))

    model = ac_rl_embedding_model(ac_index, rl_index, dim_number)
    model.summary()

    n_positive = 1024
    gen = generate_batch(pairs, ac_index, rl_index, n_positive, negative_ratio = 2)
    
    # Train
    model.fit_generator(gen, epochs = 100, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)


    # Extract embeddings
    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')

    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]

    return ac_weights, rl_weights


def generate_batch(pairs, ac_index, rl_index, n_positive = 50, negative_ratio = 1.0, classification = True):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)
    activities = list(ac_index.keys())
    role = list(rl_index.keys())
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (ac_index, rl_index) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (ac_index, rl_index, 1)
        # Increment idx by 1
        idx += 1
       
        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = random.randrange(len(activities))
            random_rl = random.randrange(len(role))
            
            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_ac, random_rl, neg_label)
                idx += 1
             
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1]}, batch[:, 2]


def ac_rl_embedding_model(ac_index, rl_index, embedding_size):
    """Model to embed activities and roles using the functional API"""
    
    # Both inputs are 1-dimensional
    activity = Input(name = 'activity', shape = [1])
    role = Input(name = 'role', shape = [1])
    
    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name = 'activity_embedding',
                               input_dim = len(ac_index),
                               output_dim = embedding_size)(activity)
    
    # Embedding the role (shape will be (None, 1, embedding_size))
    role_embedding = Embedding(name = 'role_embedding',
                               input_dim = len(rl_index),
                               output_dim = embedding_size)(role)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([activity_embedding, role_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # Otherwise loss function is mean squared error
    model = Model(inputs = [activity, role], outputs = merged)
    model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# =============================================================================
# Support
# =============================================================================
def reformat_matrix(index, weigths):
    matrix = list()
    for i, _ in enumerate(index):
        data = [i, index[i]]
        data.extend(weigths[i])
        matrix.append(data)
    return matrix

def create_index(df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    subsec_set = set()
    temp_list = df[[column]].values.tolist()
    [subsec_set.add((x[0])) for x in temp_list]
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i in range(0, len(subsec_set)):
        alias[subsec_set[i]] = i + 1
    return alias