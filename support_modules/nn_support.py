# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:53:16 2018
This module contains support functions specifically created to manipulate 
Event logs in pandas dataframe format
@author: Manuel Camargo
"""
import numpy as np
import pandas as pd

# =============================================================================
# Split an event log dataframe to peform split-validation 
# =============================================================================
def split_train_test(df, percentage):
    cases = df.caseid.unique()
    num_test_cases = int(np.round(len(cases)*percentage))
    test_cases = cases[:num_test_cases]
    train_cases = cases[num_test_cases:]
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    for case in train_cases:
        df_train = df_train.append(df[df.caseid==case]) 
    df_train = df_train.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
 
    for case in test_cases:
        df_test = df_test.append(df[df.caseid==case]) 
    df_test = df_test.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
    
    return df_train, df_test 


# =============================================================================
# Reduce the loops of a trace joining contiguous activities 
# exectuted by the same resource   
# =============================================================================
def reduce_loops(df):
    df_group = df.groupby('caseid')
    reduced = list()
    for name, group in df_group:
        temp_trace = list()
        group = group.sort_values('start_timestamp', ascending=True).reset_index(drop=True)
        temp_trace.append(dict(caseid=name, 
                          task=group.iloc[0].task, 
                          user=group.iloc[0].user, 
                          start_timestamp=group.iloc[0].start_timestamp, 
                          end_timestamp=group.iloc[0].end_timestamp, 
                          role=group.iloc[0].role))
        for i in range(1, len(group)):
            if group.iloc[i].task == temp_trace[-1]['task'] and group.iloc[i].user == temp_trace[-1]['user']:
                temp_trace[-1]['end_timestamp'] = group.iloc[i].end_timestamp
            else:
                temp_trace.append(dict(caseid=name, 
                                  task=group.iloc[i].task, 
                                  user=group.iloc[i].user, 
                                  start_timestamp=group.iloc[i].start_timestamp, 
                                  end_timestamp=group.iloc[i].end_timestamp, 
                                  role=group.iloc[i].role))
        reduced.extend(temp_trace)
    return pd.DataFrame.from_records(reduced) 

# =============================================================================
# Calculate duration and time between activities
# =============================================================================
def calculate_times(df):
   # Duration
   get_seconds = lambda x: x.seconds
   df['dur'] = (df.end_timestamp-df.start_timestamp).apply(get_seconds)
   # Time between activities per trace
   df['tbtw'] = 0
   # Multitasking time
   cases = df.caseid.unique()
   for case in cases:
       trace = df[df.caseid==case].sort_values('start_timestamp', ascending=True)
       for i in range(1,len(trace)):
           row_num = trace.iloc[i].name
           tbtw = (trace.iloc[i].start_timestamp - trace.iloc[i - 1].end_timestamp).seconds
           df.iloc[row_num,df.columns.get_loc('tbtw')] = tbtw
   return df, cases

# =============================================================================
# Standardization
# =============================================================================

def max_min_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: (x[serie] - min_value) / (max_value - min_value)
    df[serie+'_norm']=df.apply(std,axis=1)
    return df, max_value, min_value


def max_std(df, serie):
    max_value, min_value = np.max(df[serie]), np.min(df[serie])
    std = lambda x: x[serie] / max_value
    df[serie+'_norm']=df.apply(std,axis=1)
    return df, max_value, min_value

def max_min_de_std(val, max_value, min_value):
    true_value = (val * (max_value - min_value)) + min_value
    return true_value

def max_de_std(val, max_value, min_value):
    true_value = val * max_value 
    return true_value
