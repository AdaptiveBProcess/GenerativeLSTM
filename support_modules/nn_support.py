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
def split_train_test(df, percentage, one_timestamp):
    cases = df.caseid.unique()
    num_test_cases = int(np.round(len(cases)*percentage))
    test_cases = cases[:num_test_cases]
    train_cases = cases[num_test_cases:]
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    for case in train_cases:
        df_train = df_train.append(df[df.caseid==case]) 
    if one_timestamp:
        df_train = df_train.sort_values('end_timestamp', ascending=True).reset_index(drop=True)
    else:
        df_train = df_train.sort_values('start_timestamp', ascending=True).reset_index(drop=True) 
    for case in test_cases:
        df_test = df_test.append(df[df.caseid==case])
    if one_timestamp:
        df_test = df_test.sort_values('end_timestamp', ascending=True).reset_index(drop=True)
    else:
        df_test = df_test.sort_values('start_timestamp', ascending=True).reset_index(drop=True) 
    return df_train, df_test 

# =============================================================================
# Split an event log in records format in folds of events 
# =============================================================================

def split_fold_events(data, num_folds):
    num_events = int(np.round(len(data)/num_folds))
    folds = list()
    for i in range(0, num_folds):
        sidx = i * num_events
        eidx = (i + 1) * num_events
        if i == 0:
            folds.append(data[:eidx])
        elif i == (num_folds - 1):
            folds.append(data[sidx:])
        else:
            folds.append(data[sidx:eidx])
    return folds

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

#def max_min_std(df, serie):
#    max_value, min_value = np.max(df[serie]), np.min(df[serie])
#    std = lambda x: (x[serie] - min_value) / (max_value - min_value)
#    df[serie+'_norm']=df.apply(std,axis=1)
#    return df, max_value, min_value
#
#
#def max_std(df, serie):
#    max_value, min_value = np.max(df[serie]), np.min(df[serie])
#    std = lambda x: x[serie] / max_value
#    df[serie+'_norm']=df.apply(std,axis=1)
#    return df, max_value, min_value

def max_min_de_std(val, max_value, min_value):
    true_value = (val * (max_value - min_value)) + min_value
    return true_value

def max_de_std(val, max_value, min_value):
    true_value = val * max_value 
    return true_value

def scale_feature(log, feature, method, replace=False):
    """Scales a number given a technique.
    Args:
        log: Event-log to be scaled.
        feature: Feature to be scaled.
        method: Scaling method max, lognorm, normal, per activity.
        replace (optional): replace the original value or keep both.
    Returns:
        Scaleded value between 0 and 1.
    """
    if method == 'lognorm':
        log[feature + '_log'] = np.log1p(log[feature])
        max_value = np.max(log[feature])
        log[feature + '_norm'] = np.divide(log[feature + '_log'], max_value) if max_value > 0 else 0 
        log = log.drop((feature + '_log'), axis=1)
    elif method == 'normal':
        max_value = np.max(log[feature])
        min_value = np.min(log[feature])
        log[feature + '_norm'] = np.divide(
                np.subtract(log[feature], min_value), (max_value - min_value))
    elif method == 'activity':
        max_values = pd.DataFrame(log.groupby(['task']).max()[feature]
                                    .reset_index(name = ('max_' + feature)))
        max_values = {row['task']:row['max_' + feature]  for _, row in max_values.iterrows()}
        norm = lambda x: np.divide(x[feature], max_values[x['task']]) if max_values[x['task']] > 0 else 0
        log[feature + '_norm'] = log.apply(norm, axis=1)
    else: 
        max_value = np.max(log[feature])
        log[feature + '_norm'] = np.divide(log[feature], max_value) if max_value > 0 else 0
    if replace:
        log = log.drop(feature, axis=1)
    return log

def feat_sel_eval_correlation(df, threshold, keep_cols=list()):
    is_correlated = True
    while is_correlated:
        correlation = df[df.columns.difference(keep_cols)].corr(method='pearson')
        correlation = correlation.stack().reset_index()
        correlation = correlation.rename(index=str, columns={'level_0': 'var1', 'level_1': 'var2', 0: 'value'})
        corr = dict()
        for x in correlation.to_dict('records'):
            if x['var1'] == x['var2']:
                continue
            if ((x['var1'], x['var2']) not in corr) and ((x['var2'], x['var1']) not in corr):
                corr[(x['var1'], x['var2'])] = x['value']
        correlation = pd.DataFrame([{'var1':k[0],'var2':k[1],'value':v} for k,v in corr.items() if v >= threshold])
        if len(correlation)>0:
            count1 = correlation.groupby(['var1']).count()['var2'].reset_index().rename(index=str, columns={'var1': 'var', 'var2': 'count'})
            count2 = correlation.groupby(['var2']).count()['var1'].reset_index().rename(index=str, columns={'var2': 'var', 'var1': 'count'})
            count1 = count1.merge(count2, on='var', how='outer')
            count1["Total"] = count1.sum(numeric_only=True, axis=1)
            maxValueIndexObj = count1["Total"].idxmax(axis=1)
            var = count1.iloc[maxValueIndexObj]['var']
            df = df.drop(columns=var)
        else:
            is_correlated = False
    return df

