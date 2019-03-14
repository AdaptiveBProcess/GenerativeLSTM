# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def calculate_mae(log_data, simulation_data):
    log_data = log_data[['dur','task','tbtw','day_rel']]
    log_data = log_data.groupby('task')
    log_data = log_data.agg('median')
    log_data = log_data.reset_index()
    log_data.set_index(['task'])

    simulation_data = pd.DataFrame.from_records(simulation_data)
    simulation_data = simulation_data[['dur','task','tbtw','day_rel']]
    simulation_data = simulation_data.groupby('task')
    simulation_data = simulation_data.agg('median')
    simulation_data = simulation_data.reset_index()
    simulation_data.set_index(['task'])

    for column in ['dur','tbtw','day_rel']:
        simulation_data = simulation_data.rename(columns={column: column + '_sim'})

    values = log_data.merge(simulation_data, on='task', how='left')
    
    mae = dict()
    for column in ['dur','tbtw','day_rel']:
        values[column+'_err'] = np.abs(values[column+'_sim'] - values[column])
        mae[column+'_mae'] = np.sum(values[column+'_err'])/len(values[column+'_err'])
   
    return mae

def calculate_mae_onecolumn(log_data, simulation_data):
    
    log_data = log_data[['tbtw', 'task']]
    log_data = log_data.groupby('task')
    log_data = log_data.agg('mean')
    log_data = log_data.reset_index()
    log_data.set_index(['task'])

    simulation_data = pd.DataFrame.from_records(simulation_data)
    simulation_data = simulation_data[['tbtw', 'task']]
    simulation_data = simulation_data.groupby('task')
    simulation_data = simulation_data.agg('mean')
    simulation_data = simulation_data.reset_index()
    simulation_data.set_index(['task'])

    for column in ['tbtw']:
        simulation_data = simulation_data.rename(columns={column: column + '_sim'})

    values = log_data.merge(simulation_data, on='task', how='left')
    
    print(values)
    
    mae = dict()
    for column in ['tbtw']:
        values[column+'_err'] = np.abs(values[column+'_sim'] - values[column])
        mae[column+'_mae'] = np.sum(values[column+'_err'])/len(values[column+'_err'])
   
    return mae