import pandas as pd
import networkx as nx
import os
import itertools
import json
import configparser
from glob import glob

def get_stats_log_traces(traces_gen_path):
    
    traces_files = glob(os.path.join(traces_gen_path, '*.csv'))
    if len(traces_files)>0:
        df_traces = pd.read_csv(traces_files[0])
        for traces_file in traces_files[1:]:
            df_trace_tmp = pd.read_csv(traces_file)
            df_traces = pd.concat([df_traces, df_trace_tmp])
    
        return df_traces, traces_files
    else:
        return None, []

def extract_rules():
    config = configparser.ConfigParser()
    config.read('rules.ini')

    settings = {}
    settings['path'] = config['RULES']['path']
    settings['variation'] = config['RULES']['variation'][0]
    settings['prop_variation'] = float(config['RULES']['variation'][1:])

    settings['act_paths'] = list(itertools.product(*[x if type(x) is list else [x] for x in [json.loads(x.strip()) for x in settings['path'].split('>>')] ]))
    return settings

def evaluate_condition(df_case, ac_index, act_paths):

    act_paths_idx = [(ac_index[x[0]], ac_index[x[1]]) for x in act_paths]
    df_case['rank'] = df_case.groupby('caseid')['start_timestamp'].rank().astype(int)
    df_case = df_case.sort_values(by='rank')
    u_tasks = [ac_index[x] for x in df_case['task'].drop_duplicates()]
    
    G = nx.DiGraph()
    for task in u_tasks:
        G.add_node(task)

    tasks = list(df_case['task'])
    if list(df_case['rank']) == list(set(list(df_case['rank']))):
        order = [(ac_index[x[0]], ac_index[x[1]]) for x in [(a, b) for a, b in zip(tasks[:-1], tasks[1:])]]
    else:
        order = []
        for i in range(1, len(df_case['rank'])):
            c_task = list(df_case[df_case['rank']==i]['task'])
            n_task = list(df_case[df_case['rank']==i+1]['task'])
            order += [(ac_index[x[0]], ac_index[x[1]]) for x in list(itertools.product(c_task, n_task))]

    G.add_edges_from(order)
    conds = [nx.is_simple_path(G, act_path) for act_path in act_paths_idx]

    return min(conds)

class GenerateStats:
    def __init__(self, log, ac_index, act_paths) -> None:
        self.log = log
        self.log['start_timestamp'] = pd.to_datetime(self.log['start_timestamp'])
        self.log['end_timestamp'] = pd.to_datetime(self.log['end_timestamp'])
        self.log['rank'] = self.log.groupby('caseid')['start_timestamp'].rank().astype(int)
        self.ac_index = ac_index
        self.act_paths = act_paths

    def get_stats(self):

        pos_cases = 0
        total_cases = len(self.log['caseid'].drop_duplicates())
        for caseid in self.log['caseid'].drop_duplicates():

            df_case = self.log[self.log['caseid']==caseid]
            cond = evaluate_condition(df_case, self.ac_index, self.act_paths)
            if cond:
                pos_cases += 1

        return pos_cases, total_cases