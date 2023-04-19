import pandas as pd
import networkx as nx
import os
import itertools
import json

class GenerateStats:
    def __init__(self, log, ac_index, ac_paths) -> None:
        self.log = log
        self.log['start_timestamp'] = pd.to_datetime(self.log['start_timestamp'])
        self.log['end_timestamp'] = pd.to_datetime(self.log['end_timestamp'])
        self.log['rank'] = self.log.groupby('caseid')['start_timestamp'].rank().astype(int)
        self.ac_index = ac_index
        self.ac_paths = ac_paths
        self.act_paths_idx = [(ac_index[x[0]], ac_index[x[1]]) for x in self.act_paths]

    def evaluate_condition(self, df_case):
        df_case = df_case.sort_values(by='rank')
        u_tasks = [self.ac_index[x] for x in df_case['task'].drop_duplicates()]
        
        G = nx.DiGraph()
        for task in u_tasks:
            G.add_node(task)

        tasks = list(df_case['task'])
        if list(df_case['rank']) == list(set(list(df_case['rank']))):
            order = [(self.ac_index[x[0]], self.ac_index[x[1]]) for x in [(a, b) for a, b in zip(tasks[:-1], tasks[1:])]]
        else:
            order = []
            for i in range(1, len(df_case['rank'])):
                c_task = list(df_case[df_case['rank']==i]['task'])
                n_task = list(df_case[df_case['rank']==i+1]['task'])
                order += [(self.ac_index[x[0]], self.ac_index[x[1]]) for x in list(itertools.product(c_task, n_task))]

        G.add_edges_from(order)
        conds = [nx.is_simple_path(G, act_path) for act_path in self.act_paths_idx]

        return min(conds)

    def get_stats(self):

        pos_cases = 0
        total_cases = len(self.log['caseid'].drop_duplicates())
        for caseid in self.log['caseid'].drop_duplicates():

            df_case = self.log[self.log['caseid']==caseid]
            cond = self.evaluate_condition(df_case)
            if cond:
                pos_cases += 1

        return pos_cases, total_cases