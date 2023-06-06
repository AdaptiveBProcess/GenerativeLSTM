import pandas as pd
import itertools

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import xml.etree.ElementTree as ET

class LogReplayerS:
    def __init__(self, bpmn_path, bpmn, model, log) -> None:

        self.root = ET.parse(bpmn_path).getroot()
        
        namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        self.sequence_flows = self.root.findall('.//bpmn:sequenceFlow', namespace)
        
        self.model = model
        self.log = log
        self.bpmn = bpmn
        self.m_data = pd.DataFrame.from_dict(dict(model.nodes.data()),
                                             orient='index')
        self.index_ac = self.m_data['name'].to_dict()
        self.df_log = pd.DataFrame(self.log.data)
        
        #log.data para extraer las trazas del log
        self.index_id = self.m_data['id'].to_dict()
        self.ac_index = {self.index_ac[key]:key for key in self.index_ac}
        self.id_ac = {value:self.index_ac[key] for key, value in self.index_id.items()}
        
        #Execute methods
        self.find_branches()
        self.count_branc_cases()

    #def get_initial_task(self, node_id):
    #    return [x.get('sourceRef') for x in self.sequence_flows if x.get('targetRef') == node_id]
    
    def get_initial_task(self, node_id):
        start_tasks = []
        for seq_flow in self.sequence_flows:
            if seq_flow.get('targetRef') == node_id:
                start_tasks.append(seq_flow.get('sourceRef'))
        return start_tasks

    def get_end_task(self, node_id):
        end_tasks = []
        for seq_flow in self.sequence_flows:
            if seq_flow.get('sourceRef') == node_id:
                end_tasks.append(seq_flow.get('targetRef'))
        return end_tasks

    def task_from_id(self, task):
        return self.id_ac[task]

    def get_task(self, tasks_id, task_type):
        tasks = []
        for task_id in tasks_id:
            t_id = self.task_from_id(task_id)
            if t_id != '':
                tasks.append(t_id)
            else:
                if task_type == 'start': 
                    tasks += [self.task_from_id(x) for x in self.get_initial_task(task_id)]
                elif task_type == 'end':
                    tasks += [self.task_from_id(x) for x in self.get_end_task(task_id)]

        return tasks

    def find_branches(self):
        namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        self.exclusive_gateways = [x.get('id') for x in self.root.findall('.//bpmn:exclusiveGateway', namespace) if x.get('gatewayDirection') == 'Diverging']

        self.branch_nodes = []

        for exclusive_gateway in self.exclusive_gateways:
            start_tasks_id = self.get_initial_task(exclusive_gateway)
            start_tasks = self.get_task(start_tasks_id, 'start')
            
            end_tasks_id = self.get_end_task(exclusive_gateway)
            end_tasks = self.get_task(end_tasks_id, 'end')

            self.branch_nodes += list(itertools.product(start_tasks, end_tasks))

    @staticmethod
    def evaluate_condition(activities, nodes):
        second_index = 0
        for activity in activities:
            if activity == nodes[second_index]:
                second_index += 1
                if second_index == len(nodes):
                    return 1
        return 0
    
    def count_branc_cases(self):
        print('Counting branch cases...')
        branching_probs_idx = {}
        self.df_log = self.df_log.sort_values(by=['caseid', 'start_timestamp'])
        cases = self.df_log.groupby('caseid')['task'].apply(list).to_dict()
        for case in tqdm(cases.keys()):
            seq = cases[case]
            for branch in self.branch_nodes:
                branching_probs_idx[branch] = branching_probs_idx.get(branch, 0) + self.evaluate_condition(seq, branch)

        self.branching_probs_tasks = branching_probs_idx
        self.branching_probs_idx = {(self.ac_index[key[0]], self.ac_index[key[1]]):value for key, value in self.branching_probs_tasks.items()}
        self.branching_probs = {}

        for key in self.branching_probs_idx.keys():
            gate = list(self.model.out_edges(key[0]))[0][1]
            self.branching_probs[self.index_id[gate]] = {self.index_id[k[1]]:v for k, v in self.branching_probs_idx.items() if k[0] == key[0]}

        new_branching_probs = {}
        for key in self.branching_probs.keys():
            branchings = {}
            for sequence_flow in self.sequence_flows:
                id_branch = sequence_flow.get('id')
                source_ref = sequence_flow.get('targetRef')
                for task_id in self.branching_probs[key].keys():
                    if task_id == source_ref:
                        if sum(self.branching_probs[key].values()) != 0:
                            branchings[id_branch] = round(self.branching_probs[key][task_id]/sum(self.branching_probs[key].values()), 2)
            new_branching_probs[key] = branchings
        
        self.branching_probs = new_branching_probs