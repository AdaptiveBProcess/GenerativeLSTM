import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import subprocess
import os

class MergeModels:
    def __init__(self, settings) -> None:

        self.settings = settings

        self.load_structures()
        self.update_branching_probs()
        self.update_tasks()
        self.save_model()
        self.simulate()

    def load_structures(self):
        
        self.asis_tree = ET.parse(self.settings['asis_bpmn_path'])
        self.asis_root = self.asis_tree.getroot()

        self.tobe_tree = ET.parse(self.settings['tobe_bpmn_path'])
        self.tobe_root = self.tobe_tree.getroot()
                
        namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

        asis_tasks = self.asis_root.findall('.//bpmn:task', namespace)
        self.asis_tasks_data = {task.get('id'):task.get('name') for task in asis_tasks}

        tobe_tasks = self.tobe_root.findall('.//bpmn:task', namespace)
        self.tobe_tasks_data = {task.get('name'):task.get('id') for task in tobe_tasks}

        #Load information from AS-IS model
        with open(self.settings['asis_bpmn_path'], 'r') as f:
            asis_content = f.read()

        self.asis_bpmn_xml = BeautifulSoup(asis_content, "xml")
        self.asis_sim_info = self.asis_bpmn_xml.find_all('qbp:processSimulationInfo')[0]

    def update_tasks(self):

        asis_elements = self.asis_sim_info.findAll('element')

        for element in asis_elements:
            element_id_asis = element.get('elementId')
            task_asis = self.asis_tasks_data[element_id_asis]
            if task_asis in self.tobe_tasks_data.keys():
                element_id_tobe = self.tobe_tasks_data[task_asis]
                element['elementId'] = element_id_tobe

        self.tobe_root.append(ET.fromstring(str(self.asis_sim_info)))

    @staticmethod
    def change_branch_node(sequence_flows, key_node, key_task):
        def get_initial_task(sequence_flows, node_id):
            start_tasks = []
            for seq_flow in sequence_flows:
                if seq_flow.get('targetRef') == node_id:
                    start_tasks.append(seq_flow.get('sourceRef'))
            return start_tasks
        
        f = None  # Initialize f outside the loop to avoid potential NameError
        
        for seq_flow in sequence_flows:
            if seq_flow.get('id') == key_task:
                source_ref = seq_flow.get('sourceRef')
                st = [x for x in get_initial_task(sequence_flows, source_ref) if x == key_node]
                if st:
                    st = st[0]
                    f = [x.get('id') for x in sequence_flows if x.get('sourceRef') == st and x.get('targetRef') == source_ref]
                    if f:
                        f = f[0]
                    else:
                        f = None
                    break        
        return f   

    def update_branching_probs(self):

        #Delete section
        sequence_flows = self.asis_sim_info.find('qbp:sequenceFlows')
        sequence_flows.decompose()

        #Add new section
        new_sequence_flows = self.asis_bpmn_xml.new_tag('qbp:sequenceFlows')

        for key_node in self.settings['lrs'].branching_probs.keys():
            for key_task, value_task in self.settings['lrs'].branching_probs[key_node].items():
                new_branch_node = self.change_branch_node(self.settings['lrs'].sequence_flows, key_node, key_task)
                if new_branch_node is None:
                    new_seq_flow = self.asis_bpmn_xml.new_tag('qbp:sequenceFlow', elementId=key_task, executionProbability=value_task)
                else:
                    new_seq_flow = self.asis_bpmn_xml.new_tag('qbp:sequenceFlow', elementId=new_branch_node, executionProbability=value_task)
                new_sequence_flows.append(new_seq_flow)

        self.asis_sim_info.append(new_sequence_flows)
        
    def simulate(self):
        args = ['java', '-jar', self.settings['bimp_path'], self.settings['output_path'], '-csv', self.settings['csv_output_path']]
        result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Simulation was successfully executed")
        elif result.returncode == 1:
            execption_output = [result.stdout.split('\n')[i-1] for i in range(len(result.stdout.split('\n'))) if 'BPSimulatorException' in result.stdout.split('\n')[i]]
            print("Execution failed :", ' '.join(execption_output))
        

    def save_model(self):
        # Save the modified BPMN file
        self.tobe_tree.write(self.settings['output_path'], encoding='utf-8', xml_declaration=True)