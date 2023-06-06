import warnings
warnings.filterwarnings('ignore')

from support_modules.log_replayer_stochastic import LogReplayerS
from readers.process_structure import create_process_structure
import readers.log_reader as lr
import readers.bpmn_reader as br

from bs4 import BeautifulSoup
import platform as pl
import subprocess
import os

class StochasticModel:
    def __init__(self, settings):
        self.settings = settings

        self.load_structures()
        self.extract_stochastic_model()
        
    def load_structures(self):
        self.log = lr.LogReader(self.settings['log_path'], self.settings)
        self._sm3_miner()
        self.bpmn = br.BpmnReader(self.settings['tobe_bpmn_path'])
        self.model = create_process_structure(self.bpmn)

    def extract_stochastic_model(self):
        self.lrs = LogReplayerS(self.settings['tobe_bpmn_path'], self.bpmn, self.model, self.log)

    def change_branch_node(self, key_node, key_task):
        for seq_flow in self.lrs.sequence_flows:
            if seq_flow.get('id') == key_task:
                source_ref = seq_flow.get('sourceRef')
                st = [x for x in self.lrs.get_initial_task(source_ref) if x == key_node][0]
                f = [x.get('id') for x in self.lrs.sequence_flows if x.get('sourceRef') == st and x.get('targetRef') == source_ref][0]
        return f
    
    def _sm3_miner(self):

        print(" -- Mining Process Structure --")
        # Event log file_name
        sep = ';' if pl.system().lower() == 'windows' else ':'
        # Mining structure definition
        args = ['java']
        if pl.system().lower() != 'windows':
            args.extend(['-Xmx2G', '-Xss8G'])
        args.extend(['-cp',
                        (self.settings['sm3_path']+sep+os.path.join(
                            'external_tools','splitminer3','lib','*')),
                        'au.edu.unimelb.services.ServiceProvider',
                        'SMD',
                        str(self.settings['epsilon']), str(self.settings['eta']),
                        'false', 'false', 'false',
                        self.settings['log_path'],
                        self.settings['tobe_bpmn_path'].replace('.bpmn', '')])
        subprocess.call(args)