import subprocess
import os
import pandas as pd
from bs4 import BeautifulSoup

def call_simod(log_name):
    print('Executing Simod with {} log.'.format(log_name))
    if not 'Simod-Coral-Version' in os.getcwd():
        os.chdir("Simod-Coral-Version")
    subprocess.run(["python", "simod_optimizer.py", "-f", log_name, "-m", "sm3"])
    print('Execution ended.')

def combine_models(model_asis_path, model_tobe_path, rules_model_path):

    with open(model_asis_path, 'r') as f:
        data_asis = f.read()

    bpmn_asis = BeautifulSoup(data_asis, "xml") 
    sim_info_asis = str(bpmn_asis.find_all('qbp:processSimulationInfo')[0])

    with open(model_tobe_path, 'r') as f:
        data_tobe = f.read()

    bpmn_tobe = BeautifulSoup(data_tobe, "xml") 
    sim_info_tobe = str(bpmn_tobe.find_all('qbp:processSimulationInfo')[0])
    att_info_tobe = data_asis.replace(sim_info_tobe, '')

    bpmn_rules_based = '{}\n{}'.format(att_info_tobe, sim_info_asis)

    with open(rules_model_path, 'w') as file:
        file.write(bpmn_rules_based)

def extract_text(model_path, ptt_s, ptt_e):
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    start, end = None, None
    for idx, line in enumerate(lines):
        if ptt_s in line and start == None:
            start = idx
        if ptt_e in line and end == None:
            end = idx
        if start != None and end != None:
            break
    return '\n'.join(lines[start+1:end])

def standarize_metric(value, kpi):
    if 'cost' not in kpi.lower():
        if (value <= 60*1.5):
            return value, 'seconds'
        elif (value > 60*1.5) and (value <= 60*60*1.5):
            return value/(60), 'minutes'
        elif (value > 60*60*1.5) and (value <= 60*60*24*1.5):
            return value/(60*60), 'hours'
        elif (value > 60*60*24*1.5) and (value <= 60*60*24*7*1.5):
            return value/(60*60*24), 'days'
        elif (value > 60*60*24*7*1.5):
            return value/(60*60*24*7), 'weeks'
    else:
        return value, ''

def execute_simulator_simple(bimp_path, model_path, csv_output_path):
    args = ['java', '-jar', bimp_path, model_path, '-csv', csv_output_path]
    subprocess.run(args, stdout=open(os.devnull, 'wb'))

def return_message_stats(stats_path, scenario_name):

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    ptt_s = 'Scenario statistics'
    ptt_e = 'Process Cycle Time (s) distribution' 
    text = extract_text(stats_path, ptt_s, ptt_e)

    data = [x.split(',') for x in text.split('\n') if x != '']
    df = pd.DataFrame(data = data[1:], columns=data[0])

    df = df[df['KPI']== 'Process Cycle Time (s)']
    df['Average'] = df['Average'].astype(float).astype(str).apply(lambda x: format(float(x),".2f")).astype(float)
    df['Average'], df['Units'] = zip(*df.apply(lambda x: standarize_metric(x['Average'], x['KPI']), axis=1))
    df['Average'] = df['Average'].round(2)
    df['KPI'] = df.apply(lambda x: x['KPI'].replace(' (s)', ''), axis=1)


    message = '{}: \n'.format(scenario_name)
    message += '\n'.join(df['KPI'] + ': ' + df['Average'].astype(str) + ' ' + df['Units'])
    
    return message

def return_message_stats_complete(stats_path, scenario_name):
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    ptt_s = 'Scenario statistics'
    ptt_e = 'Process Cycle Time (s) distribution' 
    text = extract_text(stats_path, ptt_s, ptt_e)

    data = [x.split(',') for x in text.split('\n') if x != '']
    df = pd.DataFrame(data = data[1:], columns=data[0])

    df['Average'] = df['Average'].astype(float).astype(str).apply(lambda x: format(float(x),".2f")).astype(float)
    df['Average'], df['Units'] = zip(*df.apply(lambda x: standarize_metric(x['Average'], x['KPI']), axis=1))
    df['Average'] = df['Average'].round(2)
    df['KPI'] = df.apply(lambda x: x['KPI'].replace(' (s)', ''), axis=1)

    message = '{} \n'.format(scenario_name)
    message += '\n'.join(df['KPI'] + ': ' + df['Average'].astype(str) + ' ' + df['Units'])
    
    return message