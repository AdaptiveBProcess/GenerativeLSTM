from glob import glob
import os
import subprocess
import pandas as pd

def simulate(bimp_path, output_path, csv_output_path):
    args = ['java', '-jar', bimp_path, output_path, '-csv', csv_output_path]
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Simulation was successfully executed")
    elif result.returncode == 1:
        execption_output = [result.stdout.split('\n')[i-1] for i in range(len(result.stdout.split('\n'))) if 'BPSimulatorException' in result.stdout.split('\n')[i]]
        print("Execution failed :", ' '.join(execption_output))

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
    
def extract_stats(csv_output_path, event_log, rule, event_log_class):

    ptt_s = 'Scenario statistics'
    ptt_e = 'Process Cycle Time (s) distribution' 
    text = extract_text(csv_output_path, ptt_s, ptt_e)

    data = [x.split(',') for x in text.split('\n') if x != '']
    df = pd.DataFrame(data = data[1:], columns=data[0])

    df['Average'] = df['Average'].astype(float).astype(str).apply(lambda x: format(float(x),".2f")).astype(float)
    df['Average'], df['Units'] = zip(*df.apply(lambda x: standarize_metric(x['Average'], x['KPI']), axis=1))
    df['Average'] = df['Average'].round(2)
    df['KPI'] = df.apply(lambda x: x['KPI'].replace(' (s)', ''), axis=1)
    df['event_log'] = event_log
    df['rule'] = rule
    df['event_log_class'] = event_log_class
    df = df[['event_log', 'event_log_class', 'rule', 'KPI', 'Min',	'Average', 'Max', 'Units']]

    return df

#Simulate Manual Simulation models
files_path = os.path.join('input_files', 'tests', 'simulation_files')
files = glob(files_path + '/*')
bimp_path = os.path.join('external_tools', 'bimp', 'qbp-simulator-engine_with_csv_statistics.jar')
for file in files:
    filename = file.split('\\')[-1].split('.')[0]
    csv_output_path = os.path.join('input_files', 'tests', 'simulation_stats', f'{filename}.csv')
    simulate(bimp_path, file, csv_output_path)

#Consolidate stats
df_stats = pd.DataFrame(data=[], columns=['event_log', 'event_log_class', 'rule', 'KPI', 'Min',	'Average', 'Max', 'Units'])

#Manual files
manual_files = glob(os.path.join('input_files', 'tests', 'simulation_stats', '*'))
for manual_file in manual_files:
    
    event_log = manual_file.split('\\')[-1].split('.')[0].split(' ')[0]
    rule = ' '.join(manual_file.split('\\')[-1].split('.')[0].split(' ')[1:])
    
    df_stats_tmp = extract_stats(manual_file, event_log, rule, 'Manual')
    df_stats = pd.concat([df_stats, df_stats_tmp])

#Generated files
generated_files = glob(os.path.join('output_files', 'simulation_stats', '*'))
for generated_file in generated_files:
    
    event_log = generated_file.split('\\')[-1].split('.')[0].split(' ')[0]
    rule = ' '.join(generated_file.split('\\')[-1].split('.')[0].split(' ')[1:])
    
    df_stats_tmp = extract_stats(generated_file, event_log, rule, 'Generated')
    df_stats = pd.concat([df_stats, df_stats_tmp])

#Pivot data
df_paper = df_stats.sort_values(by=['Metric', 'Rule', 'Event Log Type'])[['Event Log', 'Event Log Type', 'Rule', 'Metric', 'Average']]
df_paper = df_paper[df_paper['Metric'].isin(['Cost', 'Process Cycle Time'])]
df_paper = df_paper.pivot(index=['Event Log', 'Rule', 'Metric'], columns='Event Log Type', values='Average')
df_paper = df_paper.reset_index()
df_paper['Error'] = abs((df_paper['Generated'] - df_paper['Manual']) / df_paper['Manual'])

#Save file
output_csv_path = os.path.join('output_files', 'consolidated_results.xlsx')
df_paper.to_excel(output_csv_path, index=False)