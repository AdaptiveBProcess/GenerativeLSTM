# -*- coding: utf-8 -*-
import itertools
import numpy as np
import random
import string
import support as sup
from operator import itemgetter
import collections


def accuracy(log_data, simulation_data, ramp_io_perc = 0.2):
    # get log data
    temp_data = log_data + simulation_data
    alias = create_task_alias(temp_data)

    log_data = reformat_events(log_data, alias)
    # get simulation data
    simulation_data = reformat_events(simulation_data, alias)
    
    actual_categories = [c['profile'] for c in log_data]
    ocurrences = dict(collections.Counter(actual_categories))
    
    predicted_categories = [c['profile'] for c in simulation_data]
    ocurrences2 = dict(collections.Counter(predicted_categories))
    
    intersect = list(set(ocurrences.keys()).intersection(set(ocurrences2.keys())))
    expected_traces = 0
    accuracy = list()
    acc = dict()
    if len(intersect)>0: 
        accuracy, expected_traces = accuracy_measurement(ocurrences, ocurrences2, intersect)
        not_expected = len(simulation_data) - expected_traces
        print('Traces not in event log :', not_expected, sep=' ')
        print('Analyzed traces:', expected_traces, sep=' ')
        print('Model Presicion:', sup.ffloat(np.mean([c['precision'] for c in accuracy]),2), sep=' ')
        print('Model Recall:', sup.ffloat(np.mean([c['recall'] for c in accuracy]),2), sep=' ')
        print('F1 score:', sup.ffloat(np.mean([c['f1'] for c in accuracy]),2), sep=' ')
        acc['t_not_in'] = not_expected
        acc['num_traces'] =  expected_traces
        acc['precision'] = np.mean([c['precision'] for c in accuracy])
        acc['recall'] = np.mean([c['recall'] for c in accuracy])
        acc['f1'] =  np.mean([c['f1'] for c in accuracy])
    else:
        acc['t_not_in'] = len(simulation_data)
        acc['num_traces'] =  np.nan
        acc['precision'] = np.nan
        acc['recall'] = np.nan
        acc['f1'] =  np.nan
    return acc
        
def accuracy_measurement(ocurrences, ocurrences2, intersect):
    expected_traces = 0
    accuracy = list()
    for cat in intersect:
        expected_traces += ocurrences2[cat]
        tp = np.min([ocurrences2[cat], ocurrences[cat]])
        fn , fp = 0, 0
        if tp <= ocurrences2[cat]:
            fp = ocurrences2[cat] - tp
        else:
            fn = tp - ocurrences2[cat]
        try: 
            precision = tp/(tp + fp) 
        except: 
            precision = 0
        try: 
            recall = tp/(tp + fn)
        except:
            recall = 0
        f1 = 2*((precision*recall)/(precision+recall))
        accuracy.append(dict(variant=cat,precision=precision,recall=recall,f1=f1))
    return accuracy, expected_traces
        


def process_results(similarity):
    data = sorted(list(similarity), key=lambda x:x['run_num'])
    run_similarity = list()
    for key, group in itertools.groupby(data, key=lambda x:x['run_num']):
        values = list(group)
        group_similarity = [x['sim_score'] for x in values]
        run_similarity.append(np.mean(group_similarity))
    print(run_similarity)
    print(np.mean(run_similarity))
    # [print(x) for x in similarity]
    
def create_task_alias(df):
    subsec_set = set()
    task_list = [[x['task'], x['user']] for x in df]
    [subsec_set.add((x[0], x[1])) for x in task_list]
    variables = sorted(list(subsec_set))
    characters = [chr(i) for i in range(0, len(variables))]
    aliases = random.sample(characters, len(variables))
    alias = dict()
    for i in range(0, len(variables)):
        alias[variables[i]] = aliases[i]
    return alias

def reformat_events(data, alias):
    # Add alias
    [x.update(dict(alias=alias[(x['task'],x['user'])])) for x in data]
    # Define cases
    cases = sorted(list(set([x['caseid'] for x in data])))
    # Reformat dataset
    temp_data = list()
    for case in cases:
        temp_dict= dict(caseid=case,profile='')
        events = sorted(list(filter(lambda x: x['caseid']==case, data)), key=itemgetter('start_timestamp'))
        for i in range(0, len(events)):
            temp_dict['profile'] = temp_dict['profile'] + events[i]['alias']
        temp_dict['start_time'] = events[0]['start_timestamp']
        temp_data.append(temp_dict)
    return sorted(temp_data, key=itemgetter('start_time'))

