# -*- coding: utf-8 -*-
from sys import stdout
import numpy as np
import datetime
import os
import csv
import uuid
import json
import platform as pl
import time


def folder_id():
    return datetime.datetime.today().strftime('%Y%m%d_') + str(uuid.uuid4()).upper().replace('-', '_')
#generate unique bimp element ids
def gen_id():
    return "qbp_" + str(uuid.uuid4())
#printing process functions
def print_progress(percentage, text):
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()

def print_performed_task(text):
    stdout.write("\r%s" % text + "...      ")
    stdout.flush()

def print_done_task():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")

def file_size(path_file):
    size = 0
    file_exist = os.path.exists(path_file)
    if file_exist:
        size = len(open(path_file).readlines())
    return size

#printing formated float
def ffloat(num, dec):
    return float("{0:.2f}".format(np.round(num,decimals=dec)))

#transform a string into date object
def get_time_obj(date, timeformat):
    date_modified = datetime.datetime.strptime(date,timeformat)
    return date_modified

#reduce list of lists with no repetitions
def reduce_list(input, dtype='int'):
    text = str(input).replace('[', '').replace(']', '')
    text = [x for x in text.split(',') if x != ' ']
    if text and not text == ['']: 
        if dtype=='int':
            return list(set([int(x) for x in text]))
        elif dtype=='float':
            return list(set([float(x) for x in text]))
        elif dtype=='str':
            return list(set([x.strip() for x in text]))
        else:
            raise ValueError(dtype)
    else:
        return list()

#print a csv file from list of lists
def create_file_from_list(index, output_file):
    with open(output_file, 'w') as f:
        for element in index:
            f.write(', '.join(list(map(lambda x: str(x), element))))
            f.write('\n')
        f.close()

#print a csv file from list of lists
def create_text_file(index, output_file):
    with open(output_file, 'w') as f:
        for element in index:
            f.write(element+'\n')
        f.close()

#print debuging csv file
def create_csv_file(index, output_file, mode='w'):
    with open(output_file, mode) as f:
        for element in index:
            w = csv.DictWriter(f, element.keys())
            w.writerow(element)
        f.close()

def create_csv_file_header(index, output_file, mode='w'):
    with open(output_file, mode, newline='') as f:
        fieldnames = index[0].keys()
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for element in index:
            w.writerow(element)
        f.close()

def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
         f.write(json.dumps(dictionary))
         
# rounding lists values preserving the sum values
def round_preserve(l,expected_sum):
    actual_sum = sum(l)
    difference = round(expected_sum - actual_sum,2)
    if difference > 0.00:
        idx= l.index(min(l))
    else:
        idx= l.index(max(l))
    l[idx] +=difference
    return l

def create_symetric_list(width, length):
    positions = list()
    numbers = list()
    [positions.append(width * (i + 1)) for i in range(0, length)]
    a = np.median(positions)
    [numbers.append(x - a) for x in positions]
    return numbers

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def copy(source, destiny):
    if pl.system().lower() == 'windows':
        os.system('copy "' + source + '" "' + destiny + '"')
    else:
        os.system('cp "' + source + '" "' + destiny + '"')


def timeit(method) -> dict:
    """
    Decorator to measure execution times of methods

    Parameters
    ----------
    method : Any method.

    Returns
    -------
    dict : execution time record

    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

# #%%
# import re
# results = pd.read_csv('C:/Users/Manuel Camargo/Documents/Repositorio/experiments/sc_lstm_dev/test_data.csv')
# results = results.drop(columns=['Unnamed: 0'])
# def string_list_to_float(input):
#     text = str(input).replace('[', '').replace(']', '').replace('\n', '').replace(',', '')
#     text = re.sub(' +', ' ', text)
#     text = text.strip()
#     temp_list = list()
#     for number in text.split(' '):
#         temp_list.append(float(number))
#     return temp_list
# results['tm_pred'] = results.tm_pred.apply(string_list_to_float)
# results['tm_expect'] = results.tm_expect.apply(string_list_to_float)

# # results['end_timestamp'] =  pd.to_datetime(results['end_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f')

# evaluator = Evaluator()
# print(evaluator.measure('mae_suffix', results, 'tm'))
# print(evaluator.measure('similarity', results, 'ac'))
# print(evaluator.measure('similarity', results, 'rl'))


