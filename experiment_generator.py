# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import itertools
import support as sup
import os
import random
import time


def create_file_list(path): 
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# kernel

output_folder = 'jobs_files'

for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

log = 'T_BPIC15_4.xes'
exp_name = 'bpic15T_4'
imp = 1
lstm_act = ['tanh', 'relu', ' ']
dense_act = ['sigmoid','linear', '']
norm_method = ['max', 'lognorm']
model_type = ['specialized','concatenated','shared_cat']
n_sizes = [5, 10, 15]
l_sizes = [50, 100, 150]
optimizers = ['Nadam', 'Adam']
choice = 'random'
num_choice = 50

listOLists = [lstm_act, dense_act, norm_method, model_type, n_sizes, l_sizes, optimizers]
configs = list()
for lists in itertools.product(*listOLists):
    configs.append(dict(lstm_act=lists[0],
                        dense_act=lists[1],
                        norm_method=lists[2],
                        model_type=lists[3],
                        n_sizes=lists[4],
                        l_sizes=lists[5],
                        optimizers=lists[6]))
print(len(configs))
#if choice == 'random':
#    configs = random.sample(configs, num_choice)
#
#for i, _ in enumerate(configs):
#    if imp == 2:
#        default = ['#!/bin/bash',
#                   '#SBATCH --partition=gpu',
#                   '#SBATCH --gres=gpu:tesla:1',
#                   '#SBATCH -J '+ exp_name,
#                   '#SBATCH -N 1',
#                   '#SBATCH --mem=7000',
#                   '#SBATCH -t 24:00:00',
#                   'module load  python/3.6.3/virtenv',
#                   'source activate lstm_gpu'
#                   ]
#    else:
#        default = ['#!/bin/bash',
#                   '#SBATCH --partition=main',
#                   '#SBATCH -J '+ exp_name,
#                   '#SBATCH -N 1',
#                   '#SBATCH --mem=7000',
#                   '#SBATCH -t 24:00:00',
#                   'module load  python/3.6.3/virtenv',
#                   'source activate lstm_cpu'
#                   ]
#
#    options = 'python lstm.py -a training -e ' + log + ' -i ' + str(imp)
#    if configs[i]['lstm_act'] is not None:
#      options += ' -l ' + configs[i]['lstm_act']
#    if configs[i]['dense_act'] is not None: 
#      options += ' -d ' + configs[i]['dense_act']
#    if configs[i]['norm_method'] is not None:       
#      options += ' -n ' + configs[i]['norm_method']
#    if configs[i]['model_type'] is not None:       
#      options += ' -t ' + configs[i]['model_type']
#    if configs[i]['n_sizes'] is not None:
#      options += ' -b ' + str(configs[i]['n_sizes'])
#    if configs[i]['l_sizes'] is not None:
#      options += ' -c ' + str(configs[i]['l_sizes'])
#    if configs[i]['optimizers'] is not None:
#      options += ' -o ' + configs[i]['optimizers']
#
#    default.append(options)
#    file_name = sup.folder_id()
#    sup.create_text_file(default, os.path.join(output_folder, file_name))
#
#       
#file_list = create_file_list(output_folder)
#print('Number of experiments:', len(file_list),sep=' ')
#for i, _ in enumerate(file_list):
#	if i%10 == 0:
#		time.sleep(20)
#		os.system('sbatch ' + os.path.join(output_folder, file_list[i]))
#	else:
#		os.system('sbatch ' + os.path.join(output_folder, file_list[i]))