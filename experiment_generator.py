# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import itertools
from support_modules import support as sup
import os
import random
import time

# =============================================================================
#  Support
# =============================================================================


def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list

# =============================================================================
# Experiments definition
# =============================================================================


def configs_creation(num_choice=0):
    configs = list()
    if is_single_exp:
        config = dict(
            lstm_act='relu',
            dense_act=None,
            optimizers='Adam',
            norm_method='lognorm',
            n_sizes=15,
            l_sizes=100)
        for model in model_type:
            configs.append({**{'model_type': model}, **config})
    else:
        # Search space definition
        dense_act = ['sigmoid', 'linear', None]
        norm_method = ['max']
        l_sizes = [50, 100, 200]
        optimizers = ['Nadam', 'Adam']
        lstm_act = ['tanh', 'sigmoid', 'relu']
        if arch == 'sh':
            n_sizes = [5, 10, 15]
            listOLists = [lstm_act, dense_act,
                          norm_method, n_sizes,
                          l_sizes, optimizers]
        else:
            listOLists = [lstm_act, dense_act,
                          norm_method, l_sizes,
                          optimizers]
        # selection method definition
        choice = 'random'
        preconfigs = list()
        for lists in itertools.product(*listOLists):
            if arch == 'sh':
                preconfigs.append(dict(lstm_act=lists[0],
                                       dense_act=lists[1],
                                       norm_method=lists[2],
                                       n_sizes=lists[3],
                                       l_sizes=lists[4],
                                       optimizers=lists[5]))
            else:
                preconfigs.append(dict(lstm_act=lists[0],
                                       dense_act=lists[1],
                                       norm_method=lists[2],
                                       l_sizes=lists[3],
                                       optimizers=lists[4]))
        # configurations definition
        if choice == 'random':
            preconfigs = random.sample(preconfigs, num_choice)
        for preconfig in preconfigs:
            for model in model_type:
                config = {'model_type': model}
                config = {**config, **preconfig}
                configs.append(config)
    return configs

# =============================================================================
# Sbatch files creator
# =============================================================================


def sbatch_creator(configs):
    for i, _ in enumerate(configs):
        if configs[i]['model_type'] in ['shared_cat', 'seq2seq']:
            exp_name = (os.path.splitext(log)[0]
                        .lower()
                        .split(' ')[0][:4] + arch)
        elif configs[i]['model_type'] in ['shared_cat_inter', 'seq2seq_inter']:
            exp_name = (os.path.splitext(log)[0]
                        .lower()
                        .split(' ')[0][:4] + arch + 'i')
        if imp == 2:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:tesla:1',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_dev_cpu'
                       ]
        else:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=main',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_dev_cpu'
                       ]

        def format_option(short, parm):
            return (' -'+short+' None'
                    if configs[i][parm] is None
                    else ' -'+short+' '+str(configs[i][parm]))

        options = 'python lstm.py -f ' + log + ' -i ' + str(imp)
        options += ' -a training'
        options += ' -o True'
        options += format_option('l', 'lstm_act')
        options += format_option('y', 'l_sizes')
        options += format_option('d', 'dense_act')
        options += format_option('n', 'norm_method')
        options += format_option('m', 'model_type')
        options += format_option('p', 'optimizers')
        if arch == 'sh':
            options += format_option('z', 'n_sizes')

        default.append(options)
        file_name = sup.folder_id()
        sup.create_text_file(default, os.path.join(output_folder, file_name))

# =============================================================================
# Sbatch files submission
# =============================================================================


def sbatch_submit(in_batch, bsize=10):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list), sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if (i % bsize) == 0:
                time.sleep(20)
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
            else:
                os.system('sbatch '+os.path.join(output_folder, file_list[i]))
        else:
            os.system('sbatch '+os.path.join(output_folder, file_list[i]))

# =============================================================================
# Kernel
# =============================================================================


# create output folder
output_folder = 'jobs_files'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition

is_single_exp = False
# s2, sh
arch = 'sh'
log = 'BPI_Challenge_2013_closed_problems.xes'
imp = 1  # keras lstm implementation 1 cpu, 2 gpu

# Same experiment for both models
if arch == 'sh':
    model_type = ['shared_cat']
else:
    model_type = ['seq2seq_inter', 'seq2seq']

# configs definition
configs = configs_creation(num_choice=15)
# sbatch creation
sbatch_creator(configs)
# submission
# sbatch_submit(True)
