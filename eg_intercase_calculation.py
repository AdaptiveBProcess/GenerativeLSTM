# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:58:30 2020

@author: Manuel Camargo
"""
import itertools
from support_modules import support as sup
import os
import random
import time
import pandas as pd
import numpy as np

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
# Sbatch files creator
# =============================================================================


def sbatch_creator(log, one_ts):
    exp_name = (os.path.splitext(log)[0]
                    .lower()
                    .split(' ')[0][:5])
    if imp == 2:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=gpu',
                   '#SBATCH --gres=gpu:tesla:1',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=14000',
                   '#SBATCH -t 72:00:00',
                   'module load cuda/10.0',
                   'module load python/3.6.3/virtenv',
                   'source activate lstm_pip'
                   ]
    else:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=amd',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=14000',
                   '#SBATCH -t 72:00:00',
                   'module load cudnn/7.6.3/cuda-10.0',
                   'module load python/3.6.3/virtenv',
                   'source activate lstm_exp_v3'
                   ]

        options = 'python lstm.py -f ' + log
        options += ' -a inter_case'
        options += ' -o '+str(one_ts)

    default.append(options)
    file_name = sup.folder_id()
    sup.create_text_file(default, os.path.join(output_folder, file_name))

# =============================================================================
# Sbatch files submission
# =============================================================================


def sbatch_submit(in_batch, bsize=20):
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

logs = create_file_list('input_files')
imp = 1  # keras lstm implementation 1 cpu, 2 gpu
two_ts_logs = ['BPI_Challenge_2012_W_Two_TS.csv',
               'BPI_Challenge_2017_W_Two_TS.csv',
               'PurchasingExample.csv',
               'Production.csv',
               'ConsultaDataMining201618.csv']

for log in logs:
    # sbatch creation
    one_ts = False if log in two_ts_logs else True
    sbatch_creator(log, one_ts)
# submission
sbatch_submit(False)
