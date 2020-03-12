# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
from support_modules import support as sup
import os
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
    configs = ['BPI_2012_W_complete.csv',
               'BPI_Challenge_2012.csv',
               'CreditRequirement.csv',
               'PurchasingExample.csv',
               'Production.csv',
               'Helpdesk.csv',
               'ConsultaDataMining201618.csv']

    return configs

# =============================================================================
# Sbatch files creator
# =============================================================================

def sbatch_creator(configs):
    for i, _ in enumerate(configs):
        log = configs[i]
        exp_name = os.path.splitext(log)[0].lower().split(' ')[0][:8]
        if imp == 2:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:tesla:1',
                       '#SBATCH -J '+ exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_pip_tf_gpu'
                       ]
        else:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=main',
                       '#SBATCH -J '+ exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=25000',
                       '#SBATCH -t 72:00:00',
                       'module load  python/3.6.3/virtenv',
                       'source activate lstm_pip_tf'
                       ]

        options = 'python lstm.py -e ' + log + ' -i ' + str(imp)
        options += ' -a f_distance'
        if log in ['PurchasingExample.csv', 'Production.csv',
                   'ConsultaDataMining201618.csv', 'CreditRequirement.csv']:
          options += ' -s False'
        else:
          options += ' -s True'
        default.append(options)
        file_name = sup.folder_id()
        sup.create_text_file(default, os.path.join(output_folder, file_name))

# =============================================================================
# Sbatch files submission
# =============================================================================

def sbatch_submit(in_batch, bsize=10):
    file_list = create_file_list(output_folder)
    print('Number of experiments:', len(file_list),sep=' ')
    for i, _ in enumerate(file_list):
        if in_batch:
            if i%bsize == 0:
            		time.sleep(20)
            		os.system('sbatch ' + os.path.join(output_folder, file_list[i]))
            else:
            		os.system('sbatch ' + os.path.join(output_folder, file_list[i]))
        else:
            	os.system('sbatch ' + os.path.join(output_folder, file_list[i]))

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
imp = 1 # keras lstm implementation 1 cpu, 2 gpu
# configs definition
configs = configs_creation()
# sbatch creation
sbatch_creator(configs)
# submission
sbatch_submit(True)