# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import datetime
import utils.support as sup
import os
import time


def create_file_list(path):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            file_list.append(f)
    return file_list


def create_folder_list(path, num_models):
    file_list = list()
    for _, dirs, _ in os.walk(path):
        for d in dirs:
            for _, _, files in os.walk(os.path.join(path, d)):
                files_filtered = list()
                for f in files:
                    _, file_extension = os.path.splitext(f)
                    if file_extension == '.h5':
                        files_filtered.append(f)
                creation_list = list()
                for f in files_filtered:
                    date = os.path.getmtime(os.path.join(path, d, f))
                    creation_list.append(
                        {'filename': f,
                         'creation': datetime.datetime.utcfromtimestamp(date)})
                creation_list = sorted(creation_list,
                                       key=lambda x: x['creation'],
                                       reverse=True)
                for f in creation_list[:num_models]:
                    file_list.append(dict(folder=d, file=f['filename']))
    return file_list

# =============================================================================
# Sbatch files creator
# =============================================================================

def sbatch_creator(file_list, activity):
    exp_name = activity[:4]
    for file in file_list:
        if imp == 2:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=gpu',
                       '#SBATCH --gres=gpu:tesla:1',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --mem=7000',
                       '#SBATCH -t 24:00:00',
                       'module load cuda/10.0',
                       'module load python/3.6.3/virtenv',
                       'source activate deep_generator_pip'
                       ]
        else:
            default = ['#!/bin/bash',
                       '#SBATCH --partition=main',
                       '#SBATCH -J ' + exp_name,
                       '#SBATCH -N 1',
                       '#SBATCH --cpus-per-task=20',
                       '#SBATCH --mem=32000',
                       '#SBATCH -t 120:00:00',
                       'module load cuda/10.0',
                       'module load python/3.6.3/virtenv',
                       'source activate deep_generator_pip'
                       ]
    
        default.append('python lstm.py' +
                       ' -o False' +
                       ' -a ' + activity +
                       ' -c ' + file['folder'] +
                       ' -b "' + file['file'] + '"' +
                       ' -v "Random Choice"' +
                       ' -r 5')
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

# kernel
# repeat_num = 1


imp = 1
models_folder = 'output_files'
file_list = create_folder_list(models_folder, 1)

output_folder = 'jobs_files'

activities = ['pred_log']

for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

for activity in activities:
    sbatch_creator(file_list, activity)
    # submission
# sbatch_submit(True)
