# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:08:16 2019

@author: Manuel Camargo
"""
import datetime
from support_modules import support as sup
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

# kernel
# repeat_num = 1


imp = 1
exp_name = 'help_next'
models_folder = 'output_files'
file_list = create_folder_list(models_folder, 2)

output_folder = 'jobs_files'

for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

for file in file_list:
    if imp == 2:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=gpu',
                   '#SBATCH --gres=gpu:tesla:1',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=7000',
                   '#SBATCH -t 24:00:00',
                   'module load  python/3.6.3/virtenv',
                   'source activate lstm_dev_cpu'
                   ]
    else:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=main',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=7000',
                   '#SBATCH -t 24:00:00',
                   'module load  python/3.6.3/virtenv',
                   'source activate lstm_dev_cpu'
                   ]

    default.append('python lstm.py' +
                   ' -a predict_next' +
                   ' -c ' + file['folder'] +
                   ' -b "' + file['file'] + '"' +
                   ' -o True' +
                   ' -x False' +
                   ' -t 100')
    file_name = sup.folder_id()
    sup.create_text_file(default, os.path.join(output_folder, file_name))

file_list = create_file_list(output_folder)
print('Number of experiments: ' + str(len(file_list)))
for i, _ in enumerate(file_list):
    if (i % 10) == 0:
        time.sleep(20)
        os.system('sbatch ' + os.path.join(output_folder, file_list[i]))
    else:
        os.system('sbatch ' + os.path.join(output_folder, file_list[i]))
