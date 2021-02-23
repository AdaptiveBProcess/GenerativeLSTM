# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:04:49 2021

@author: Manuel Camargo
"""
import os
import time
import utils.support as sup


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


def sbatch_creator(log, fam, opt, ev):
    exp_name = (os.path.splitext(log)[0]
                    .lower()
                    .split(' ')[0][:5])
    if imp == 2:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=gpu',
                   '#SBATCH --gres=gpu:tesla:1',
                   '#SBATCH -J ' + exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'module load python/3.6.3/virtenv',
                   'source activate deep_generator_pip',
                   ]
    else:
        default = ['#!/bin/bash',
                   '#SBATCH --partition=main',
                   '#SBATCH -J '+exp_name,
                   '#SBATCH -N 1',
                   '#SBATCH --cpus-per-task=10',
                   '#SBATCH --mem=32000',
                   '#SBATCH -t 120:00:00',
                   'module load python/3.6.3/virtenv',
                   'source activate deep_generator_pip',
                   ]
    options = 'python lstm_pipeline.py -f ' + log
    options += ' -m ' + fam
    options += ' -o ' + opt
    options += ' -e ' + str(ev)
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
# Xserver ip

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# clean folder
for _, _, files in os.walk(output_folder):
    for file in files:
        os.unlink(os.path.join(output_folder, file))

# parameters definition
imp = 1  # keras lstm implementation 1 cpu, 2 gpu
logs = [
        ('confidential_1000.xes', 'gru', 'bayesian', 20),
        ('confidential_2000.xes', 'gru', 'rand_hpc', 40),
        ('BPI_Challenge_2012_W_Two_TS.xes', 'gru', 'rand_hpc', 40),
        ('BPI_Challenge_2017_W_Two_TS.xes', 'gru', 'rand_hpc', 40),
        ('poc_processmining.xes', 'gru', 'rand_hpc', 40),
        ('cvs_pharmacy.xes', 'gru', 'rand_hpc', 40),
        ('PurchasingExample.xes', 'gru', 'bayesian', 20),
        ('Production.xes', 'gru', 'bayesian', 20),
        ('ConsultaDataMining201618.xes', 'gru', 'bayesian', 20),
        ('insurance.xes', 'gru', 'bayesian', 20),
        ('callcentre.xes', 'gru', 'bayesian', 20),
        ('BPI_Challenge_2012_W_Two_TS.xes', 'lstm', 'rand_hpc', 40),
        ('BPI_Challenge_2017_W_Two_TS.xes', 'lstm', 'rand_hpc', 40),
        ('poc_processmining.xes', 'lstm', 'rand_hpc', 40),
        ('confidential_1000.xes', 'lstm', 'bayesian', 20),
        ('confidential_2000.xes', 'lstm', 'rand_hpc', 40),
        ('cvs_pharmacy.xes', 'lstm', 'rand_hpc', 40),
        ('PurchasingExample.xes', 'lstm', 'bayesian', 20),
        ('Production.xes', 'lstm', 'bayesian', 20),
        ('ConsultaDataMining201618.xes', 'lstm', 'bayesian', 20),
        ('insurance.xes', 'lstm', 'bayesian', 20),
        ('callcentre.xes', 'lstm', 'bayesian', 20),
        ('confidential_1000.xes', 'gru_cx', 'bayesian', 20),
        ('confidential_2000.xes', 'gru_cx', 'rand_hpc', 40),
        ('BPI_Challenge_2012_W_Two_TS.xes', 'gru_cx', 'rand_hpc', 40),
        ('BPI_Challenge_2017_W_Two_TS.xes', 'gru_cx', 'rand_hpc', 40),
        ('poc_processmining.xes', 'gru_cx', 'rand_hpc', 40),
        ('cvs_pharmacy.xes', 'gru_cx', 'rand_hpc', 40),
        ('PurchasingExample.xes', 'gru_cx', 'bayesian', 20),
        ('Production.xes', 'gru_cx', 'bayesian', 20),
        ('ConsultaDataMining201618.xes', 'gru_cx', 'bayesian', 20),
        ('insurance.xes', 'gru_cx', 'bayesian', 20),
        ('callcentre.xes', 'gru_cx', 'bayesian', 20),
        ('BPI_Challenge_2012_W_Two_TS.xes', 'lstm_cx', 'rand_hpc', 40),
        ('BPI_Challenge_2017_W_Two_TS.xes', 'lstm_cx', 'rand_hpc', 40),
        ('poc_processmining.xes', 'lstm_cx', 'rand_hpc', 40),
        ('confidential_1000.xes', 'lstm_cx', 'bayesian', 20),
        ('confidential_2000.xes', 'lstm_cx', 'rand_hpc', 40),
        ('cvs_pharmacy.xes', 'lstm_cx', 'rand_hpc', 40),
        ('PurchasingExample.xes', 'lstm_cx', 'bayesian', 20),
        ('Production.xes', 'lstm_cx', 'bayesian', 20),
        ('ConsultaDataMining201618.xes', 'lstm_cx', 'bayesian', 20),
        ('insurance.xes', 'lstm_cx', 'bayesian', 20),
        ('callcentre.xes', 'lstm_cx', 'bayesian', 20)
        ]

for log, fam, opt, ev in logs:
    # sbatch creation
    sbatch_creator(log, fam, opt, ev)
# submission
sbatch_submit(False)
