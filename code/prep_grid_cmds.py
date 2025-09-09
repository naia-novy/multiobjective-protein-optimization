#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:40:16 2022

@author: natenovy
"""

# Script used for preping commands for grid search on CHTC

import itertools

encodings = ['ESM_35M_12','METL-G-50M-3D']
batch_sizes = [128, 256, 512]
learning_rates = [0.01,0.001]
activation_fxn = ['gelu', 'relu']

combinations = list(itertools.product(encodings, batch_sizes, learning_rates, activation_fxn))


command_list = []
for idx in range(len(combinations)):
    cmd = f"python3 code/train.py @training_args/cnn_dms_comb_motif_ensemble_comparison_threshold.txt --encoding {combinations[idx][0]} --batch_size {combinations[idx][1]} --lr {combinations[idx][2]} --fc_activation_function {combinations[idx][3]}\n"
    command_list.append(cmd)


with open('gen_grid_coms.txt', 'w+') as f:
    for com in command_list:
        f.write(com)
