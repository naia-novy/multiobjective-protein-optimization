#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:40:16 2022

@author: natenovy
"""

# Script used for preping commands for SA on CHTC

import itertools
from collections import Counter
#
# optim_options = list(itertools.product([0, 1], repeat=5))
# optim_options = [''.join(list(str(j) for j in i)) for i in optim_options]
# optim_options = list(set([i[:3]+'x'+i[-1] for i in optim_options]))
# optim_options = ['10101','11111']
optim_options = ['1xxxx','x1xxx','xx1xx','xxx1x','xxxx1', '11111', '00000']
num_steps = [100000]
fxn = ['gap']
mut_options = [16,8,4,2]
outputs_per_command = list(range(100))
# optim_options = ['1xxxx']

# # optim_options.remove('111x1')
# outputs_per_generalist = list(range(len(outputs_per_command)*20))

combinations = list(itertools.product(optim_options,mut_options,num_steps,fxn, outputs_per_command))
# combinations_generalists = list(itertools.product(['111x1'],mut_options,num_steps,cooling_schedule,fxn, outputs_per_generalist))

# combinations = combinations + combinations_generalists
# combinations = combinations_generalists
command_list = []
for idx in range(len(combinations)):
    cmd = f"python3 code/multiobjective_SA.py --strict_num_mut --checkpoint_base_dir RI_CNN_all --outputs_to_maximize {combinations[idx][0]} --num_designs 2 --num_muts {combinations[idx][1]} --num_steps {combinations[idx][2]} --fitness_fxn {combinations[idx][3]} --seed {idx} @RI_CNN_all/A7LAxd7d/version_0/args.txt  \n"
    command_list.append(cmd)


with open('sa_CNN.txt', 'w+') as f:
    for com in command_list:
        f.write(com)





optim_options = ['1xxxx','x1xxx','xx1xx','xxx1x','xxxx1']
output_dir = 'RI_CNN_all'
fxn = ['sum']
outputs_per_command = list(range(50))
num_steps = [50000]
mut_options = [12,9,6,3]

combinations = list(itertools.product(optim_options,mut_options,num_steps,fxn, outputs_per_command))
command_list = []
for idx in range(len(combinations)):
    cmd = f"python3 code/multiobjective_SA.py --strict_num_mut --output_directory {output_dir} --checkpoint_base_dir RI_CNN_all --outputs_to_maximize {combinations[idx][0]} --num_designs 2 --num_muts {combinations[idx][1]} --num_steps {combinations[idx][2]} --fitness_fxn {combinations[idx][3]} --seed {idx} @RI_CNN_all/A7LAxd7d/version_0/args.txt \n"
    command_list.append(cmd)


with open('RI_infectivity_SA.txt', 'w+') as f:
    for com in command_list:
        f.write(com)


################################################################

import itertools
from collections import Counter

optim_options = list(itertools.product(['0', '1', 'x'], repeat=5))
optim_options = [''.join(i) for i in optim_options]
iterable = optim_options.copy()
for i in iterable:
    counts = Counter(i)
    if counts['1'] == 0: # remove no optima options
        optim_options.remove(i)
    elif counts['0'] == 0: # remove and only maximizing funcitons
        optim_options.remove(i)
    elif counts['x'] >= 4: # remove infectivities and no objective
        optim_options.remove(i)


optim_options += ['1xxxx','x1xxx','xx1xx','xxx1x','xxxx1']

fxn = ['utopia_gap_percentile']
outputs_per_command = list(range(2))
output_dir = 'RI_10part_trial_10T'
num_steps = [50000]
mut_options = [6]
num_designs = 16

combinations = list(itertools.product(optim_options,mut_options,num_steps,fxn, outputs_per_command))
command_list = []
for idx in range(len(combinations)):
    cmd = f"python3 code/multiobjective_SA.py --strict_num_mut --output_directory {output_dir} --checkpoint_base_dir RI_CNN_all_10part --outputs_to_maximize {combinations[idx][0]} --num_designs {num_designs} --num_muts {combinations[idx][1]} --num_steps {combinations[idx][2]} --fitness_fxn {combinations[idx][3]} --seed {idx} @RI_CNN_all_10part/aziLgQXr/version_0/args.txt  \n"
    command_list.append(cmd)


with open('trial_for_specificity_coms.txt', 'w+') as f:
    for com in command_list:
        f.write(com)
