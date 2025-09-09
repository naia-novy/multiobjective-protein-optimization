import pandas as pd
import numpy as np


filename = 'data/T7RBD_all_separate/all_separate_outputs.pkl'
output_filename = '/'.join(filename.split('/')[:-1]) + "/dms_comb_motif_score_weights.csv"

filename = 'data/T7RBD_dms_comb/dms_comb_separate.pkl'
output_filename = '/'.join(filename.split('/')[:-1]) + "/dms_comb_score_weights.csv"

df = pd.read_pickle(filename)

weight_columns = [col for col in df.columns if 'weight' in col]
total_weight = df.loc[:, weight_columns]
total_weight = total_weight.sum().sum()
weight_dict = {}

# normalize to total weights for score
for col in df.columns:
    if np.any([i in col for i in ['DMS', 'comb', 'motif']]) and np.any(
            [i in col for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG']]) and 'score' in col:
        dataset = [i for i in ['DMS', 'comb', 'motif'] if i in col][0]
        strain = [i for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG'] if i in col][0]
        weight = np.nansum(1/df[f'{dataset}_{strain}_weight']) / total_weight

        weight_dict[col] = weight


# normalize to total observances for class
class_columns = [col for col in df.columns if 'class' in col]
total_class_observances = (~df.loc[:, class_columns].isna()).sum().sum()
for col in df.columns:
    if np.any([i in col for i in ['DMS', 'comb', 'motif']]) and np.any(
            [i in col for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG']]) and 'class' in col and 'weight' not in col:
        dataset = [i for i in ['DMS', 'comb', 'motif'] if i in col][0]
        strain = [i for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG'] if i in col][0]
        weight = (~df[f'{dataset}_{strain}_class'].isna()).sum().sum() / total_class_observances

        weight_dict[col] = weight

weights = pd.Series(weight_dict)

# normalize across strains, so each strain should add up to one
weighted_by_strain = weight_dict.copy()
for col in df.columns:
    if np.any([i in col for i in ['DMS', 'comb', 'motif']]) and np.any(
            [i in col for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG']]) and 'score' in col:
        dataset = [i for i in ['DMS', 'comb', 'motif'] if i in col][0]
        strain = [i for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG'] if i in col][0]
        var_columns_strain = [col for col in weights.index if strain in col and 'class' not in col]
        total_strain_weight = weights.loc[var_columns_strain].sum().sum()
        weight = weights[f'{dataset}_{strain}_score'] / total_strain_weight

        weighted_by_strain[col] = weight

# normalize across strains, so each strain should add up to one
for col in df.columns:
    if np.any([i in col for i in ['DMS', 'comb', 'motif']]) and np.any(
            [i in col for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG']]) and 'class' in col and 'weight' not in col:
        dataset = [i for i in ['DMS', 'comb', 'motif'] if i in col][0]
        strain = [i for i in ['10G', 'BW2', 'BL21', 'rfaD', 'rfaG'] if i in col][0]
        var_columns_strain = [col for col in weights.index if strain in col and 'score' not in col]
        total_strain_weight = weights.loc[var_columns_strain].sum().sum()
        weight = weights[f'{dataset}_{strain}_class'] / total_strain_weight

        weighted_by_strain[col] = weight



weights = pd.Series(weighted_by_strain)
weights.to_csv(output_filename)