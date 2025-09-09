#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 07:35:49 2022

@author: natenovy
"""

import glob
import itertools
import pandas as pd
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
import argparse
from collections import Counter

import encode


WT = 'QVWSGSAGGGVSVTVSQDLRFRNIWIKCANNSWNFFRTGPDGIYFIASDGGWLRFQIHSNGLGFKNIADSRSVPNAIMVENE'
AA_options = ['ACDEFGHIKLMNPQRSTVWY'] * len(WT)
AAs = 'ACDEFGHIKLMNPQRSTVWY-'

# Set seaborn settings to produce high quality figures
sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 100})
sns.set_style("ticks")

def plot(hams, mut):
    # Function for plotting hamming distance
    df = pd.DataFrame(hams,columns=['hams'])
    p = sns.histplot(data=df, x = 'hams',log_scale=(False,True), bins=20)

    p.set_xlabel("Hamming Distance", fontsize=12)
    p.set_ylabel("Frequency", fontsize=12)
    p.set(title=mut)

    print(mut)
    plt.savefig(f'{base_dir}{mut}_hamming.pdf')
    plt.show()
    plt.clf()

def calculate_hamming(sequences):
    # Funciton for calculating hamming distances between all combinations of sequences in a list
    len_seq = len(sequences[0])
    combos = list(itertools.combinations(sequences,2))

    hams = []
    for combo in combos:
        ham = hamming(list(combo[0]),list(combo[1]))*len_seq
        hams.append(ham)

    return hams

def plot_pos_freq(df, WT, mut):
    # Plot the frequency at which mutations are made at each position in the protein
    sequences = df.sequence.tolist()
    pos_diffs = []
    for sequence in sequences:
        seq_diffs = [i for i in range(len(sequence)) if sequence[i] != WT[i]]
        pos_diffs.append(seq_diffs)

    pos_diffs = [x for xs in pos_diffs for x in xs]

    p = sns.histplot(pos_diffs,bins=len(WT))

    p.set_xlabel("Sequence index", fontsize=12)
    p.set_ylabel("Frequency", fontsize=12)
    p.set(title=f'Index Frequency {mut}')

    plt.savefig(f'{base_dir}{mut}_index_frequency.pdf')
    plt.show()
    plt.clf()


def filter_set(df, objective, mut_option, min_percentile=50):
    # Filter designed datasets before clustering
    # Select mutational distance
    df = df.loc[df.num_mutations == mut_option]
    print(f'{objective} initial len: ', len(df))

    # Remove duplicate entries
    df.drop_duplicates('sequence', inplace=True)
    print(f'{objective} len after removing duplicates: ', len(df))

    # Make sure not to include anything that has not gone through very many iterations for SA
    if min_percentile != 0:
        df = df[df.objective_score > df.objective_score.quantile(min_percentile/100)]
        print(f'{objective} len after requiring taking {min_percentile}th percentile: ', len(df))

    # Remove non endpoint sequences
    if args.endpoints_only:
        endpoint = df['num_iter'].max()
        df = df.loc[df.num_iter == endpoint]
        print(f'{objective} len after requiring sequences be endpoints: ', len(df))

    df.reset_index(drop=True, inplace=True)

    return df


def main(directory, base_dir, desired_outputs, min_percentile):
    # Main wrapper function for combining all outputs

    # Remove extra directories before loading files
    if '*' in directory or directory == None:
        dirs = glob.glob(f'{base_dir}*')
        # if f'{base_dir}condor_logs' in dirs:
        #     dirs.remove(f'{base_dir}condor_logs')

        # remove output xlsx if present
        if f'{base_dir}top_subsets_for_everything.xlsx' in dirs:
            dirs.remove(f'{base_dir}top_subsets_for_everything.xlsx')

        # remove any output pdfs
        for i in glob.glob(f'{base_dir}*.pdf'):
            if i in dirs:
                dirs.remove(i)
    else:
        dirs = [directory]

    # Loop over each subdir and combine to single dir
    for dir_ in dirs:
        objective = str(dir_.split('/')[-1])
        var_df_list = []
        variants = glob.glob(f'{dir_}/*.csv')
        for var in variants:
            variant = pd.read_csv(var, index_col=0)
            variant.loc[:,'objective'] = objective
            order = len(objective) - Counter(objective)['x']
            variant.loc[:, 'order'] = order
            remove = False

            if not remove:
                var_df_list.append(variant)

        # Combine all dfs in var_df_list and filter
        # If not all of the data is in directory (single entry in var_df_list) this may fail
        if var_df_list == []:
            continue
        full_df = pd.concat(var_df_list)
        full_df.reset_index(inplace=True,drop=True)
        full_df.to_pickle(f'{dir_}/{objective}_full.pkl')

        # Set default mut_options
        mut_options = list(set(full_df.num_mutations.tolist()))

        # PAM method is better, but these datasets are very large and it takes too long
        kmedoids_fxn = 'alternate'
        # Use PAM for all other smaller datasets becasue it is more accurate
        kmedoids_fxn = 'pam'

        # For each mutational frequency, cluster subset using most stringently filtered df that also has >= 300 datapoints
        for mut_option in mut_options:
            sub_df = filter_set(full_df, objective, mut_option, min_percentile=min_percentile)
            if len(sub_df) == 0:
                continue

            # Sort sequences by lower bound of objective score
            sub_df.loc[:,'sort_val'] = sub_df.objective_score - sub_df.std_ensemble_score
            sub_df = sub_df.sort_values(by='sort_val', ascending=False)
            sub_df.reset_index(drop=True,inplace=True)

            # Save df with all sorted sequences to excel
            sub_df.to_pickle(f'{dir_}/{objective}_all_{mut_option}_mutant.pkl')
            out_df = sub_df.copy()

            select_top_x = len(sub_df)
            if args.should_cluster:
                # Set clustering hyperparameters
                select_top_x = desired_outputs
                cluster_number = select_top_x * 2 if select_top_x*2 < len(sub_df) else len(sub_df)

                print(f'Number of clusters used ({objective}): {cluster_number}')

                # Encode sequences: this says aaindex, but it is really converting AA letter codes to integers for hamming clustering
                encoded_seqs = encode.seq2ind(sub_df.sequence)
                encoded_seqs = encoded_seqs.reshape((encoded_seqs.shape[0], encoded_seqs.shape[1])).tolist()

                scores = []
                for i,r in sub_df.iterrows():
                    scores.append(r.objective_score)

                # Kmedoids clustering based on hamming distance of encoded_seqs
                # encoded_seqs is the integer encoding of the sequence concatenated to the [objective_score]*len(encoded_seq)
                encoded_seqs = [encoded_seqs[idx] + [scores[idx]]*len(encoded_seqs[idx]) for idx in range(len(encoded_seqs))]
                sub_df.loc[:, 'encoded_seqs'] = encoded_seqs
                kmedoids = KMedoids(n_clusters=cluster_number, metric='hamming', method=kmedoids_fxn, init='heuristic', max_iter=3000).fit(encoded_seqs)



                # Determine cluster IDs and select best variant from each cluster
                sub_df.loc[:, 'cluster_id'] = kmedoids.labels_
                best_variant_idxs = []
                for cluster in range(len(kmedoids.medoid_indices_)):
                    cluster_df = sub_df.loc[sub_df.cluster_id == cluster]
                    index = cluster_df.loc[cluster_df.sort_val == cluster_df.sort_val.max()].index[0]
                    best_variant_idxs.append(index)

                best_variants = sub_df.loc[best_variant_idxs]
                best_variants.sort_values(by='sort_val', ascending=False, inplace=True)
                best_variants.reset_index(inplace=True,drop=True)
                best_variants = best_variants.iloc[:select_top_x]
                best_variants = best_variants.drop('encoded_seqs', axis=1)
                out_df = best_variants.copy()

                # # Determine cluster centers
                # cluster_centers = []
                # for i, r in sub_df.iterrows():
                #     for km in kmedoids.cluster_centers_:
                #         if r.encoded_seqs == km.tolist():
                #             cluster_centers.append(r)
                # # Sort cluster centers and select top x to characterize
                # cluster_centers = pd.DataFrame(cluster_centers)
                # cluster_centers.sort_values(by='sort_val', ascending=False, inplace=True)
                # cluster_centers.reset_index(inplace=True,drop=True)
                # top_centers = cluster_centers.iloc[:select_top_x]
                # top_centers = top_centers.drop('encoded_seqs', axis=1)
                # out_df = top_centers.copy()

            # Save top designs to excel
            out_df.to_pickle(f'{dir_}/{objective}_top_{select_top_x}_{mut_option}_mutant.pkl')

            # Plot hamming distance of top designs to illustrate positional diversity
            hams = calculate_hamming(out_df.sequence.tolist())
            name = f'{mut_option}_mut'
            plot(hams,name)

def plot_averages(df):

    objectives = list(set(df.objective))
    for order in list(set(df.order)) + ['all']:
        sub_df = df.loc[df.order == order] if order != 'all' else df
        g = sns.barplot(data=sub_df, x='objective', y='objective_score', order=sorted(list(set(sub_df.objective))), errwidth=1, color='b', capsize=0.2)
        g.figure.set_figwidth(len(objectives)/4)
        x_labels = g.get_xticklabels()
        g.set_xticklabels(x_labels, rotation=90)
        plt.tight_layout()
        plt.show()



def combsorted(list(set(sub_df.order))ine_tops(base_dir):
    # After combining all subsets, this function will combine these datasets into a single file
    # load all tops and combine
    files = glob.glob(f'{base_dir}*/*top*.pkl')

    # Load in top subsets
    df_list = []
    for file in files:
        df = pd.read_pickle(file)
        df_list.append(df)

    df = pd.concat(df_list)
    df.reset_index(inplace=True,drop=True)
    hams = calculate_hamming(df.sequence.tolist())
    df.sort_values(by='sort_val', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    name = f'{base_dir}top_subsets_for_everything'
    df.to_pickle(f'{name}.pkl')

    # Plot hamming distance and positonal frequency for all sequences
    plot_pos_freq(df,WT, 'all_all')
    plot(hams,'all_all')

    # Plot hammind distance and positional frequency for each mutaitonal distance
    for mut in list(set(df.num_mutations.tolist())):
        mut_df = df.copy().loc[df.num_mutations == mut]
        hams = calculate_hamming(mut_df.sequence.tolist())

        plot_pos_freq(mut_df, WT, f'{mut}_all')
        plot(hams, f'{mut}_all')

    plot_averages(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments for batch processing
    parser.add_argument("--dir",
                        help="directory of objective to focus on, will do all mutants for this dir"
                             "use dir='subdirname/*' if choosing to process all directories",
                        type=str,
                        default='SA_outputs/RI_10part_trial/*')#'"SA_outputs/RI_CNN_all/1xxxx")
    parser.add_argument("--desired_outputs",
                        help="number of outputs to select if clustering",
                        type=int,
                        default=5)
    parser.add_argument("--min_percentile",
                        help="filter to select only variants above this percentile",
                        type=int,
                        default=0)
    parser.add_argument("--combine_tops_only",
                        help="True indicates combining tops for all subsets, False indicates determining"
                             "tops for the subset indicated by args.dir",
                        action="store_true")

    parser.add_argument("--should_cluster",
                        help="Whether sequences should be clustered or not",
                        action="store_true")
    parser.add_argument("--endpoints_only",
                        help="Should only the endpoint sequences be utilized?",
                        action="store_true")


    args = parser.parse_args()
    base_dir = '/'.join(args.dir.split('/')[:-1]) +'/' if '*' not in args.dir else args.dir[:-1]

    if not args.combine_tops_only:
        # The following line was run to combine a single objective (in order to speed up processing)
        main(directory=args.dir,base_dir=base_dir,
            desired_outputs=args.desired_outputs, min_percentile=args.min_percentile)

    # Combine all top datasets and if there are local optima, determine scores of base variants
    combine_tops(base_dir)

    print('Finished')

