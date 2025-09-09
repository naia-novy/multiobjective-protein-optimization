import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import argparse
import glob
import random
import seaborn as sns
import uuid
import copy
import tasks
import time
import scipy

from datamodules import DMSDatasetDataModule
import utils
import parse_args
from additional_predictors import ProteinMPNNOptimizer
from scipy.optimize import basinhopping

# set seaborn settings to produce high quality figures
sns.set(rc={"figure.dpi": 50, 'savefig.dpi': 100})
sns.set_style("ticks")


def generate_all_point_mutants(WT,AA_options):
    # generate every possible single point mutant
    all_mutants = []
    for pos in range(len(WT)):
        for aa in AA_options[pos]:
            if WT[pos] != aa: # not mutating to self
                mut = WT[pos]+str(pos)+aa
                all_mutants.append(mut)
    return all_mutants


def mut2seq(WT, mutations):
    mutant_seq = WT
    if type(mutations) is str:
        mutations = mutations.split(',')
    for mut in mutations:
        pos = int(mut[1:-1])
        newAA = mut[-1]
        if mut[0] != WT[pos]: print('Warning: WT residue in mutation %s does not match WT sequence' % mut)
        mutant_seq = mutant_seq[:pos] + newAA + mutant_seq[pos+1:]
    return mutant_seq

def seq2mut(sequences, wt_aa):
    """ Convert a list of sequence strings to a list of mutant variants """
    variants = []
    for seq in sequences:
        mutations = []
        for i,aa in enumerate(seq):
            if aa != wt_aa[i]:
                mutation = f'{wt_aa[i]}{i}{aa}'
                mutations.append(mutation)
        variants.append(','.join(mutations))

    return variants

def find_top_n_mutations(seq2fitness,all_mutants,WT,n=10):
    
    # evaluate fitness of all single mutants from WT
    single_mut_fitness = []
    for mut in all_mutants:
        seq = mut2seq(WT,(mut,))
        fit, stats = seq2fitness(mut,WT)
        single_mut_fitness.append((mut,fit))

    # find the best mutation per position
    best_mut_per_position = []
    for pos in range(len(WT)):
        best_mut_per_position.append(max([m for m in single_mut_fitness if int(m[0][1:-1])==pos],key = lambda x: x[1]))
    
    # take the top n
    sorted_by_fitness = sorted(best_mut_per_position, key = lambda x: x[1], reverse=True)
    topn = [m[0] for m in sorted_by_fitness[:n]]
    topn = tuple([n[1] for n in sorted([(int(m[1:-1]),m) for m in topn])]) # sort by position
    
    return topn


def generate_random_mut(WT, AA_options, num_mut):
    # Want to make the probability of getting any mutation the same:
    AA_mut_options = []
    for WT_AA, AA_options_pos in zip(WT, AA_options):
        if WT_AA in AA_options_pos:
            options = list(AA_options_pos).copy()
            options.remove(WT_AA)
            AA_mut_options.append(options)
    mutations = []

    for n in range(num_mut):
        
        num_mut_pos = sum([len(row) for row in AA_mut_options])
        prob_each_pos = [len(row)/num_mut_pos for row in AA_mut_options]
        rand_num = random.random()
        for i, prob_pos in enumerate(prob_each_pos):
            rand_num -= prob_pos
            if rand_num <= 0:
                mutations.append(WT[i]+str(i)+random.choice(AA_mut_options[i]))
                AA_mut_options.pop(i)
                AA_mut_options.insert(i, [])
                break
    return ','.join(mutations)


def generate_random_seq(WT, AA_options, num_mut):
    # Want to make the probability of getting any mutation the same:
    AA_mut_options = []
    for WT_AA, AA_options_pos in zip(WT, AA_options):
        if WT_AA in AA_options_pos:
            options = list(AA_options_pos).copy()
            options.remove(WT_AA)
            AA_mut_options.append(options)

    mut_seq = list(WT)

    for n in range(num_mut):
        num_mut_pos = sum([len(row) for row in AA_mut_options])
        prob_each_pos = [len(row) / num_mut_pos for row in AA_mut_options]
        rand_num = random.random()
        for i, prob_pos in enumerate(prob_each_pos):
            rand_num -= prob_pos
            if rand_num <= 0:
                mut_seq[i] = random.choice(AA_mut_options[i])
                AA_mut_options.pop(i)
                AA_mut_options.insert(i, [])
                break
    return ''.join(mut_seq)

class SA_optimizer:
    def __init__(self, models, dm, optim_objective, fxn, specificity_bias, AA_options, fig_name,
                 num_mut, T0, predictions_for_train_ds=None, strict_num_mut=True, compute_ProteinMPNN=False,
                 compute_ESMFold=False, merge_with_bias=False, mut_rate=1, nsteps=1000, cool_sched='log', compute_percentiles=False, weights=None):

        self.model_args = model_args
        self.AA_options = AA_options
        self.num_mut = num_mut
        self.strict_num_mut = strict_num_mut
        self.mut_rate = mut_rate
        self.nsteps = nsteps
        self.cool_sched = cool_sched
        self.models = models 
        self.optim_objective = optim_objective
        self.merge_with_bias = merge_with_bias
        self.all_col_names = dm.output_col_names
        self.T0 = T0

        if self.merge_with_bias:
            self.output_col_names = list(sorted(set(['_'.join(name.split('_')[1:]) for name in dm.output_col_names])))
        else:
            self.output_col_names = self.all_col_names

        print(f"Order of columns for objectives is {', '.join(self.output_col_names)}\n")

        if self.merge_with_bias and weights != None:
            self.merge_with_bias = pd.read_csv(weights, index_col=0)

        self.optim_min_objectives = [self.output_col_names[i] for i in range(len(optim_objective)) if optim_objective[i] == '0']
        self.optim_max_objectives = [self.output_col_names[i] for i in range(len(optim_objective)) if optim_objective[i] == '1']
        self.max_output_idx = len(optim_objective)
        self.fxn = fxn
        self.fig_name = fig_name
        self.specificity_bias = specificity_bias if len(self.optim_max_objectives+self.optim_min_objectives) > 1 else 1


        # dm params
        self.encoding = dm.encoding
        self.ds_name = dm.ds_name
        self.frame_shift = dm.frame_shift
        self.ncomp = dm.ncomp
        self.batch_converter = dm.batch_converter
        ds_info = utils.load_ds_index()[self.ds_name]
        self.WT = ds_info["wt_aa"]
        self.pdb_path = ds_info["pdb_path"]
        if 'percentile' in self.fxn:
            if predictions_for_train_ds == None:
                raise Exception('File path of model predictions is required for objective function utilizing percentiles')
            self.og_ds = utils.load_ds(predictions_for_train_ds)
            self.og_ds = self.og_ds.loc[:, [i for i in self.og_ds.columns if 'score' in i]]

            if compute_percentiles:
                self.ds = pd.DataFrame(columns = self.og_ds.columns)

                # convert to interpolated percentiles to allow for optimization even after making variants predicted
                # to be better than anything observed in the dataset. Due to a change in magnitude at this point,
                # it will be favorable for all objectives to get to this point before maximizing beyond the observed fitness scores

                for col in self.ds.columns:
                    if col[0] != 'p':
                        continue

                    x = self.og_ds.loc[:, col].tolist() + [-50, 50]
                    y = np.vectorize(lambda i: scipy.stats.percentileofscore(x, i))(x)
                    fit = scipy.interpolate.interp1d(x, y, kind='linear')
                    x2 = np.linspace(-50, 50, num=100000)
                    y2 = fit(x2)
                    self.ds.loc[:, col] = y2
                    # fig, ax = plt.subplots()
                    # plt.plot(x, y, 'o')
                    # plt.plot(x2, y2, 'r-')
                    # plt.show()
            else:
                self.ds = self.og_ds

        else:
            self.ds = None

        self.compute_ProteinMPNN = compute_ProteinMPNN
        self.compute_ESMFold = compute_ESMFold
        if self.compute_ProteinMPNN:
            self.ProteinMPNN = ProteinMPNNOptimizer(n_iterations=3, checkpoint_path='data/pre_trained_models/Protein_MPNN_v_48_010.pt', pdb_path=self.pdb_path)
        if self.compute_ESMFold:
            self.ESMFold = None #torch.load('data/pre_trained_models/esmfold_v1.pt')

    def optimize(self, start_seq=None, num_designs=1, seed=0):
        self.num_designs = num_designs
        random.seed(seed)
        
        if start_seq is None:
            start_seq = generate_random_seq(self.WT, self.AA_options, self.num_mut).split(',')
        if type(start_seq) != list:
            start_seq = [start_seq]

        start_seq = start_seq * self.num_designs
            
        all_mutants = generate_all_point_mutants(self.WT, self.AA_options)

        assert ((self.cool_sched == 'log') or (self.cool_sched == 'lin')), 'cool_sched must be \'log\' or \'lin\''
        if self.cool_sched == 'log':
            temp = np.logspace(self.T0, -7.0, self.nsteps)
        elif self.cool_sched == 'lin':
            temp = np.linspace(self.T0, 1e-7, self.nsteps)

        fit = self.seq2fitness(start_seq)

        current_seq = [start_seq, fit.copy()]
        self.best_seq = [start_seq, fit.copy()]
        self.fitness_trajectory = np.array([[fit.copy(), fit.copy()]]) # fitness_trajectory = [best_fit, current_fit]

        # for loop over decreasing temperatures
        check_convergence_distance = np.max([int(0.2*len(temp)), 2000])
        for iter in range(len(temp)):
            x = time.time()


            T = temp[iter]
            # postions that already have a mutation
            current_seq_list = list(list(current_seq[0][i]) for i in range(self.num_designs))
            occupied_positions = []
            for seq_idx in range(self.num_designs):
                positions = []
                for i in range(len(self.WT)):
                    if self.WT[i] != current_seq_list[seq_idx][i]:
                        positions.append(i)

                occupied_positions.append(positions)

            # choose the number of mutations to make to current sequence
            n = np.random.poisson(self.mut_rate)
            n = min([self.num_mut-1,max([1,n])]) # bound n within the range [1,num_mut-1]

            for seq_idx in range(self.num_designs):
                # remove random mutations from current sequence until it contains (num_mut-n) mutations
                while len(occupied_positions[seq_idx]) > (self.num_mut-n):
                    revert_pos = random.choice(occupied_positions[seq_idx])
                    current_seq_list[seq_idx][revert_pos] = self.WT[revert_pos]
                    occupied_positions[seq_idx].remove(revert_pos)

                # if not strict_num_mut, the total number of mutaitons can decrease by 1 each time
                # this adds the option of increasing total current mutations by 1 if there is less than the max
                if len(occupied_positions[seq_idx]) < (self.num_mut-n):
                    n = np.random.poisson(self.mut_rate+1)
                    n = min([self.num_mut - 1, max([1, n])])

                if self.strict_num_mut:
                    mut_options = [m for m in all_mutants if m[1:-1] not in occupied_positions[seq_idx]] # mutations at unoccupied positions
                    while len(occupied_positions[seq_idx]) < self.num_mut:
                        mutation = random.choice(mut_options)
                        current_seq_list[seq_idx][int(mutation[1:-1])] = mutation[-1]
                        occupied_positions[seq_idx].append(mutation[1:-1])
                        mut_options = [m for m in all_mutants if m[1:-1] not in occupied_positions[seq_idx]]
                else:
                    for i in range(n): # will be 1
                        mutation = random.choice(all_mutants)
                        current_seq_list[seq_idx][int(mutation[1:-1])] = mutation[-1]

            # evaluate fitness of new mutant
            fitness = self.seq2fitness([''.join(i) for i in current_seq_list])

            # Simulated annealing acceptance criteria: 
            #   If mutant is better than current seq, move to mutant seq
            #   If mutant is worse than current seq, move to mutant with some exponentially decreasing probability with delta_F 
            delta_F = fitness - current_seq[1] # new seq is worse if delta_F is neg.

            for seq_idx in range(self.num_designs):
                if np.exp(min([0 , delta_F[seq_idx]/(T)])) > random.random():
                    current_seq[0][seq_idx] = ''.join(current_seq_list[seq_idx].copy())
                    current_seq[1][seq_idx] = fitness[seq_idx]

                # if mutant is better than best sequence, reassign best sequence
                if fitness[seq_idx] > self.best_seq[1][seq_idx]:
                    self.best_seq[0][seq_idx] = current_seq[0][seq_idx]
                    self.best_seq[1][seq_idx] = fitness[seq_idx]

            # store the current fitness
            self.fitness_trajectory = np.append(self.fitness_trajectory.copy(),np.array([[self.best_seq[1].copy().tolist(), current_seq[1].copy().tolist()]]), axis=0)




            if iter % 1000 == 0 and iter != 0:
                self.save_seq_data(self.best_seq, iter)

            y = time.time()
            # print((y - x)/self.num_designs)
            # check to see if optimization has converged (measured by whether the best fitness has not changed for 10% of total iterations)
            if iter > check_convergence_distance:
                if np.all(self.best_seq[1] == self.fitness_trajectory[iter-check_convergence_distance][0]):
                    print('BREAK___________________________________________________________')
                    print(iter)
                    break

            
        self.plot_trajectory(f'{self.fig_name}.png')
            
        return self.best_seq # returns [best_mut, best_fit]
            
    def plot_trajectory(self, savefig_name=None):
        start = int(np.min([100, (len(self.fitness_trajectory)-1)/10]))
        # if start == len(self.fitness_trajectory)+1:
        #     start = 0
        for i in range(self.fitness_trajectory.shape[2]):
            plt.plot(np.array(self.fitness_trajectory)[start:,0,i])
            plt.plot(np.array(self.fitness_trajectory)[start:,1,i])
            plt.xlabel('Step')
            plt.ylabel('Fitness')
            plt.legend(['Best mut found', 'Current mut'])
            if savefig_name is None:
                plt.show()
            else:
                plt.savefig(savefig_name[:-4] + f"_{i}.png")

            plt.show()
            plt.close()

    def seq2fitness(self, sequence):
        scores = np.empty(shape=(0, self.num_designs))
        for model in self.models:
            output = model.predict_from_checkpoint(sequence, self.frame_shift, self.ds, self.merge_with_bias)
            score = self.objective_fxn(output)
            scores = np.vstack((scores, score))

        score = np.median(scores,axis=0)

        if self.compute_ProteinMPNN:
            MPNN_score = self.ProteinMPNN.predict(sequence[0])
            # score = np.tanh(MPNN_score[0])*score
            score = score/MPNN_score[0]#(np.exp(-1/MPNN_score[0])) # convert neg log likilihood to average probability
            print(np.exp(-1/MPNN_score[0]))
        if self.compute_ESMFold:
            ESM_plddt = 1
            score = ESM_plddt*score

        return score

    def objective_fxn(self, output):
        # Determine which scores to consider for objective fxn
        max_values = [output[i].to_list() for i in self.optim_max_objectives]
        min_values = [output[i].to_list() for i in self.optim_min_objectives]

        # Append 0 if lists are empty
        if len(max_values) == 0:
            max_values.append([0]*self.num_designs)
        if len(min_values) == 0:
            min_values.append([0]*self.num_designs)

        max_values, min_values = np.array(max_values), np.array(min_values)

        # Objective function for simulated annealing
        if self.fxn == 'gap':
            score = (self.specificity_bias*max_values.min(axis=0) - (1-self.specificity_bias)*min_values.max(axis=0))
        elif self.fxn == 'sum':
            score =  (self.specificity_bias*max_values.sum(axis=0) - (1-self.specificity_bias)*min_values.sum(axis=0))
        elif self.fxn == 'utopia_gap_percentile':
            # compute euclidian distance of percentile_gap_score for utopia
            # take inverse because this script is set up for maximization
            max_values, min_values = np.array(max_values), np.array(min_values)
            score = [100.0]*self.num_designs - np.sqrt(((max_values.min(axis=0) - 100.0)**2 + (min_values.max(axis=0) - 0.0)**2)/2)
        else:
            raise ValueError("Objective function '{}' not implemented!".format(self.fxn))

        return score


    def save_seq_data(self, out, iteration_num):

        for seq_idx in range(self.num_designs):
            muts = seq2mut([out[0][seq_idx]],self.WT)[0]
            data = [out[0][seq_idx]] + [muts] + [len(muts.split(','))] + [self.model_args.log_dir_base.split('/')[-1]] + [args.outputs_to_maximize] + [args.fitness_fxn] + [iteration_num] + [out[1][seq_idx]]

            df = pd.DataFrame([data], columns=['sequence', 'mutations', 'num_mutations', 'model_name','objective', 'fxn', 'num_iter', 'objective_score'])
            df = self.get_scores(df)
            df.to_csv(f'{fig_name}_{iteration_num}_{seq_idx}.csv')

    def get_scores(self,df):

        cols = ['p' + i for i in self.output_col_names]
        cols_std = [i+'_std' for i in cols]

        ProteinMPNN = ProteinMPNNOptimizer(n_iterations=20,
                                                checkpoint_path='data/pre_trained_models/Protein_MPNN_v_48_010.pt',
                                                pdb_path=self.pdb_path)

        for i,r in df.iterrows():
            predictions = []
            objective_outputs = []
            MPNN_score = ProteinMPNN.predict(r.sequence)

            for model in self.models:
                output = model.predict_from_checkpoint([r.sequence], self.frame_shift, self.ds, self.merge_with_bias)

                score = self.objective_fxn(output)
                output = [output[i] for i in self.output_col_names]
                objective_outputs.append(score)
                predictions.append(output)

            predictions = np.array(predictions)[0]
            objective_outputs = np.array(objective_outputs)

            stats_ = [np.max(objective_outputs),
                     np.min(objective_outputs),
                     np.mean(objective_outputs),
                     np.std(objective_outputs),
                     MPNN_score[0]]

            df.loc[i, ['max_ensemble_score', 'min_ensemble_score','mean_ensemble_score', 'std_ensemble_score', 'MPNN_score']] = stats_
            df.loc[i, ['MPNN_constrained', 'ESMFold_constrained']] = [self.compute_ProteinMPNN, self.compute_ESMFold]
            df.loc[i, cols] = np.mean(predictions, axis=1)
            df.loc[i, cols_std] = np.std(predictions, axis=1)

        # save unaveraged scores
        if self.output_col_names != self.all_col_names:
            cols = ['p' + i for i in self.all_col_names]
            df[cols] = [np.NaN] * len(cols)

            for i, r in df.iterrows():
                predictions = []
                for model in self.models:
                    output = model.predict_from_checkpoint([r.sequence], self.frame_shift, None, False)
                    output = [output['p' + i] for i in self.all_col_names]
                    predictions.append(output)

                predictions = np.array(predictions)[0]
                df.loc[i, cols] = np.mean(predictions, axis=1)

        return df

    def compute_acceptance_rate(self, n_iter, T0, start_seq=None, seed=0):
        random.seed(seed)
        acceptance = []

        if start_seq is None:
            start_seq = generate_random_seq(self.WT, self.AA_options, self.num_mut).split(',')

        all_mutants = generate_all_point_mutants(self.WT, self.AA_options)

        assert ((self.cool_sched == 'log') or (self.cool_sched == 'lin')), 'cool_sched must be \'log\' or \'lin\''
        if self.cool_sched == 'log':
            temp = np.logspace(T0, -7.0, self.nsteps)
        elif self.cool_sched == 'lin':
            temp = np.linspace(T0, 1e-7, self.nsteps)

        fit = self.seq2fitness(start_seq)

        current_seq = [start_seq, fit]
        self.best_seq = [start_seq, fit]

        # for loop over decreasing temperatures
        check_convergence_distance = int(0.1 * len(temp))
        for iter in range(n_iter):
            T = temp[iter]
            # postions that already have a mutation
            occupied_positions = [i for i in range(len(self.WT)) if self.WT[i] != current_seq[0][0][i]]
            # print('top', len(occupied_positions))
            current_seq_list = list(current_seq[0][0])

            # choose the number of mutations to make to current sequence
            n = np.random.poisson(self.mut_rate)
            n = min([self.num_mut - 1, max([1, n])])  # bound n within the range [1,num_mut-1]

            # remove random mutations from current sequence until it contains (num_mut-n) mutations
            while len(occupied_positions) > (self.num_mut - n):
                revert_pos = random.choice(occupied_positions)
                current_seq_list[revert_pos] = self.WT[revert_pos]
                occupied_positions.remove(revert_pos)

            # if not strict_num_mut, the total number of mutaitons can decrease by 1 each time
            # this adds the option of increasing total current mutations by 1 if there is less than the max
            if len(occupied_positions) < (self.num_mut - n):
                n = np.random.poisson(self.mut_rate + 1)
                n = min([self.num_mut - 1, max([1, n])])

            if self.strict_num_mut:
                mut_options = [m for m in all_mutants if
                               m[1:-1] not in occupied_positions]  # mutations at unoccupied positions
                while len(occupied_positions) < self.num_mut:
                    mutation = random.choice(mut_options)
                    current_seq_list[int(mutation[1:-1])] = mutation[-1]
                    occupied_positions.append(mutation[1:-1])
                    mut_options = [m for m in all_mutants if m[1:-1] not in occupied_positions]
            else:
                for i in range(n):  # will be 1
                    mutation = random.choice(all_mutants)
                    current_seq_list[int(mutation[1:-1])] = mutation[-1]

            # evaluate fitness of new mutant
            fitness = self.seq2fitness([''.join(current_seq_list)])

            # Simulated annealing acceptance criteria:
            #   If mutant is better than current seq, move to mutant seq
            #   If mutant is worse than current seq, move to mutant with some exponentially decreasing probability with delta_F
            delta_F = fitness - current_seq[1]  # new seq is worse if delta_F is neg.

            accepted = 0.0
            if np.exp(min([0, delta_F / (T)])) > random.random():
                current_seq = [[''.join(current_seq_list)], fitness]
                accepted = 1.0

            # if mutant is better than best sequence, reassign best sequence
            if fitness > self.best_seq[1]:
                self.best_seq = current_seq

            acceptance.append(accepted)

        # print(sum(acceptance)/n_iter, T0)
        return sum(acceptance)/n_iter # returns acceptance rate
        
    
            
if __name__ == "__main__":

    model_args = parse_args.main()

    parser = argparse.ArgumentParser()
    # Arguments for simulated annealing
    parser.add_argument("--outputs_to_maximize",
                        help="which model outputs to maximize, indicated by binary encoding",
                        type=str,
                        default="101xx")
    parser.add_argument("--num_muts",
                        help="max number of mutations to make from WT seq",
                        type=int,
                        default=6)
    parser.add_argument("--num_steps",
                        help="number of SA steps to take",
                        type=int,
                        default=20000)
    parser.add_argument("--checkpoint_base_dir",
                        help="location of dir with checkpoint files to use",
                        type=str,
                        default='output/ensembles/x')

    parser.add_argument("--predictions_for_train_ds",
                        help="location of file with predictions made for training data for use in percentile calculation",
                        type=str,
                        default='data/pRI_CNN_training_percentile_scores.pkl')
    parser.add_argument("--num_designs",
                        help="number of designs to output",
                        type=int,
                        default=4)
    parser.add_argument("--fitness_fxn",
                        help="what operation to use for caluclating fitness",
                        type=str,
                        default='utopia_gap_percentile')
    parser.add_argument("--specificity_bias",
                        help="Weight (0-1) for favoring speficitiy over funciton. High bias favors function",
                        type=float,
                        default=0.5)
    parser.add_argument("--cooling_schedule",
                        help="log or lin cooling schedule",
                        type=str,
                        default='log')
    parser.add_argument("--seed",
                        help="random seed for reproducability",
                        type=int,
                        default=0)
    parser.add_argument("--WT",
                        help="amino acid sequence of the wildtype protein",
                        type=str,
                        default='QVWSGSAGGGVSVTVSQDLRFRNIWIKCANNSWNFFRTGPDGIYFIASDGGWLRFQIHSNGLGFKNIADSRSVPNAIMVENE')
    parser.add_argument("--pre_computed_weights_for_average",
                        help="file containing weights of each output column to compute average",
                        type=str,
                        default=None)#'data/T7RBD_all_separate/dms_comb_motif_score_weights.csv')


    parser.add_argument("--compute_ProteinMPNN",
                        help="whether to constrain predictions with ProteinMPNN",
                        action="store_true")
    parser.add_argument("--compute_ESMFold",
                        help="whether to constrain predictions with ESMFold",
                        action="store_true")
    parser.add_argument("--strict_num_mut",
                        help="whether num_mut should be considered as max or exact",
                        action="store_true")


    parser.add_argument("--compute_percentiles",
                        help="whether to compute percentiles if loss utilizes percentiles. note that if using a percentile objective function it is expected that percentile scores will be computed or provided in predictions_for_train_ds (can save a lot of time to have these precomputed)",
                        action="store_true")
    parser.add_argument("--should_basinhop",
                        help="whether T0 should be initialized with basinhopping, if False, will use default value (enabling this can increase runtime)",
                        action="store_true")


    parser.add_argument("--output_directory",
                        help="where to store outputs within SA_outputs directory",
                        type=str,
                        default='test')

    parser.add_argument("--output_bias",
                        help="weights to use for averaging in score computation",
                        nargs='+',
                        default=[0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33])

    args, unknown = parser.parse_known_args()

    # input for SA may differ from any biases used during training and stored in provided args file
    model_args.output_bias = [' '.join([str(i) for i in args.output_bias])]

    AA_options = ['ACDEFGHIKLMNPQRSTVWY']*len(args.WT)
    AAs = 'ACDEFGHIKLMNPQRSTVWY-'


    base_check_paths = glob.glob(args.checkpoint_base_dir)
    base_check_paths = [glob.glob(i+'/*') for i in base_check_paths]
    check_paths = [glob.glob(i+'/version_0/checkpoints/*.ckpt') for i in base_check_paths[0]]
    check_paths = [path[0] for path in check_paths if len(path) == 1] # if glob returns none do not select path

    out_dir = f'SA_outputs/{args.output_directory}/{args.outputs_to_maximize}/'
    # os.makedirs('SA_outputs', exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    model_args.log_dir, model_args.num_datasets, model_args.hpopt, model_args.wandb_online = None, 1, False, False
    dm = DMSDatasetDataModule(**vars(model_args))

    # set up the task
    task = tasks.Task[model_args.task_name].cls(dm=dm,
                                          val_df=None,
                                          **vars(model_args))

    trained_models = []
    print('Loading checkpoints...')
    for check_path in check_paths:
        model = copy.deepcopy(task.model)

        checkpoint = torch.load(check_path)

        state_dict = checkpoint['state_dict'].copy()
        for k, v in checkpoint['state_dict'].items():
            state_dict['.'.join(k.split('.')[1:])] = checkpoint['state_dict'][k]
            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        trained_models.append(model)
    print('Done loading models.')




    T0 = 10.0
    if args.should_basinhop:
        # determine optimal starting temperature using basin hoppint
        opt_models = trained_models[0]  # use only first model of ensemble for convergence speedup
        target_acceptance_rate = 0.3
        n_iter = 40
        n_init = 10
        SA = SA_optimizer(trained_models, dm, args.outputs_to_maximize, args.fitness_fxn, args.specificity_bias,AA_options, 'fig_name',
                          args.num_muts, T0, predictions_for_train_ds=args.predictions_for_train_ds, strict_num_mut=args.strict_num_mut,
                          compute_ProteinMPNN=args.compute_ProteinMPNN,
                          compute_ESMFold=args.compute_ESMFold, merge_with_bias=True, mut_rate=1, nsteps=args.num_steps,
                          cool_sched=args.cooling_schedule, weights=args.pre_computed_weights_for_average)



        func = lambda x: ((target_acceptance_rate - SA.compute_acceptance_rate(n_iter+1, x[0], seed=args.seed)))**2 # add setoff to prevent exact guesses by chance
        opt = basinhopping(func, [T0], niter=n_init, minimizer_kwargs={'method':'Nelder-Mead'}, seed=args.seed)

        T0 = opt.x
        print(f"T0 = {T0}")

    rand = str(uuid.uuid4()).split('-')[0]
    fig_name = f'{out_dir}{args.outputs_to_maximize}_{args.num_muts}_{args.num_steps}_{args.fitness_fxn}_{args.cooling_schedule}_{rand}'

    SA = SA_optimizer(trained_models, dm, args.outputs_to_maximize, args.fitness_fxn, args.specificity_bias,AA_options, fig_name,
                      args.num_muts, T0, predictions_for_train_ds=args.predictions_for_train_ds, strict_num_mut = args.strict_num_mut, compute_ProteinMPNN = args.compute_ProteinMPNN,
                      compute_ESMFold = args.compute_ESMFold, merge_with_bias=True, mut_rate=1, nsteps=args.num_steps,
                      cool_sched=args.cooling_schedule,
                      compute_percentiles=args.compute_percentiles, weights=args.pre_computed_weights_for_average)


    print('done with init')
    out = SA.optimize(seed=args.seed, num_designs=args.num_designs) # [best_mut, best_fit]
    print('done with opt')

    SA.save_seq_data(out, args.num_steps)
