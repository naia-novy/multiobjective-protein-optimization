import torch
import numpy as np
import copy
from protein_mpnn_utils import StructureDatasetPDB, ProteinMPNN, _scores, tied_featurize, parse_PDB

# from esmfold import ESMFold
torch.set_num_threads(4)


class ProteinMPNNOptimizer:
    def __init__(self, checkpoint_path, pdb_path, n_iterations=10, prepend_sequence='', append_sequence=''):
        # initialize
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(self.alphabet, range(21)))

        # set hyperparameters
        hidden_dim = 128
        self.prepend_sequence = prepend_sequence
        self.append_sequence = append_sequence
        self.n_iterations = n_iterations

        # load pdb
        self.pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
        self.dataset_valid = StructureDatasetPDB(self.pdb_dict_list, truncate=None, max_length=10000)
        self.all_chain_list = [item[-1:] for item in list(self.pdb_dict_list[0]) if
                          item[:9] == 'seq_chain']  # ['A','B', 'C',...]
        self.designed_chain_list = []#self.all_chain_list # assume entire pdb
        self.fixed_chain_list = [letter for letter in self.all_chain_list if letter not in self.designed_chain_list]
        self.chain_id_dict = {}
        self.chain_id_dict[self.pdb_dict_list[0]['name']] = (self.designed_chain_list, self.fixed_chain_list)

        # load model
        checkpoint = torch.load(checkpoint_path)
        model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                            hidden_dim=hidden_dim, k_neighbors=checkpoint['num_edges'])
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.eval()



    def predict(self, sequences):
        if type(sequences) == str:
            sequences = [sequences]
        predicted_scores = []
        for sequence in sequences:
            native_score_list = []
            global_native_score_list = []
            sequence = self.prepend_sequence + sequence + self.append_sequence
            sequence = sequence * len(self.all_chain_list) # assume exact homomer (check PDB file that these are the same)
            for ix, protein in enumerate(self.dataset_valid):

                batch_clones = [copy.deepcopy(protein) for i in range(self.n_iterations)] # only use batch size 1
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, self.device, self.chain_id_dict)
                input_seq_length = len(sequence)

                # Assume homomer
                S_input = torch.tensor([self.alphabet_dict[AA] for AA in sequence], device=self.device)[None, :].repeat(
                    X.shape[0], 1)
                S[:, :input_seq_length] = S_input  # assumes that S and S_input are alphabetically sorted for masked_chains

                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = self.model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask * chain_M * chain_M_pos
                scores = _scores(S, log_probs, mask_for_loss)
                native_score = scores.cpu().data.numpy()
                native_score_list.append(native_score)
                global_scores = _scores(S, log_probs, mask)
                global_native_score = global_scores.cpu().data.numpy()
                global_native_score_list.append(global_native_score)
                global_native_score = np.concatenate(global_native_score_list, 0)

                global_ns_mean = global_native_score.mean()

                predicted_scores.append(global_ns_mean)

        return predicted_scores


#
# class ESMFoldLoader:
#     def __init__(self):
#
#         # load model
#         self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#         checkpoint = torch.load('data/pre_trained_models/esmfold_3B_v1.pt')
#         model = ESMFold(checkpoint=checkpoint)
#         model.to(self.device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         self.model = model.eval()
#
#
#
#     def predict(self, sequence):
#         output = self.model.infer_pdb(sequence)
#         print(output)
#         adj_plddt = output.b_factor.mean()/100.0
#
#         return adj_plddt



if __name__ == '__main__':
    # ESMFold_model = ESMFoldLoader()
    MPNN = ProteinMPNNOptimizer(n_iterations=20, checkpoint_path='data/pre_trained_models/Protein_MPNN_v_48_010.pt', pdb_path='data/T7RBD_all_separate/T7_trimer_Cterm.pdb')

    # make predictions for first design attempt

    import pandas as pd
    df = pd.read_excel('data/motifs_averaged_deep.xlsx')
    df['MPNN_score'] = [0]*len(df)

    for i,r in df.iterrows():
        score = MPNN.predict(r.Sequence)
        df.loc[i, 'MPNN_score'] = score[0]
        print(i/len(df))

    df.to_pickle('motifs_averaged_deep_MPNN.pkl')


    # # test which MPNN model does best on the WT sequence to decide which one to use during simulated annealing
    # for model in ['data/pre_trained_models/Protein_MPNN_v_48_002.pt','data/pre_trained_models/Protein_MPNN_v_48_010.pt',
    #               'data/pre_trained_models/Protein_MPNN_v_48_020.pt', 'data/pre_trained_models/Protein_MPNN_v_48_030.pt',]:
    #     MPNN = ProteinMPNNOptimizer(n_iterations=10, checkpoint_path=model,
    #                                 pdb_path='data/T7RBD_all_separate/T7_trimer_Cterm.pdb')
    #
    #     print(model, MPNN.predict("QVWSGSAGGGVSVTVSQDLRFRNIWIKCANNSWNFFRTGPDGIYFIASDGGWLRFQIHSNGLGFKNIADSRSVPNAIMVENE")[0])

    # results indicate to use 010 noising
    # data/pre_trained_models/Protein_MPNN_v_48_002.pt 1.1910646
    # data/pre_trained_models/Protein_MPNN_v_48_010.pt 1.1236556
    # data/pre_trained_models/Protein_MPNN_v_48_020.pt 1.2347293
    # data/pre_trained_models/Protein_MPNN_v_48_030.pt 1.4039972
