from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data as data_utils
from torch import Tensor
import pytorch_lightning as pl
import pickle

import utils
import splits
import encode


def handle_quality_treatment(kwargs):
    # set up to handle different options for treatment of quality that are not currently implemented
    if kwargs['quality_treatment'] == "quality_filter":
        kwargs['quality'] = True
        kwargs['regression_thresholds'] = None
    elif kwargs['quality_treatment'] == "threshold_regression":
        kwargs['quality'] = False
        kwargs['regression_thresholds'] = [float(i) for i in kwargs['regression_thresholds'][0].split()]
    elif kwargs['quality_treatment'] == 'include_all' or not kwargs['quality_available']:
        kwargs['quality'] = False
        kwargs['regression_thresholds'] = None
    else:
        raise Exception(f"Quality treatment '{kwargs['quality_treatemnt']}' is not implemented")
    return kwargs


def handle_loss_method(kwargs):
    if kwargs['loss_method'] == "regression_only":
        kwargs['classification'] = False
        kwargs['weighted_classification'] = False
    elif kwargs['loss_method'] == "add_classification":
        kwargs['classification'] = True
        kwargs['weighted_classification'] = False
    elif kwargs['loss_method'] == "add_weighted_classification":
        kwargs['classification'] = True
        kwargs['weighted_classification'] = True
    else:
        raise Exception(f"Loss method '{kwargs['loss_method']}' is not implemented")
    return kwargs



class DMSDataset(torch.utils.data.Dataset):
    """ Dataset for DMS data, in-memory, similar to PyTorch's TensorDataset, supports dict return value """

    def __init__(self, inputs: Tensor, scores: Tensor,
                 weight: Tensor, qual: Tensor, class_name: Tensor, class_weight: Tensor) -> None:
        self.inputs = inputs
        self.scores = scores
        self.weight_in = weight
        self.qual_in = qual
        self.class_name_in = class_name
        self.class_weight_in = class_weight

    def __getitem__(self, index):
        out_dict = {"inputs": self.inputs[index],
                    "scores": self.scores[index],
                    'weight': self.weight_in[index],
                    'qual': self.qual_in[index],
                    'class_name': self.class_name_in[index],
                    'class_weight': self.class_weight_in[index]}

        return out_dict

    def __len__(self):
        return self.inputs.size(0)

class DMSDatasetDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        # basic dataset and encoding info
        self.ds_name = kwargs['ds_name']
        self.frame_shift = kwargs['frame_shift']
        self.augmentation_factor = kwargs['augmentation_factor']
        self.ncomp = kwargs['ncomp']
        self.weight_decay = float(kwargs['weight_decay']) if kwargs['weight_decay'] != None else kwargs['weight_decay']
        self.scaling_weight = float(kwargs['scaling_weight'])
        self.lr = kwargs['lr']
        self.log_dir = kwargs['log_dir']

        # process quality treatment to handle ['quality'] and ['classification','weighted_classification', 'threshold_regression'] flags
        kwargs = handle_quality_treatment(kwargs)
        kwargs = handle_loss_method(kwargs)

        # set conditional flags describing how/which data is available for use
        self.weighted = kwargs['weighted']
        self.quality = kwargs['quality']
        self.quality_treatment = kwargs['quality_treatment']
        self.quality_available = kwargs['quality_available']
        self.classification = kwargs['classification']
        self.weighted_classification = kwargs['weighted_classification']
        self.regression_thresholds = kwargs['regression_thresholds']

        # define names of targets
        target_names_1 = kwargs['target_names']
        target_names_2 = kwargs['target_names'][0].split()
        if len(target_names_2) > len(target_names_1): target_names = target_names_2
        else: target_names = target_names_1
        target_names_is_list = isinstance(target_names, list) or isinstance(target_names, tuple)

        # number of model output tasks (classificaiton + regression) and names
        self.objective_names = target_names
        self.num_objectives = len(self.objective_names)

        # Set up bias for each output
        self.output_bias = kwargs['output_bias'][0]
        self.output_bias = [self.output_bias] if self.output_bias == None else self.output_bias.split(' ')
        self.bias_dict = None # this is only needed for hpopt, but still needs to be defined to load in task
        if len(self.output_bias) == 1:
            self.output_bias = [1]*self.num_objectives
        elif len(self.output_bias) == len(target_names):
            self.output_bias = [float(i) for i in self.output_bias]
            #self.output_bias = [i/sum(self.output_bias) for i in self.output_bias]
        else:
            raise Exception('Invalid input for output_bias')

        # set score and output names
        output_col_names_1 = kwargs['output_col_names']
        output_col_names_2 = kwargs['output_col_names'][0].split()
        if len(output_col_names_2) > len(output_col_names_1):
            self.output_col_names = output_col_names_2
        else:
            self.output_col_names = output_col_names_1
        self.score_names = self.output_col_names[:self.num_objectives]

        # set classification names if necessary and adjust num_objetives
        if self.classification:
            self.class_name = self.output_col_names[self.num_objectives:]
            self.num_objectives *= 2
        else:
            self.output_col_names = self.output_col_names[:self.num_objectives]

        # define names of classes, quality, and var columns (these will not be used if flags for these
        # parameters are not set, but will still be defined)
        self.qual = [s + '_qual' for s in self.objective_names] if target_names_is_list else [
            self.objective_names + '_qual']
        self.weight = [s + '_weight' for s in self.objective_names] if target_names_is_list else [
            self.objective_names + '_weight']
        self.class_weight = [s + '_class_weight' for s in self.objective_names] if target_names_is_list else [
            self.objective_names + '_class_weight']

        self.additional_inputs = kwargs['additional_input_data'][0].split() if len(kwargs['additional_input_data']) > 1 else kwargs['additional_input_data'][0]
        self.ligand_col_name = kwargs['ligand_col_name']

        # mut_col_name is included in case processed dataheader is different for the mutant column
        self.mut_col_name = kwargs['mut_col_name']

        # the directory containing the train/val/test split and the set names within that dir
        self.split_dir = kwargs['split_dir']
        self.train_name = kwargs['train_name']
        self.val_name = kwargs['val_name']
        self.test_name = kwargs['test_name']

        # load hyperparameters
        self.encoding = kwargs['encoding']
        self.num_pre_trained_layers_to_retrain = int(kwargs['num_pre_trained_layers_to_retrain'])
        self.pool_pretrained_representation = kwargs['pool_pretrained_representation']
        self.batch_size = int(kwargs['batch_size'])

        if 'ESM_' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                esm_tuple = pickle.load(f)
            self.alphabet = esm_tuple[1]
            self.batch_converter = self.alphabet.get_batch_converter()
        elif 'METL-G-' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                METL_tuple = pickle.load(f)
            self.alphabet = None
            self.batch_converter = METL_tuple[1]
        else:
            self.model_name = None
            self.alphabet = None
            self.batch_converter = None

        # load dataset info from datasets.yml
        ds_info = utils.load_ds_index()[kwargs['ds_name']]

        # load the pandas dataframe for the dataset and the dictionary containing split indices
        self.ds = utils.load_ds(ds_info['ds_fn'])
        self.split_idxs = splits.load_split(self.split_dir)

        # amino acid sequence length for this protein
        self.aa_seq_len = len(ds_info["wt_aa"])
        if self.encoding == 'one_hot_nuc_uni':
            self.aa_seq_len *= 3

        # Encode variants to see what the encoding length will be, (batch_size, aa_seq_len, enc_len)
        # this also makes a convenient example_input_array
        temp_variants = self.ds.iloc[0:self.batch_size].loc[:, self.mut_col_name]
        temp_encoded_variant = encode.encode(encoding=self.encoding, variants=temp_variants.tolist(),
                                             ds_name=self.ds_name, frame_shift=self.frame_shift,
                                             ncomp=self.ncomp, augmentation_factor=1,
                                             batch_converter=self.batch_converter)

        # sequence encoding lengths, used in model construction
        # amino acid sequence length and encoding lengths, used in model construction
        if self.encoding == 'aaindex':
            self.aa_encoding_len = self.ncomp
        else:
            self.aa_encoding_len = temp_encoded_variant.shape[-1]
        self.seq_encoding_len = self.aa_seq_len * self.aa_encoding_len

        # set up an example input array here, to make it easier to print full summaries in model defs
        if 'ESM_' in self.encoding:
            self.example_input_array = temp_encoded_variant
        else:
            self.example_input_array = torch.from_numpy(temp_encoded_variant)


    def get_encoded_data(self, set_name):
        """ encode an entire set """
        idxs = self.split_idxs[set_name]
        variants = self.ds.iloc[idxs].loc[:, self.mut_col_name]
        if self.ligand_col_name != None:
            ligs = self.ds.iloc[idxs].loc[:, self.ligand_col_name]
        if self.additional_inputs != None:
            additional_inputs = self.ds.iloc[idxs].loc[:, self.additional_inputs]

        if set_name != 'train': # should not do data augmention
            augmentation_factor = 1
        else:
            augmentation_factor = self.augmentation_factor

        enc_data = encode.encode(encoding=self.encoding, variants=variants.tolist(),
                                 ds_name=self.ds_name, frame_shift=self.frame_shift,
                                 ncomp=self.ncomp, augmentation_factor=augmentation_factor,
                                 batch_converter=self.batch_converter)

        if 'ESM_' not in self.encoding and 'METL-G-' not in self.encoding:
            empty_padded = np.empty(shape=enc_data.shape)[:, :1, :]
            if self.ligand_col_name != None:
                padded = empty_padded.copy()
                padded[:, 0, 0] = ligs
                enc_data = np.append(enc_data, padded, axis=1)
            if self.additional_inputs != None:
                padded = empty_padded.copy()
                padded[:, 0, 0] = additional_inputs
                enc_data = np.append(enc_data, padded, axis=1)

        return enc_data

    def get_targets(self, set_name):
        """ get targets for an entire set """
        idxs = self.split_idxs[set_name]
        scores = self.ds.iloc[idxs].loc[:, self.score_names].to_numpy().astype(np.float32)
        scores = scores.reshape(len(idxs), len(self.score_names))

        # start data dicitonary, it will only contain targets if classificaiton,
        # qual, and weighted are false
        data = {'scores': scores}

        # initialize empty array to use if conditionals are false
        empty_array = np.empty(np.shape(scores))
        empty_array[:] = np.NaN

        # handling conditionals for classification, quality, and weighting
        # empty array is set when condition is false, this is to allow adaptability
        if self.classification:
            class_name = self.ds.iloc[idxs].loc[:, self.class_name].to_numpy().astype(np.float32)
            class_name = class_name.reshape(len(idxs), len(self.class_name))
            data['class_name'] = class_name
        else:
            data['class_name'] = empty_array.copy()

        if self.quality and self.quality_available:
            qual = self.ds.iloc[idxs].loc[:, self.qual].to_numpy().astype(np.float32)
            qual = qual.reshape(len(idxs), len(self.qual))
            data['qual'] = qual
        else:
            # set all to poor quality, this allows for inclusion of these data in threshold regression
            data['qual'] = np.nan_to_num(empty_array.copy(), 0.0)

        # whether the dataset contains information about data quality to use for model weighting (e.g., inverse variance of replicates)
        if self.weighted:
            weight = self.ds.iloc[idxs].loc[:, self.weight].to_numpy().astype(np.float32)
            weight = weight.reshape(len(idxs), len(self.weight))
            data['weight'] = weight
        else:
            data['weight'] = empty_array.copy()

        # whether the dataset contains information about weights relative abundances of classes to mitigate class imbalance
        if self.weighted_classification:
            class_weight = self.ds.iloc[idxs].loc[:, self.class_weight].to_numpy().astype(np.float32)
            class_weight = class_weight.reshape(len(idxs), len(self.class_weight))
            data['class_weight'] = class_weight
        else:
            data['class_weight'] = empty_array.copy()

        if self.ligand_col_name != None:
            ligs = self.ds.iloc[idxs].loc[:, self.ligand_col_name].to_numpy().astype(np.float32)
            ligs = ligs.reshape(len(idxs), len(self.ligand_col_name))
            data['ligs'] = ligs
        else:
            data['ligs'] = None

        if self.additional_inputs != None:
            additional_inputs = self.ds.iloc[idxs].loc[:, self.additional_inputs].to_numpy().astype(np.float32)
            additional_inputs = additional_inputs.reshape(len(idxs), len(self.additional_inputs))
            data['additional_inputs'] = additional_inputs
        else:
            data['additional_inputs'] = None

        return data

    def get_ds(self, set_name):
        data = self.get_targets(set_name)
        enc_data = self.get_encoded_data(set_name)

        # define outputs from data
        scores = data['scores']
        weight = data['weight']

        qual = data['qual']
        class_name = data['class_name']
        class_weight = data['class_weight']

        if 'nuc' not in self.encoding:
        # augmentation factor should always be 1 in these instances because it is not possible to augment these encodings
            self.augmentation_factor = 1
        else:
            # reshape augmented inputs
            enc_data = enc_data.reshape((-1, self.aa_seq_len, self.aa_encoding_len))

            # multiply first dimension of data by augmentation factor
            new_shape = list(scores.shape)
            new_shape[0] *= self.augmentation_factor
            new_shape = tuple(new_shape)

            # repeat data and reshape with new dimensions
            scores = np.repeat(scores[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            weight = np.repeat(weight[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            qual = np.repeat(qual[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            class_name = np.repeat(class_name[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            class_weight = np.repeat(class_weight[np.newaxis], self.augmentation_factor, axis=0).reshape(
                new_shape)

        # ESM gives data as tensor, but other encoding methods don't
        if not torch.is_tensor(enc_data):
            enc_data = torch.from_numpy(enc_data)

        torch_ds = DMSDataset(inputs=enc_data,
                              scores=torch.from_numpy(scores),
                              weight=torch.from_numpy(weight),
                              qual=torch.from_numpy(qual),
                              class_name=torch.from_numpy(class_name),
                              class_weight=torch.from_numpy(class_weight))
        return torch_ds

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.get_ds(self.train_name)
            if self.val_name in self.split_idxs:
                self.val_ds = self.get_ds(self.val_name)

        if stage == 'test' or stage is None:
            self.test_ds = self.get_ds(self.test_name)

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def predict_dataloader(self):
        return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]


class DMSDatasetDataModule_hpopt(pl.LightningDataModule):

    def __init__(self, trial, **kwargs):
        super().__init__()

        kwargs['encoding'] = trial.suggest_categorical("encoding", kwargs['hp_encoding'])
        kwargs['batch_size'] = trial.suggest_categorical("batch_size", kwargs['hp_batch_size'])
        kwargs['weight_decay'] = trial.suggest_categorical("weight_decay", kwargs['hp_weight_decay'])
        kwargs['lr'] = trial.suggest_float("lr", kwargs['hp_lr'][0], kwargs['hp_lr'][1])

        kwargs['quality_treatment'] = trial.suggest_categorical("quality_treatment", kwargs['hp_quality_treatment'])
        kwargs['loss_method'] = trial.suggest_categorical("loss_method", kwargs['hp_loss_method'])
        kwargs['weighted'] = trial.suggest_categorical("weighted", kwargs['hp_weighted'])
        self.quality_treatment = kwargs['quality_treatment']

        # set up ESM alphabet if declared by encoding
        self.encoding = kwargs['encoding']
        self.num_pre_trained_layers_to_retrain = int(kwargs['num_pre_trained_layers_to_retrain'])
        self.pool_pretrained_representation = kwargs['pool_pretrained_representation']
        if 'ESM_' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                esm_tuple = pickle.load(f)
            self.alphabet = esm_tuple[1]
            self.batch_converter = self.alphabet.get_batch_converter()
        elif 'METL-G-' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                METL_tuple = pickle.load(f)
            self.alphabet = None
            self.batch_converter = METL_tuple[1]
        else:
            self.model_name = None
            self.alphabet = None
            self.batch_converter = None

        # process quality treatment to handle ['quality'] and ['classification','weighted_classification'] flags
        kwargs = handle_quality_treatment(kwargs)
        kwargs = handle_loss_method(kwargs)
        self.regression_thresholds = kwargs['regression_thresholds']


        if kwargs['classification']:
            kwargs['scaling_weight'] = trial.suggest_float("scaling_weight", kwargs['hp_scaling_weight'][0],
                                                           kwargs['hp_scaling_weight'][1])
        else:
            kwargs['scaling_weight'] = trial.suggest_float("scaling_weight", 0, 0.0)

        if 'aaindex' in kwargs['encoding']:
            kwargs['ncomp'] = trial.suggest_int("ncomp", kwargs['hp_ncomp'][0],
                                                           kwargs['hp_ncomp'][1])
        else:
            kwargs['ncomp'] = trial.suggest_int("hp_ncomp", 0, 0)

        if 'nuc' in kwargs['encoding']:
            kwargs['augmentation_factor'] = trial.suggest_int("augmentation_factor", kwargs['hp_augmentation_factor'][0], kwargs['hp_augmentation_factor'][1])
        else:
            kwargs['augmentation_factor'] = trial.suggest_int("augmentation_factor", 1, 1)

        # basic dataset and encoding info
        self.trial = trial
        self.ds_name = kwargs['ds_name']
        self.frame_shift = kwargs['frame_shift']
        self.augmentation_factor = kwargs['augmentation_factor']
        self.ncomp = kwargs['ncomp']
        self.log_dir = kwargs['log_dir']

        # define names of targets
        target_names_1 = kwargs['target_names']
        target_names_2 = kwargs['target_names'][0].split()
        if len(target_names_2) > len(target_names_1):
            target_names = target_names_2
        else:
            target_names = target_names_1
        target_names_is_list = isinstance(target_names, list) or isinstance(target_names, tuple)

        # set conditional flags describing how/which data is available for use
        self.quality = kwargs['quality']
        self.quality_available = kwargs['quality_available']
        self.weighted = kwargs['weighted']
        self.multiple_weighting_options = True if len(kwargs['hp_weighted']) == 2 else False
        self.classification = kwargs['classification']
        self.weighted_classification = kwargs['weighted_classification']
        self.additional_inputs = kwargs['additional_input_data'][0].split() if len(kwargs['additional_input_data']) > 1 else kwargs['additional_input_data'][0]
        self.ligand_col_name = kwargs['ligand_col_name']

        # load hyperparameters
        self.batch_size = kwargs['batch_size']
        self.weight_decay = kwargs['weight_decay']
        self.lr = kwargs['lr']
        self.scaling_weight = kwargs['scaling_weight']

        # bumber of model output tasks (classificaiton + regression) and names
        self.objective_names = target_names
        self.num_objectives = len(self.objective_names)

        # Initailize bias: this can be provided as values or labels to enable hyperparameter optimization
        self.output_bias = kwargs['output_bias'][0]
        self.output_bias = [self.output_bias] if self.output_bias == None else self.output_bias.split(' ')
        if len(self.output_bias) == 1:
            self.output_bias = [1]*self.num_objectives # no output bias
            self.bias_dict = None
        elif len(self.output_bias) == len(target_names):
            if type(self.output_bias[0]) == str:
                # do hyperparameter optimization for bias
                bias_labels = sorted(set(self.output_bias))
                bias_values = [trial.suggest_float(f"bias_{label}", 0, 1) for label in bias_labels] # sample bias value for each option
                bias_values = [i/sum(bias_values) for i in bias_values] # normalize so all values add to 1
                self.bias_dict = dict(zip(bias_labels, bias_values)) # assign each sampled bias to each of the label options
                self.output_bias = [self.bias_dict[label] for label in self.output_bias] # assign each bias to each of the true labels
            else:
                # User should have provided floats for each output (no hyperparamter optimization)
                self.output_bias = [float(i) for i in self.output_bias]
                self.output_bias = [i/sum(self.output_bias) for i in self.output_bias]
                self.bias_dict = None
        else:
            raise Exception('Invalid input for output_bias')

        # set score and output names
        output_col_names_1 = kwargs['output_col_names']
        output_col_names_2 = kwargs['output_col_names'][0].split()
        if len(output_col_names_2) > len(output_col_names_1):
            self.output_col_names = output_col_names_2
        else:
            self.output_col_names = output_col_names_1
        self.score_names = self.output_col_names[:self.num_objectives]

        # set classification names if necessary and adjust num_objetives
        if self.classification:
            self.class_name = self.output_col_names[self.num_objectives:]
            self.num_objectives *= 2
        else:
            self.output_col_names = self.output_col_names[:self.num_objectives]

        # define names of classes, quality, and weight columns (these will not be used if flags for these
        # parameters are not set, but will still be defined)
        self.qual = [s + '_qual' for s in self.objective_names] if target_names_is_list else [self.objective_names + '_qual']
        self.weight = [s + '_weight' for s in self.objective_names] if target_names_is_list else [self.objective_names + '_weight']
        self.class_weight = [s + '_class_weight' for s in self.objective_names] if target_names_is_list else [self.objective_names + '_class_weight']

        # mut_col_name is included in case processed dataheader is different for the mutant column
        self.mut_col_name = kwargs['mut_col_name']

        # the directory containing the train/val/test split and the set names within that dir
        self.split_dir = kwargs['split_dir']
        self.train_name = kwargs['train_name']
        self.val_name = kwargs['val_name']
        self.test_name = kwargs['test_name']

        # load dataset info from datasets.yml
        ds_info = utils.load_ds_index()[kwargs['ds_name']]

        # load the pandas dataframe for the dataset and the dictionary containing split indices
        self.ds = utils.load_ds(ds_info['ds_fn'])
        self.split_idxs = splits.load_split(self.split_dir)

        # amino acid sequence length for this protein
        self.aa_seq_len = len(ds_info["wt_aa"])
        if self.encoding == 'one_hot_nuc_uni':
            self.aa_seq_len *= 3


        temp_variants = self.ds.iloc[0:self.batch_size].loc[:, self.mut_col_name]

        # Encode variants to see what the encoding length will be, (batch_size, aa_seq_len, enc_len)
        # this also makes a convenient example_input_array
        temp_encoded_variant = encode.encode(encoding=self.encoding, variants=temp_variants.tolist(),
                                             ds_name=self.ds_name,frame_shift=self.frame_shift,
                                             ncomp=self.ncomp, augmentation_factor=1,
                                             batch_converter=self.batch_converter)

        # sequence encoding lengths, used in model construction
        # amino acid sequence length and encoding lengths, used in model construction
        if self.encoding == 'aaindex':
            self.aa_encoding_len = self.ncomp
        else:
            self.aa_encoding_len = temp_encoded_variant.shape[-1]

        self.seq_encoding_len = self.aa_seq_len * self.aa_encoding_len

        # set up an example input array here, to make it easier to print full summaries in model defs
        self.example_input_array = torch.from_numpy(temp_encoded_variant)

    def get_encoded_data(self, set_name):
        """ encode an entire set """
        idxs = self.split_idxs[set_name]
        variants = self.ds.iloc[idxs].loc[:, self.mut_col_name]

        enc_data = encode.encode(encoding=self.encoding, variants=variants.tolist(),
                                 ds_name=self.ds_name,frame_shift=self.frame_shift,
                                 ncomp=self.ncomp, augmentation_factor=self.augmentation_factor,
                                 batch_converter=self.batch_converter)

        if self.ligand_col_name != None:
            padded = enc_data.shape
        if self.additional_inputs != None:
            padded = enc_data.shape
            additional_inputs = None







        return enc_data

    def get_targets(self, set_name):
        """ get targets for an entire set """
        idxs = self.split_idxs[set_name]
        scores = self.ds.iloc[idxs].loc[:, self.score_names].to_numpy().astype(np.float32)
        scores = scores.reshape(len(idxs), len(self.score_names))

        # start data dicitonary, it will only contain targets if classificaiton,
        # qual, and weighted are false
        data = {'scores': scores}

        # initialize empty array to use if conditionals are false
        empty_array = np.empty(np.shape(scores))
        empty_array[:] = np.NaN

        # handling conditionals for classification, quality, and weighting
        # empty array is set when condition is false, this is to allow adaptability
        if self.classification:
            class_name = self.ds.iloc[idxs].loc[:, self.class_name].to_numpy().astype(np.float32)
            class_name = class_name.reshape(len(idxs), len(self.class_name))
            data['class_name'] = class_name
        else:
            data['class_name'] = empty_array.copy()

        if self.quality or self.regression_thresholds != None:
            qual = self.ds.iloc[idxs].loc[:, self.qual].to_numpy().astype(np.float32)
            qual = qual.reshape(len(idxs), len(self.qual))
            data['qual'] = qual
        else:
            data['qual'] = empty_array.copy()

        # whether the dataset contains information about data quality to use for model weighting (e.g., inverse variance of replicates)
        if self.weighted:
            weight = self.ds.iloc[idxs].loc[:, self.weight].to_numpy().astype(np.float32)
            weight = weight.reshape(len(idxs), len(self.weight))
            data['weight'] = weight
        else:
            data['weight'] = empty_array.copy()

        # whether the dataset contains information about weights relative abundances of classes to mitigate class imbalance
        if self.weighted_classification:
            class_weight = self.ds.iloc[idxs].loc[:, self.class_weight].to_numpy().astype(np.float32)
            class_weight = class_weight.reshape(len(idxs), len(self.class_weight))
            data['class_weight'] = class_weight
        else:
            data['class_weight'] = empty_array.copy()

        if self.ligand_col_name != None:
            ligs = self.ds.iloc[idxs].loc[:, self.ligand_col_name].to_numpy().astype(np.float32)
            ligs = ligs.reshape(len(idxs), len(self.ligand_col_name))
            data['ligs'] = ligs
        else:
            data['ligs'] = None

        if self.additional_inputs != None:
            additional_inputs = self.ds.iloc[idxs].loc[:, self.additional_inputs].to_numpy().astype(np.float32)
            additional_inputs = additional_inputs.reshape(len(idxs), len(self.additional_inputs))
            data['additional_inputs'] = additional_inputs
        else:
            data['additional_inputs'] = None

        return data

    def get_ds(self, set_name):
        data = self.get_targets(set_name)
        enc_data = self.get_encoded_data(set_name)

        # define outputs from data
        scores = data['scores']
        weight = data['weight']
        qual = data['qual']
        class_name = data['class_name']
        class_weight = data['class_weight']

        if self.encoding == 'aaindex' or self.encoding == 'one_hot_AA':
            # augmentation factor should always be 1 in these instances because it is not possible to augment these encodings
            self.augmentation_factor = 1
        else:
            # reshape augmented inputs
            enc_data = enc_data.reshape((-1,self.aa_seq_len,self.aa_encoding_len))

            # multiply first dimension of data by augmentation factor
            new_shape = list(scores.shape)
            new_shape[0] *= self.augmentation_factor
            new_shape = tuple(new_shape)

            # repeat data and reshape with new dimensions
            scores = np.repeat(scores[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            weight = np.repeat(weight[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            qual = np.repeat(qual[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            class_name = np.repeat(class_name[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)
            class_weight = np.repeat(class_weight[np.newaxis], self.augmentation_factor, axis=0).reshape(new_shape)

        # duplicate dimensions of augmented output data to match the input, then flatten
        # If augmentation_factor is 1, this will have no effect
        torch_ds = DMSDataset(inputs=torch.from_numpy(enc_data),
                              scores=torch.from_numpy(scores),
                              weight=torch.from_numpy(weight),
                              qual=torch.from_numpy(qual),
                              class_name=torch.from_numpy(class_name),
                              class_weight=torch.from_numpy(class_weight))
        return torch_ds

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.get_ds(self.train_name)
            if self.val_name in self.split_idxs:
                self.val_ds = self.get_ds(self.val_name)

        if stage == 'test' or stage is None:
            self.test_ds = self.get_ds(self.test_name)

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return data_utils.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                                     num_workers=torch.get_num_threads(), persistent_workers=True, pin_memory=torch.cuda.is_available())

    def predict_dataloader(self):
        return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]
