import enum
import torch.nn as nn
import pytorch_lightning as pl
import torch
import numpy as np
import wandb
import pickle
from scipy import stats
import pandas as pd

import encode
import utils


class fully_configurable(pl.LightningModule, nn.Module):
    """ a fully configurable model for training or conducting optuna hyperparameter search"""
    def __init__(self,
                 task,
                 aa_encoding_len: int,
                 **kwargs):
        super().__init__()

        # Set encoding type
        self.encoding = task.encoding
        self.pool_pretrained_representation = task.pool_pretrained_representation
        if self.encoding == 'aaindex':
            self.aaindex = encode.EncodeAAindex(aa_encoding_len).aaindex
            self.embed = nn.Embedding.from_pretrained(self.aaindex, freeze=False)
        else:
            self.aaindex, self.embed = None, None

        # load hyperparameters from task module
        self.aa_seq_len = task.aa_seq_len
        self.batch_converter = task.batch_converter
        self.ds_name = task.ds_name
        self.num_tasks = task.num_tasks
        self.output_col_names = task.output_col_names
        self.output_bias = task.output_bias
        self.ndim = int(aa_encoding_len)  # dimensions of AA embedding
        self.batch_size = task.batch_size
        self.lr = task.lr
        self.scaling_weight = task.scaling_weight
        self.weight_decay = task.weight_decay

        # load additional parameters
        ds_info = utils.load_ds_index()[self.ds_name]
        self.pdb_path = ds_info['pdb_path']
        self.max_parameters = int(kwargs['max_parameters'])
        activation_functions = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'leaky_relu': nn.LeakyReLU(), 'elu': nn.ELU()}

        # define network
        if task.hpopt:
            self.trial = task.trial
            n_dense = self.trial.suggest_int("n_dense", kwargs['hp_n_dense'][0], kwargs['hp_n_dense'][1])
            if n_dense > 1:
                self.fc_activation_function = self.trial.suggest_categorical("fc_activation_function", kwargs["hp_fc_activation_function"])
                if self.fc_activation_function != 'linear':
                    self.dense_dropout = self.trial.suggest_float("dense_dropout", 0.0, 0.5)
                else: self.dense_dropout = None

            n_conv = self.trial.suggest_int("n_conv", kwargs['hp_n_conv'][0], kwargs['hp_n_conv'][1])
            if n_conv > 0:
                cnn_filter_factor = self.trial.suggest_categorical("cnn_filter_factor", kwargs['hp_cnn_filter_factor']) # should num_filters stay constant, double
                cnn_starting_filters = self.trial.suggest_categorical('cnn_starting_filters', kwargs['hp_cnn_starting_filters'])
                self.ks = self.trial.suggest_categorical("ks", kwargs['hp_ks'])
                self.cnn_dropout = self.trial.suggest_float("cnn_dropout", 0.0, 0.5)
                self.cnn_activation_function = self.trial.suggest_categorical("cnn_activation_function", kwargs["hp_cnn_activation_function"])
        else:
            n_dense = int(kwargs['n_dense'])
            if n_dense > 1:
                self.fc_activation_function = kwargs['fc_activation_function']
                self.dense_dropout = kwargs["dense_dropout"]

            n_conv = int(kwargs['n_conv'])
            if n_conv > 0:
                cnn_filter_factor = int(kwargs['cnn_filter_factor']) # should num_filters stay constant, double
                cnn_starting_filters = int(kwargs['cnn_starting_filters'])
                self.ks = int(kwargs['ks'])
                self.cnn_activation_function = kwargs['cnn_activation_function']
                self.cnn_dropout = float(kwargs["cnn_dropout"])

        # set some parameters to None for logging purposes
        if n_dense <= 1:
            self.fc_activation_function, self.dense_dropout = None, None
        if n_conv == 0:
            self.ks, self.cnn_dropout, self.cnn_activation_function = None, None, None
            cnn_filter_factor, cnn_starting_filters = None, None

        # set up pre trained models if applicable
        if 'ESM_' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                esm_tuple = pickle.load(f)
            self.repr_layer = int(self.encoding.split('_')[-1])
            self.pre_trained_model = esm_tuple[0]
            self.ndim = 1
            if self.pool_pretrained_representation:
                self.pre_dense_params = self.pre_trained_model.layers[-1].final_layer_norm.normalized_shape[0]
            else:
                self.pre_dense_params = self.pre_trained_model.layers[-1].final_layer_norm.normalized_shape[0]*(self.aa_seq_len+2)
        elif 'METL-G-' in self.encoding:
            with open(f'data/pre_trained_models/{self.encoding}.pkl', 'rb') as f:
                METLG_tuple = pickle.load(f)
            self.pre_trained_model = METLG_tuple[0]
            if self.pool_pretrained_representation:
                self.pre_trained_model.model = self.pre_trained_model.model[:3] # drop fcn from rosetta task
                self.pre_dense_params = self.pre_trained_model.model[-2].norm.normalized_shape[0]
            else:
                self.pre_trained_model.model = self.pre_trained_model.model[:2] # drop fcn and pooling layers from rosetta task
                self.pre_dense_params = self.pre_trained_model.model[-1].norm.normalized_shape[0]*self.aa_seq_len
            self.ndim = 1

        # initialize convolutional layers
        layers = []
        in_features = 1
        if n_conv != 0:
            out_features = in_features * cnn_starting_filters
            # create convolutional layer for each in n_conv
            for i in range(1, n_conv + 1):
                layers.append(nn.Conv1d(in_channels=int(in_features * self.ndim),
                                        out_channels=int(out_features * self.ndim),
                                        kernel_size=self.ks, padding='same'))
                layers.append(activation_functions[self.cnn_activation_function])
                layers.append(nn.Dropout(self.cnn_dropout))
                if i != n_conv:
                    in_features = int(out_features)
                    out_features = cnn_filter_factor * in_features
        else:
            out_features = in_features

        # if not using a pre-trained model
        if not ('METL-G-' in self.encoding or 'ESM' in self.encoding):
            self.pre_dense_params = int((self.aa_seq_len) * (out_features * self.ndim))

        # add flatten layer after either CNN or pre trained layers
        layers.append(nn.Flatten())

        # initialize fully connected layers
        in_features = self.pre_dense_params
        for i in range(1,n_dense+1):
            if i == n_dense:
                # last layer
                layers.append(nn.Linear(in_features, self.num_tasks))
            else:
                if task.hpopt:
                    out_features = round(self.trial.suggest_int("n_dense_units_l{}".format(i),
                                                                np.maximum(self.num_tasks,round(np.sqrt(in_features), 0)),
                                                                np.maximum(self.num_tasks,round(in_features/2, 0))))
                else:
                    out_features = int(kwargs["n_dense_units_l{}".format(i)])

                if out_features < self.num_tasks:
                    out_features = self.num_tasks

                layers.append(nn.Linear(in_features, out_features))
                if self.fc_activation_function != 'linear' and self.fc_activation_function != None:
                    layers.append(activation_functions[self.fc_activation_function])
                    layers.append(nn.Dropout(self.dense_dropout))

                self.hparams["n_dense_units_l{}".format(i)] = out_features
                in_features = out_features


        # define network and loss
        self.custom_network = nn.Sequential(*layers)
        self.MSE_loss = nn.MSELoss(reduction='none')
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.float()

        # add informative parameters
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.hparams['pre_dense_params'] = self.pre_dense_params
        self.hparams['total_params'] = self.total_params
        self.hparams['max_parameters'] = self.max_parameters
        self.hparams['pool_pretrained_representation'] = self.pool_pretrained_representation
        self.hparams['num_pre_trained_layers_to_retrain'] = task.num_pre_trained_layers_to_retrain


        # add model architecture params
        self.hparams['classification'] = task.classification
        self.hparams['weighted_classification'] = task.weighted_classification
        self.hparams['quality_treatment'] = task.quality_treatment
        self.hparams['threshold_regression'] = True if task.regression_thresholds != None else False

        self.hparams['weighted'] = task.weighted
        self.hparams['encoding'] = task.encoding
        self.hparams['ncomp'] = task.ncomp
        self.hparams['batch_size'] = self.batch_size
        self.hparams['augmentation_factor'] = task.augmentation_factor

        self.hparams['lr'] = self.lr
        self.hparams['scaling_weight'] = self.scaling_weight
        self.hparams['weight_decay'] = self.weight_decay

        self.hparams['n_dense'] = n_dense
        self.hparams['fc_activation_function'] = self.fc_activation_function
        self.hparams['dense_dropout'] = self.dense_dropout

        self.hparams['n_conv'] = n_conv
        self.hparams['cnn_activation_function'] = self.cnn_activation_function
        self.hparams['cnn_filter_factor'] = cnn_filter_factor
        self.hparams['cnn_starting_filters'] = cnn_starting_filters
        self.hparams['ks'] = self.ks
        self.hparams['cnn_dropout'] = self.cnn_dropout

        if task.bias_dict != None:
            for k,v in task.bias_dict.items():
                self.hparams[f'bias_{k}'] = v

        if kwargs['wandb_online']:
            for k,v in self.hparams.items():
                wandb.run.summary[k] = v

    def forward(self, x):

        # Make sure models don't get unreasonably big
        if self.total_params > self.max_parameters:
            raise ValueError("Model is too large")

        if self.encoding == 'aaindex':
            x = self.embed(x)
            x = x.view(-1, self.aa_seq_len, self.ndim)
        if 'ESM_' in self.encoding:
            x = self.pre_trained_model(x, repr_layers=[self.repr_layer], return_contacts=False)
            x = x["representations"][self.repr_layer]
            if self.pool_pretrained_representation:
                x = torch.mean(x[:, 1:(self.aa_seq_len + 3)], dim=1) # ESM adds two to the sequence length
        elif 'METL-G-' in self.encoding:
            x = self.pre_trained_model(x, pdb_fn=self.pdb_path)
        else: # all non-pretrained encodings
            x = x.permute(0, 2, 1)  # swap length and channel dims
        if not self.pool_pretrained_representation and ('ESM_' in self.encoding or 'METL-G-' in self.encoding):
            x = x.reshape(-1, x.shape[1]*x.shape[2])

        x = self.custom_network(x)
        x = x.view(-1, self.num_tasks)

        return x

    def predict_from_checkpoint(self, seq, frame_shift, ds_for_percentiles, merge_with_bias=False):
        # This section is meant for downsampling analysis, or other situations
        # for which model is loaded from models module instead of tasks module
        if type(seq) == list and type(seq[0]) == str:
            seq = encode.encode(encoding=self.encoding, variants=seq,
                                     ds_name=self.ds_name, frame_shift=frame_shift,
                                     ncomp=self.ndim, augmentation_factor=1,
                                aaindex=self.aaindex, batch_converter=self.batch_converter)
        elif type(seq) == list and type(seq[0]) == list: # already encoded, just need to convert to np array
            seq = np.array(seq)

        # make sure input is a tensor
        x = torch.tensor(seq) if not torch.is_tensor(seq) else seq

        if self.encoding == 'aaindex' or 'ESM_' in self.encoding or 'METL-G-' in self.encoding:
            x = x.view(1, -1)  # add batch dimension

        pred = self(x)
        pred = pred.detach().numpy()
        pred_df = pd.DataFrame(pred, columns=['p'+i for i in self.output_col_names])

        if np.all(ds_for_percentiles) != None:
            for col in pred_df.columns:
                pred_df.loc[:,col] = stats.percentileofscore(ds_for_percentiles[col].tolist(), pred_df.loc[:,col])

        if type(merge_with_bias) != bool:
            # use pre defined series for computing weighted averages
            for col in pred_df.columns:
                pred_df.loc[:,col] = pred_df.loc[:,col] * merge_with_bias.loc[col].item()
        elif merge_with_bias:
            # use arg file defined output bias for weighting
            bias_dict = {self.output_col_names[i]: self.output_bias[i] for i in range(pred.shape[1])}
            for col in pred_df.columns:
                pred_df.loc[:,col] = pred_df.loc[:,col] * bias_dict[col[1:]]

        if type(merge_with_bias) != bool or merge_with_bias:
            base_objectives = list(set(['_'.join(i.split('_')[1:]) for i in self.output_col_names]))
            objectives_dict = {i: [name for name in self.output_col_names if i in name] for i in
                                    base_objectives}
            pred_df = pd.DataFrame({k:pred_df.loc[:, ['p'+i for i in v]].sum(axis=1) for k,v in objectives_dict.items()})


        # pred_dict = {self.output_col_names[i]: pred[i] for i in range(len(pred))}
        #
        # if np.all(ds_for_percentiles) != None:
        #     for k, v in pred_dict.items():
        #         percentile = stats.percentileofscore(ds_for_percentiles[f"p{k}"].tolist() + [-10000, 10000], v)
        #         pred_dict[k] = percentile
        #
        # if type(merge_with_bias) != bool:
        #     # use pre defined series for computing weighted averages
        #     for k, v in pred_dict.items():
        #         pred_dict[k] = pred_dict[k]*merge_with_bias.loc[k].item()
        # elif merge_with_bias:
        #     # use arg file defined output bias for weighting
        #     bias_dict = {self.output_col_names[i]: self.output_bias[i] for i in range(len(pred))}
        #     for k, v in pred_dict.items():
        #         pred_dict[k] = pred_dict[k]*bias_dict[k]
        #
        # if type(merge_with_bias) != bool or merge_with_bias:
        #     base_objectives = list(set(['_'.join(i.split('_')[1:]) for i in self.output_col_names]))
        #     pred_objectives_dict = {i: [pred_dict[name] for name in self.output_col_names if i in name] for i in
        #                             base_objectives}
        #     pred_dict = {k: sum(pred_objectives_dict[k]) for k in pred_objectives_dict.keys()}

        return pred_df

    def predict(self, seq):
        x = torch.tensor(seq) if not torch.is_tensor(seq) else seq
        # handling depends on encoding method
        if self.encoding == 'aaindex' or 'ESM_' in self.encoding or 'METL-G-' in self.encoding:
            x = x.view(1, -1)  # add batch dimension
        else:
            x = x.view(1, len(seq), -1)  # add batch dimension
        pred = self(x)

        return pred.clone().detach()


class Model(enum.Enum):
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, cls):
        self.cls = cls

    # list out models and corresponding classes here
    fully_configurable = fully_configurable
