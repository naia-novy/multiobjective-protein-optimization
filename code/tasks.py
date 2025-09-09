import pandas as pd
import torch
import torch.distributed
import torch.nn as nn
import pytorch_lightning as pl
import enum

import models
import analysis
import encode


class DMSPredictionTask(pl.LightningModule):

    def __init__(self, dm, val_df, **kwargs):
        super().__init__()

        # define hyperparams
        if kwargs['hpopt']:
            self.hpopt = True
            self.trial = dm.trial
            self.multiple_weighting_options = dm.multiple_weighting_options
        else:
            self.hpopt = False

        # load params from datamodule
        self.log_dir = dm.log_dir
        self.batch_size = dm.batch_size
        self.lr = dm.lr
        self.scaling_weight = dm.scaling_weight
        self.weight_decay = dm.weight_decay
        self.encoding = dm.encoding
        self.num_pre_trained_layers_to_retrain = dm.num_pre_trained_layers_to_retrain
        self.pool_pretrained_representation = dm.pool_pretrained_representation
        self.ncomp = dm.ncomp
        self.quality = dm.quality
        self.quality_treatment = dm.quality_treatment
        self.quality_available = dm.quality_available
        self.weighted = dm.weighted
        self.regression_thresholds = dm.regression_thresholds
        self.classification = dm.classification
        self.weighted_classification = dm.weighted_classification
        self.frame_shift = dm.frame_shift
        self.augmentation_factor = dm.augmentation_factor
        self.aa_seq_len = dm.aa_seq_len
        self.batch_converter = dm.batch_converter
        self.output_bias = dm.output_bias
        self.bias_dict = dm.bias_dict
        self.ds_name = dm.ds_name
        self.objective_names = dm.objective_names
        self.output_col_names = dm.output_col_names

        # set dataset descriptor parameters
        self.num_tasks = len(self.objective_names)
        if self.classification:
            self.num_tasks *= 2

        # define network
        self.model = models.Model[kwargs['model_name']].cls(task=self,
                                                            aa_seq_len=dm.aa_seq_len,
                                                            aa_encoding_len=dm.aa_encoding_len,
                                                            **kwargs)

        # determine max observed weights for each output, these will be used as the weights for
        if self.regression_thresholds != None and self.weighted:
            weight_cols = ['_'.join(i.split('_')) + '_weight' for i in self.objective_names]
            self.output_means = dm.ds.loc[:,weight_cols].mean()

        # define loss funcitons
        self.MSE_loss = nn.MSELoss(reduction='none')
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')

        # validation df is used for calculating spearman correlation at each epoch
        self.val_df = val_df

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _calc_loss(self, output_score, output_class, score, class_name, weight, qual, class_weight):

        # Take MSE by dividing by number of examples considered for MSE
        MSE_loss = self.MSE_loss(output_score, score)
        MSE_loss = torch.mul(MSE_loss, torch.tensor(self.output_bias))

        # torch.nanmean(MSE_loss) resulted in propagation of nans and all model weights becoming nan after training
        # to get around this, I instead masked nans with zero and calculated a 'non-zero' mean
        MSE_mask = score.detach().clone()
        MSE_mask[MSE_mask!=10000.0] = 1.0
        MSE_mask[MSE_mask==10000.0] = 0.0

        # threshold regression to set loss as zero if true and predicted scores are below threshold
        if self.regression_thresholds != None:
            # Only train towards data that is observed
            for i_row, x in enumerate(score):
                for i_col, y in enumerate(x):
                        # Data I normally would have ignored for regression and only done classification on in class reg
                        if qual[i_row, i_col] == 0.0:
                            if score[i_row, i_col] <= self.regression_thresholds[i_col]:
                                if output_score[i_row, i_col] <= self.regression_thresholds[i_col]:
                                    # set loss for this point to zero because output and true are both below threshold
                                    MSE_mask[i_row, i_col] = 0.0

                                if self.weighted:
                                    # set weight to max weight observed for output col (we are really confident this is dead)
                                    weight[i_row, i_col] = self.output_means[i_col]

        # multiply MSE_loss by qual and weight if relevant, adjust MSE_mask as well
        if self.weighted:
            MSE_mask[weight == 10000.0] = 0.0
            MSE_loss = torch.mul(MSE_loss, weight)
        if self.quality: # Quality needs to come after mask counting so the number of observations is not miscounted
            MSE_mask[qual == 10000.0] = 0.0
            MSE_loss = torch.mul(MSE_loss, qual)

        # take mean of MSE_loss according to # observed observations
        regression_observations = torch.sum(MSE_mask)
        MSE_loss = torch.mul(MSE_loss, MSE_mask)
        MSE_loss = torch.div(torch.sum(torch.nan_to_num(MSE_loss, nan=0.0, posinf=0.0, neginf=0.0)), regression_observations)
        if regression_observations == 0.0:
            # this might occur with small batch sizes, if there were no observations the loss is zero
            MSE_loss = 0.0

        # compute classification loss if classification data is available
        if self.classification:
            # Calculate BCE_loss for qualitative score
            BCE_loss = self.BCE_loss(output_class, class_name)
            BCE_loss = torch.mul(BCE_loss, torch.tensor(self.output_bias))

            # torch.nanmean(BCE_loss) resulted in propagation of nans and all model weights becoming nan after training
            # to get around this, I instead masked nans with zero and calculated a 'non-zero' mean
            BCE_mask = class_name.detach().clone()
            BCE_mask[BCE_mask != 10000.0] = 1.0
            BCE_mask[BCE_mask == 10000.0] = 0.0

            # weight classiciation by observation frequency to mitigate class imbalance
            if self.weighted_classification:
                BCE_mask[class_weight == 10000.0] = 0.0
                BCE_loss = torch.mul(BCE_loss, class_weight)

            # take mean of BCE_loss using # observations determined by BCE_mask
            class_observations = torch.sum(BCE_mask)
            BCE_loss = torch.mul(BCE_loss, BCE_mask) #BCE_loss[BCE_mask == 1.0]
            BCE_loss = torch.div(torch.sum(torch.nan_to_num(BCE_loss, nan=0.0, posinf=0.0, neginf=0.0)), class_observations)
            if class_observations == 0.0:
                # this might occur with small batch sizes, if there were no observations the loss is zero
                BCE_loss = 0.0
        else:
            BCE_loss = 0.0

        return MSE_loss, BCE_loss

    def training_step(self, batch, batch_idx):
        output_score, score, MSE_loss, BCE_loss = self._shared_step(batch, batch_idx, compute_loss=True)

        # weight MSE and BCE based on set scaling fraction
        loss = BCE_loss*self.scaling_weight + MSE_loss*(1-self.scaling_weight)

        # log loss metrics
        self.log("train/train_loss", loss, prog_bar=True, logger=True,  on_step=False, on_epoch=True)
        self.log("train/BCE_loss", BCE_loss*self.scaling_weight, on_step=False, on_epoch=True)
        self.log("train/MSE_loss", MSE_loss*(1-self.scaling_weight), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output_score, score, MSE_loss, BCE_loss = self._shared_step(batch, batch_idx, compute_loss=True)

        # weight MSE and BCE based on set scaling fraction
        loss = BCE_loss*self.scaling_weight + MSE_loss*(1-self.scaling_weight)

        # log loss metrics
        self.log("val/val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/BCE_loss", BCE_loss*self.scaling_weight, on_step=False, on_epoch=True)
        self.log("val/MSE_loss", MSE_loss*(1-self.scaling_weight), on_step=False, on_epoch=True)

        # compute correlation and log to optuna (if doing hyperparameter optimization)
        if batch_idx == 0:
            # log spearman's r on val dataset once per epoch. Prune trials when the number of parameters is too large
            self.stat_name, stats = analysis.calculate_hp_stats(self, self.val_df, self.ds_name, self.objective_names, self.output_col_names,
                            self.log_dir, self.classification, self.quality_available)

            self.R = stats.loc[stats['Objective'] =='Average'].Value.values[0]
            if self.hpopt:
                if self.classification or self.multiple_weighting_options:
                    # cannot use loss for hpopt if weighing is going to be added and removed as scale of loss will change
                    # cannot use loss for hpopt for classiciation in case the scaling weight between regression and
                    # classification is being optimized
                    self.trial.report(self.R,self.current_epoch)
                else:
                    self.trial.report(loss,self.current_epoch)


        self.log(self.stat_name,self.R,logger=True,on_epoch=True,on_step=False)

        return loss

    def configure_optimizers(self):
        if self.num_pre_trained_layers_to_retrain == 0:
            num_pre_trained_layers_to_retrain = None
        else:
            num_pre_trained_layers_to_retrain = -self.num_pre_trained_layers_to_retrain

        if 'ESM_' in self.encoding:
            for esm_layer in self.model.pre_trained_model.layers[:num_pre_trained_layers_to_retrain]:
                for param in esm_layer.parameters():
                    param.requires_grad = False
        elif 'METL-G-' in self.encoding:
            for metl_layer in self.model.pre_trained_model.model.tr_encoder.layers[:num_pre_trained_layers_to_retrain]:
                for param in metl_layer.parameters():
                    param.requires_grad = False

        if 'ESM_' in self.encoding or 'METL-G-' in self.encoding:
            self.optimizer = torch.optim.AdamW([{'params':self.model.pre_trained_model.parameters(), 'lr':self.lr/10.0},
                                            {'params':self.model.custom_network.parameters(), 'lr':self.lr}],
                                            lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                            lr=self.lr, weight_decay=self.weight_decay)

        return self.optimizer

    def _shared_step(self, batch, batch_idx, compute_loss=True):
        # define metrics that might be used for determining loss
        # nans set to 10000.0 as a mask. 10000.0 is not observed in any of the datasets and is later replaced by 0.0 in mask
        # 0.0 can't be used here becasuse some data have meaningful 0.0 values. Pytorch weights all go to nan if nans are retained,
        # even when replacing them with zeros downstream
        sequence = batch["inputs"]
        score = torch.nan_to_num(batch["scores"], nan=10000.0, neginf=10000.0, posinf=10000.0)
        weight = torch.nan_to_num(batch["weight"], nan=10000.0, neginf=10000.0, posinf=10000.0)
        qual = torch.nan_to_num(batch["qual"], nan=10000.0, neginf=10000.0, posinf=10000.0)
        class_name = torch.nan_to_num(batch['class_name'], nan=10000.0, neginf=10000.0, posinf=10000.0)
        class_weight = torch.nan_to_num(batch['class_weight'], nan=10000.0, neginf=10000.0, posinf=10000.0)

        # determine predicted scores
        output = self(sequence)

        # split prediciton data
        if self.classification:
            split_idx = int(output.shape[1] / 2)
        else:
            split_idx = output.shape[1]

        output_score = output[:, :split_idx]
        output_class = output[:, split_idx:] # this will be nothing if not classification

        if compute_loss:
            MSE_loss, BCE_loss = self._calc_loss(output_score, output_class, score, class_name, weight, qual, class_weight)
            return output_score, score, MSE_loss, BCE_loss
        else:
            return output_score, score

    def predict(self, seq):

        if type(seq) == str:
            seq = encode.encode(encoding=self.encoding, variants=[seq],
                                         ds_name=self.ds_name, frame_shift=self.frame_shift,
                                         ncomp=self.ncomp, augmentation_factor=1, batch_converter=self.batch_converter)

        x = torch.tensor(seq) if not torch.is_tensor(seq) else seq
        # handling depends on encoding method
        if self.encoding == 'aaindex':
            x = x.view(1, -1)  # add batch dimension
        elif self.encoding == 'one_hot_nuc_uni' or self.encoding == 'one_hot_nuc_tri':
            x = x.squeeze(axis=0)

        pred = self(x)
        return pred.clone().detach()


class Task(enum.Enum):
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, cls):
        self.cls = cls

    # list out tasks and corresponding classes here
    DMSPredictionTask = DMSPredictionTask
