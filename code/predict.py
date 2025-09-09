import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from glob import glob
import sys

import encode
import utils
import analysis
import tasks
import parse_args
from datamodules import DMSDatasetDataModule

# set seaborn settings to produce high quality figures
sns.set(rc={"figure.dpi": 50, 'savefig.dpi': 50})
sns.set_style("ticks")


class PredictionHandler:
    def __init__(self, args, sequence_df):

        # define class variables
        self.sequence_df = sequence_df
        self.output_filename, self.format = args.output_filename.split('.') if len(args.output_filename.split('.')) == 2 else args.output_filename, None
        self.validation_headers = args.validation_headers
        self.monitor_percentiles = args.ensemble_monitor_percentiles
        if type(args.score_weights) == str:
            self.score_weights = pd.read_csv(args.score_weights, index_col=0)
        else:
            self.score_weights = None
        self.base_output_names = args.base_output_names
        self.simple_average = args.simple_average

    def fetch_checkpoint(self, log_dir):
        # load metrics and dataset info
        metrics = utils.load_ds(f'{log_dir}/metrics.csv')

        # load and format all train, val, test datasets
        train_metrics = metrics[~metrics['train/train_loss'].isna()].drop(self.validation_headers, axis=1)
        val_metrics = metrics[~metrics['val/val_loss'].isna()]
        val_metrics = val_metrics.loc[:, self.validation_headers + ['epoch']]
        loss_df = pd.merge(val_metrics, train_metrics, on='epoch')

        # locate checkpoint with minimum observed loss
        best_epoch = loss_df.loc[loss_df['val/val_loss'] == min(loss_df['val/val_loss'])].epoch.item()
        checkpoint = glob(f"{log_dir}/checkpoints/epoch={best_epoch}*")[0]

        return checkpoint

    def load_model(self, log_dir):
        # load model parameters to initialize model
        model_args = f"@{log_dir}/args.txt"
        for i in range(len(sys.argv)):
            if sys.argv[i][0] == '@':
                sys.argv[i] = model_args
            elif i+1 == len(sys.argv):
                # @args is not present inf sys.argv yet
                sys.argv.append(model_args)


        model_args = parse_args.main()
        model_args.log_dir, model_args.num_datasets, model_args.wandb_online, model_args.hpopt = None, 1, False, False
        self.true_col_names = model_args.output_col_names[0].split(' ')
        self.true_col_names = [i for i in self.true_col_names if 'class' not in i]

        self.prediction_col_names = ['p'+i for i in self.true_col_names]

        # set up the datamodule
        dm = DMSDatasetDataModule(**vars(model_args))

        # set up the task
        task = tasks.Task[model_args.task_name].cls(dm=dm,
                                              val_df=None,
                                              **vars(model_args))

        # load checkpoint from filepath
        checkpoint = self.fetch_checkpoint(log_dir)
        checkpoint = torch.load(checkpoint)

        # parse and correct checkpoint layer names by removing model prefix
        state_dict = checkpoint['state_dict'].copy()
        for k, v in checkpoint['state_dict'].items():
            state_dict['.'.join(k.split('.')[1:])] = checkpoint['state_dict'][k]
            del state_dict[k]

        # load checkpoint into model
        model = task.model
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()

        # encode sequence data: this assumes that all ensemble members will have the same encoding scheme
        try:
            if 'encoding' not in self.sequence_df.columns:
                self.sequence_df['encoding'] = np.squeeze(encode.encode(model_args.encoding, self.sequence_df['Sequence'].tolist(),
                                                               model_args.ds_name, model_args.frame_shift, model_args.ncomp,
                                                                        augmentation_factor=1, batch_converter=dm.batch_converter)).tolist()
        except:
            if 'encoding' not in self.sequence_df.columns:
                self.sequence_df['encoding'] = np.squeeze(
                    encode.encode(model_args.encoding, self.sequence_df['sequence'].tolist(),
                                  model_args.ds_name, model_args.frame_shift, model_args.ncomp,
                                  augmentation_factor=1, batch_converter=dm.batch_converter)).tolist()

        return model

    def load_ensemble(self, ensemble_log_dir):
        # load all components of ensemble and store as list
        models = []
        log_dirs = glob(f"{ensemble_log_dir}/*")
        for log_dir in log_dirs:
            log_dir = f"{log_dir}/version_0"
            # load model for log_dir
            # some model training may have failed. These will not be loadable
            try:
                model = self.load_model(log_dir)
                models.append(model)
                print('working')
            except: pass

        return models


    def predict(self, model, save_predictions=True):
        # make prediction for each sequence and append to df
        predictions = []
        sequence_df = self.sequence_df.copy()
        for i, r in sequence_df.iterrows():
            predictions.append(model.predict(r.encoding).detach().numpy().tolist()[0])
        sequence_df[self.prediction_col_names] = predictions

        # save predictions to excel
        if save_predictions:
            if self.simple_average:
                averaged_df = self.weight_predictions(sequence_df, simple_average=True)
                self.save_predictions(averaged_df, suffix='_avg')

            if np.all(self.score_weights) != None:
                averaged_df = self.weight_predictions(sequence_df, weighted_average=True)
                self.save_predictions(averaged_df, suffix='wght_avg')

            if not self.simple_average and not np.all(self.score_weights) != None:
                self.save_predictions(sequence_df)

        return sequence_df


    def weight_predictions(self, sequence_df, simple_average=False, weighted_average=False):
        if np.all([simple_average, weighted_average]) or not (simple_average or weighted_average):
            # one must be true
            raise Exception

        if simple_average:
            for col in self.base_output_names:
                output_score_cols = [i for i in self.prediction_col_names if col in i and 'score' in i]
                output_class_cols = [i for i in self.prediction_col_names if col in i and 'class' in i]
                if len(output_score_cols) > 0:
                    sequence_df[f'p{col}_score'] = sequence_df.loc[:, output_score_cols].mean(axis=1)
                if len(output_class_cols) > 0:
                    sequence_df[f'p{col}_class'] = sequence_df.loc[:, output_class_cols].mean(axis=1)

        elif weighted_average:
            for col in self.base_output_names:
                output_score_cols = [i for i in self.prediction_col_names if col in i and 'score' in i]
                output_class_cols = [i for i in self.prediction_col_names if col in i and 'class' in i]
                if len(output_score_cols) > 0:
                    weighted_data = [sequence_df.loc[:, i] * self.score_weights.loc[i[1:]].item() for i in output_score_cols]
                    sequence_df[f'p{col}_score'] = pd.concat(weighted_data, axis=1).sum(axis=1)
                if len(output_class_cols) > 0:
                    weighted_data = [sequence_df.loc[:,'_'.join(i.split('_')[:-1])+'_class']*self.score_weights.loc[i[1:]].item() for i in output_score_cols]
                    sequence_df[f'p{col}_class'] = pd.concat(weighted_data, axis=1).sum(axis=1)

        return sequence_df

    def predict_ensemble(self, models):
        predictions_dfs = []
        for model in models:
            predictions_df = self.predict(model, save_predictions=False)
            predictions_dfs.append(predictions_df)

        # store predictions determined by user provided percentiles
        for percentile in self.monitor_percentiles:
            for col in self.prediction_col_names:
                col_data = pd.concat([predictions_dfs[i].loc[:, col] for i in range(len(predictions_dfs))], axis=1)
                predictions_df.loc[:, col] = col_data.quantile(q=percentile/100, axis=1, interpolation='lower')

            # save perentile specific predictions to excel
            # if np.all(self.score_weights != None):
            #     predictions_df = self.weight_predictions(predictions_df)
            # self.save_predictions(predictions_df, suffix=f"_{percentile}_percentile")

            if self.simple_average:
                averaged_df = self.weight_predictions(predictions_df, simple_average=True)
                self.save_predictions(averaged_df, suffix=f"_{percentile}_percentile_avg")

            if np.all(self.score_weights) != None:
                averaged_df = self.weight_predictions(predictions_df, weighted_average=True)
                self.save_predictions(averaged_df, suffix=f"_{percentile}_percentile_wght_avg")

            if not self.simple_average and not np.all(self.score_weights) != None:
                self.save_predictions(predictions_df)

        return predictions_df


    def save_predictions(self, predictions_df, suffix=None):
        if suffix != None:
            if type(self.output_filename) == list:
                output_filename = '.'.join(self.output_filename) + suffix
            else:
                output_filename = self.output_filename+suffix
        else:
            output_filename = self.output_filename

        predictions_df.drop(columns=['encoding'], errors='ignore', inplace=True)

        utils.save_ds(predictions_df, output_filename, format='pkl')



def main(args):
    sequence_df = utils.load_ds(args.sequence_filename)
    # sequence_df = sequence_df.iloc[:100]
    # sequence_df.drop(columns='comb_rfaD_score', inplace=True, errors='ignore')

    predictor = PredictionHandler(args, sequence_df)

    if args.ensemble_log_dir != None:
        # make predictions for ensemble
        models = predictor.load_ensemble(args.ensemble_log_dir)
        predictions_df = predictor.predict_ensemble(models)
    else:
        # make predictions for single model
        model = predictor.load_model(args.log_dir)
        predictions_df = predictor.predict(model)
    print('done')






if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--log_dir",
                        help="base dir where training data is stored, should include version number",
                        type=str,
                        default='output/ensembles/cnn_dms_comb_ensemble_quality/X7p6ZYWE/version_0')

    parser.add_argument("--validation_headers",
                        help="column names reported in metrics file for validation runs",
                        type=str, nargs="+", default=['val/val_loss', 'val/MSE_loss', 'val/BCE_loss'])
    parser.add_argument("--base_output_names",
                        help="column names reported in metrics file for validation runs",
                        type=str, nargs="+", default=['10G', 'BW2', 'BL21', 'rfaD', 'rfaG'])
    parser.add_argument("--sequence_filename",
                        help="base dir where all different ensemble members checkpoints can be found. "
                             "Assumes version_0 for all",
                        type=str, default="motifs_averaged_deep_MPNN.pkl")
    parser.add_argument("--output_filename",
                        help="directory and filename for saving predictions",
                        type=str,
                        default='output/pmotif_dms_comb_unweighted')
    # type=str, default='output/ensembles/fcn_dms_comb_ensemble')

    parser.add_argument("--score_weights",
                        help="column names reported in metrics file for validation runs",
                        # type=str, default='data/T7RBD_all_separate/dms_comb_score_weights.csv')
                        type=str, default=None)

    parser.add_argument("--simple_average",
                        help="outputs an average of all predicitons (unweighted)",
                        action='store_true')

    parser.add_argument("--ensemble_log_dir",
                        help="base dir where all different ensemble members checkpoints can be found."
                             "Assumes version_0 for all",
                        # type=str, default='output/ensembles/lr_dms_comb_ensemble')
                        # type=str, default='CNN_ensemble')
                        type=str, default=None)

    parser.add_argument("--ensemble_monitor_percentiles",
                        help="column names reported in metrics file for validation runs",
                        type=int, nargs="+", default=[50])


    args = parser.parse_args()

    main(args)

    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # #
    # df = pd.read_pickle('output/first_attempt_new_CNN_50_percentile.pkl')
    #
    # sns.scatterplot(data=df[df.num_mutations.isin([3, 6, 9, 12, 15])], x='10G', y='p10G_score', linewidth=0, alpha=0.1)
    # plt.show()
    #
    #

    # # color by num_muts and plot score vs MPNN
    # sns.scatterplot(data=df[df.num_mutations.isin([3, 6, 9, 12, 15])], x='MPNN_score', y='p10G_score',
    #
    #                  linewidth=0, alpha=0.1)
    # plt.show()
    #





    #
    #
    #
    # # test whether model can predict first attempt data
    #
    # seq_df = pd.read_excel('/Volumes/sraman4/General/Nate/seq_results/20230120_all5_batch/outputs/counts/all_scores.xlsx')
    # seq_df = seq_df.drop(seq_df.loc[seq_df.num_mutations == 0].index)
    # seq_df = seq_df.loc[~seq_df.num_mutations.isna()]
    #
    # # seq_df = pd.read_excel('/Volumes/sraman4/General/Nate/seq_results/20230209_dms+comb/10G_scores.xlsx')
    # # seq_df = seq_df.drop(seq_df.loc[seq_df.num_mutations_x == 1].index)
    # # seq_df = seq_df.loc[~seq_df.mutations.isna()]
    #
    # new_df = seq_df.copy()
    #
    # seq_df['encoding'] = np.squeeze(encode.encode(args.encoding, seq_df['sequence'].tolist(),
    #                                               args.ds_name, 1,
    #                                               args.ncomp ,augmentation_factor=1)).tolist()
    #
    #
    # predictions = []
    # for i ,r in seq_df.iterrows():
    #     predictions.append(model.predict_after(r.encoding).detach().numpy().tolist())
    # #
    # # new_df[['p10G', 'pBW2', 'pBL21', 'prfaD', 'prfaG', 'p10G_class', 'pBW2_class', 'pBL21_class', 'prfaD_class', 'prfaG_class']] = predictions
    # # corr_df = new_df[['p10G', 'pBW2', 'pBL21', 'prfaD', 'prfaG', '10G2', '10G', 'BW2', 'BL21', 'rfaD', 'rfaG']].corr(method='spearman')
    # # corr_df2 = new_df[['p10G_class', 'pBW2_class', 'pBL21_class', 'prfaD_class', 'prfaG_class', '10G2','10G', 'BW2', 'BL21', 'rfaD', 'rfaG']].corr(method='spearman')
    # #
    #
    # new_df[['p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS', 'p10G_comb', 'pBW2_comb', 'pBL21_comb', 'prfaG_comb']] = predictions
    # new_df_short = new_df.loc[new_df['10G'] > -1]
    # corr_df = new_df_short[['p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS', 'p10G_comb', 'pBW2_comb', 'pBL21_comb', 'prfaG_comb', '10G']].corr(method='spearman')
    # # corr_df2 = new_df[['p10G_class', 'pBW2_class', 'pBL21_class', 'prfaD_class', 'prfaG_class', '10G2','10G', 'BW2', 'BL21', 'rfaD', 'rfaG']].corr(method='spearman')
    #
    # corr_df = new_df_short[['p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS', 'p10G_comb', 'pBW2_comb', 'pBL21_comb', 'prfaG_comb', '10G_score', 'BL21_score', 'BW2_score', 'rfaD_score', 'rfaG_score']].corr(method='spearman')
    #
    # sns.heatmap(corr_df)
    # plt.show()
    #
    # for strain in ['10G', 'BL21', 'BW2', 'rfaG', 'rfaD']:
    #     if strain != 'rfaD':
    #         sns.scatterplot(x=new_df[f'p{strain}_comb'], y=new_df[f'{strain}_score'])
    #         plt.show()
    #     sns.scatterplot(x=new_df[f'p{strain}_DMS'], y=new_df[f'{strain}_score'])
    #     plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    # seq_df = pd.read_excel("/Volumes/sraman4/General/Nate/T7_RBD_ML/manual_data_processing/outputs/motifs_averaged.xlsx")
    # new_df = seq_df.copy()
    #
    # seq_df['encoding'] = np.squeeze(encode.encode(args.encoding, seq_df['Sequence'].tolist(),
    #                                               args.ds_name, 1,
    #                                               args.ncomp ,augmentation_factor=1)).tolist()
    #
    # predictions = []
    # for i ,r in seq_df.iterrows():
    #     predictions.append(model.predict_after(r.encoding).detach().numpy().tolist()[0])
    # new_df[['p10G_DMS', 'pBL21_DMS', 'pBW2_DMS', 'prfaD_DMS', 'prfaG_DMS', 'p10G_comb', 'pBL21_comb', 'pBW2_comb', 'prfaG_comb']] = predictions
    #
    # new_df.rename(columns={'Mutations' :'variant',
    #                        'num_muts' :'num_mutations',
    #                        '10G_score' :'10G',
    #                        'BL21_score' :'BL21',
    #                        'BW2_score' :'BW2',
    #                        'rfaD_score' :'rfaD',
    #                        'rfaG_score' :'rfaG'}, inplace=True)
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21', 'BW2', 'rfaD', 'rfaG',
    #                        'p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS']]
    # prediction_cols = ['p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS']
    # sub_df.rename(columns={i :i.split('_')[0] for i in prediction_cols}, inplace=True)
    # sub_df.to_pickle('DMS_pMotif.pkl')
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21', 'BW2', 'rfaG',
    #                        'p10G_comb', 'pBW2_comb', 'pBL21_comb', 'prfaG_comb']]
    # prediction_cols = ['p10G_comb', 'pBW2_comb', 'pBL21_comb', 'prfaG_comb']
    # sub_df.rename(columns={i :i.split('_')[0] for i in prediction_cols}, inplace=True)
    # sub_df.to_pickle('Combinatorial_pMotif.pkl')
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21', 'BW2', 'rfaD', 'rfaG']]
    # prediction_cols = [i.split('_')[0] for i in ['p10G_DMS', 'pBW2_DMS', 'pBL21_DMS', 'prfaD_DMS', 'prfaG_DMS']]
    # sub_df.loc[: ,prediction_cols] = np.NaN
    # for col in prediction_cols:
    #     if col == 'prfaD':
    #         sub_df.loc[:, col] = new_df.loc[:, f'{col}_DMS']
    #         print('working')
    #         print(sub_df['prfaD'])
    #     else:
    #         sub_df.loc[:, col] = new_df.loc[: ,[f'{col}_DMS' ,f'{col}_comb']].mean(axis=1)
    #
    # sub_df.to_pickle('DMS+Combinatorial_pMotif.pkl')
    #
    #
    #
    #
    #
    #
    #
    # seq_df = pd.read_excel('/Volumes/sraman4/General/Nate/seq_results/20230120_all5_batch/outputs/counts/all_scores.xlsx')
    # seq_df = seq_df.drop(seq_df.loc[seq_df.num_mutations == 0].index)
    # seq_df = seq_df.loc[~seq_df.num_mutations.isna()]
    # seq_df = seq_df.loc[: ,['sequence', 'mutations', 'num_mutations', '10G2', 'p10G', 'BL21', 'pBL21']]
    # new_df = seq_df.copy()
    #
    # new_df.reset_index(drop=True, inplace=True)
    # new_df.rename(columns={'mutations' :'variant',
    #                        '10G2' :'10G',
    #                        'p10G' :'p10G_first_attempt'}, inplace=True)
    # new_df.to_pickle('first_attempt_10G.pkl')
    # new_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21']]
    #
    # seq_df['encoding'] = np.squeeze(encode.encode(args.encoding, seq_df['sequence'].tolist(),
    #                                               args.ds_name, 1,
    #                                               args.ncomp ,augmentation_factor=1)).tolist()
    #
    # predictions = []
    # for i ,r in seq_df.iterrows():
    #     predictions.append(model.predict_after(r.encoding).detach().numpy().tolist()[0])
    # new_df[['p10G_DMS', 'pBL21_DMS', 'pBW2_DMS', 'prfaD_DMS', 'prfaG_DMS', 'p10G_comb', 'pBL21_comb', 'pBW2_comb', 'prfaG_comb']] = predictions
    #
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations' ,'10G' ,'p10G_DMS', 'BL21', 'pBL21_DMS']]
    # prediction_cols = ['p10G_DMS', 'pBL21_DMS']
    # sub_df.rename(columns={i :i.split('_')[0] for i in prediction_cols}, inplace=True)
    # sub_df.to_pickle('DMS_pfirst.pkl')
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21',
    #                        'p10G_comb', 'pBL21_comb']]
    # prediction_cols = ['p10G_comb', 'pBL21_comb']
    # sub_df.rename(columns={i :i.split('_')[0] for i in prediction_cols}, inplace=True)
    # sub_df.to_pickle('Combinatorial_pfirst.pkl')
    #
    # sub_df = new_df.loc[: ,['variant', 'num_mutations', '10G', 'BL21']]
    # prediction_cols = [i.split('_')[0] for i in ['p10G_DMS', 'pBL21_DMS']]
    # sub_df.loc[: ,prediction_cols] = np.NaN
    # for col in prediction_cols:
    #     sub_df.loc[:, col] = new_df.loc[: ,[f'{col}_DMS' ,f'{col}_comb']].mean(axis=1)
    #
    # sub_df.to_pickle('DMS+Combinatorial_pfirst.pkl')
    #
    # test_df = utils.load_ds(ds_info['ds_fn'])
    # test_df = test_df.iloc[list(dm.split_idxs['test'])]
    #
    # test_df['encoding'] = np.squeeze(encode.encode(args.encoding, test_df['Sequence'].tolist(),
    #                                                args.ds_name, args.frame_shift,
    #                                                args.ncomp, augmentation_factor=1)).tolist()
    # predictions = []
    # for i, r in test_df.iterrows():
    #     predictions.append(model.predict(r.encoding).detach().numpy().tolist()[0])
    #
    # test_df.drop(columns='comb_rfaD_score', inplace=True)
    # test_df[["pDMS_p10G_score", "pDMS_BL21_score", "pDMS_BW2_score", "pDMS_rfaG_score", "pDMS_rfaD_score",
    #          "pcomb_10G_score", "pcomb_BL21_score", "pcomb_BW2_score", "pcomb_rfaG_score", "pmotif_10G_score",
    #          "pmotif_BL21_score", "pmotif_BW2_score", "pmotif_rfaG_score", "pmotif_rfaD_score",
    #          "pDMS_10G_class", "pDMS_BL21_class", "pDMS_BW2_class", "pDMS_rfaG_class", "pDMS_rfaD_class",
    #          "pcomb_10G_class", "pcomb_BL21_class", "pcomb_BW2_class", "pcomb_rfaG_class", "pmotif_10G_class",
    #          "pmotif_BL21_class", "pmotif_BW2_class", "pmotif_rfaG_class", "pmotif_rfaD_class"]] = predictions
    #
    # for strain in ['10G', 'BL21', 'BW2', 'rfaD', "rfaG"]:
    #     strain_cols = [i for i in test_df.columns if f"{strain}_score" in i]
    #     var_cols = [i for i in test_df.columns if f"{strain}_var" in i]
    #     true_strain_cols = [i for i in strain_cols if i[0] != 'p']
    #     pred_strain_cols = [i for i in strain_cols if i[0] == 'p']
    #     if strain == 'rfaD':
    #         test_df['all_true'] = test_df[true_strain_cols[0]].fillna(test_df[true_strain_cols[1]])
    #         test_df['var'] = test_df[var_cols[0]].fillna(test_df[var_cols[1]])
    #     else:
    #         test_df['all_true'] = test_df[true_strain_cols[0]].fillna(test_df[true_strain_cols[1]]).fillna(
    #             test_df[true_strain_cols[2]])
    #         test_df['var'] = test_df[var_cols[0]].fillna(test_df[var_cols[1]]).fillna(test_df[var_cols[2]])
    #     weights = 1 / test_df.loc[:, 'var']
    #
    #     # Generate correlation matrix df, but we will replace these values with the weighted correlations
    #     sub_df = test_df.loc[:, strain_cols + ["all_true"]]
    #     corr_df = sub_df.loc[:, strain_cols].corr(method='spearman')
    #     for i, r in corr_df.iterrows():
    #         for col in r.keys():
    #             if i != col:
    #                 corr_df.loc[i, col] = analysis.WeightedCorr(x=sub_df[i], y=sub_df[col], w=weights)(
    #                     method='spearman')
    #
    #     sns.heatmap(corr_df)
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # Generate correlation matrix df, but we will replace these values with the weighted correlations
    #     sub_df = test_df.loc[:, pred_strain_cols + ["all_true"]]
    #     corr_df = sub_df.corr(method='spearman')
    #     for i, r in corr_df.iterrows():
    #         for col in r.keys():
    #             if i != col:
    #                 corr_df.loc[i, col] = analysis.WeightedCorr(x=sub_df[i], y=sub_df[col], w=weights)(
    #                     method='spearman')
    #
    #     sns.heatmap(corr_df)
    #     plt.tight_layout()
    #     plt.show()



