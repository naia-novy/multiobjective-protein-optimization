from argparse import ArgumentParser
import argparse

import models
import tasks

def main():
    # load all arguments (most come from @file)
    parser = ArgumentParser(add_help=True)

    # HTCondor args
    parser.add_argument("--cluster",
                        help="cluster (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--process",
                        help="process (when running on HTCondor)",
                        type=str,
                        default="local")
    parser.add_argument("--github_tag",
                        help="github tag for current run",
                        type=str,
                        default="no_github_tag")


    # logging params
    parser.add_argument("--log_dir_base",
                        help="log directory base",
                        type=str,
                        default="output/training_logs")
    parser.add_argument("--uuid",
                        help="model uuid to resume from or custom uuid to use from scratch",
                        type=str,
                        default=None)
    parser.add_argument("--wandb_online",
                        action="store_true",
                        default=False)
    parser.add_argument("--wandb_project",
                        type=str,
                        default="default")
    parser.add_argument("--wandb_entity",
                        type=str,
                        default='')


    # dataset arguments and dataspecific descriptors
    parser.add_argument("--ds_name",
                        help="name of the dms dataset defined in constants.py",
                        type=str, default="T7RBD_merged_datasets")
    parser.add_argument("--split_dir",
                        help="the directory containing the train/tune/test split",
                        type=str, default="data/T7RBD_merged_datasets/splits/standard_tr0.8_tu0.1_te0.1_r5")
    parser.add_argument("--train_name",
                        help="name of the train set in the split dir",
                        type=str, default="train")
    parser.add_argument("--val_name",
                        help="name of the validation set in the split dir",
                        type=str, default="val")
    parser.add_argument("--test_name",
                        help="name of the test set in the split dir",
                        type=str, default="test")
    parser.add_argument("--mut_col_name",
                        help="name of column from which to pull mutants",
                        type=str, default='Sequence', choices=['Sequence','sequence','Mutations'])


    parser.add_argument("--target_names",
                        help="names of objectives (ex, strains)",
                        type=str, nargs="+", default=["score"])
    parser.add_argument("--output_col_names",
                        help="names of columns to use as targets",
                        type=str, nargs="+", default=["score_wt_norm"])
    parser.add_argument("--output_bias",
                        help="label or weight to indicate each distinct output that you want to bias. Should be same length as "
                             "target_names and will do nothing if there is only one output.",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--frame_shift",
                        help="subtract this amount from the amino acid index to yield zero positional indexing",
                        type=str, default=0)
    parser.add_argument("--additional_input_data",
                        help="column names for numerical inputs to append to sequence embedding",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--ligand_col_name",
                        help="column names for numerical inputs to append to sequence embedding",
                        type=str, default=None)



    # add model specifiers
    parser.add_argument("--model_name",
                        choices=[m.name for m in list(models.Model)],
                        help="name of model class to use for training",
                        type=str, default="fully_configurable")
    parser.add_argument("--task_name",
                        choices=[t.name for t in list(tasks.Task)],
                        help="name of the task class to use for training",
                        type=str, default="DMSPredictionTask")

    # add model architechure/output formatting arguments
    parser.add_argument("--weighted",
                        help="whether to use inverse variance/SE weighting",
                        action="store_true")
    parser.add_argument("--loss_method",
                        help="method for loss training",
                        type=str, default="regression_only", choices=['regression_only','add_classification','add_weighted_classification'])
    parser.add_argument("--quality_available",
                        help="whether loaded dataset has labels for good (1) and poor (0) quality data",
                        action="store_true")
    parser.add_argument("--quality_treatment",
                        help="how to treat low quality data",
                        type=str, default="include_all", choices=['include_all','threshold_regression','quality_filter'])
    parser.add_argument("--regression_thresholds",
                        help="score thresholds to use for threshold regressor",
                        type=str, nargs="+", default=[None])

    # define hyperparameters
    parser.add_argument("--batch_size",
                        help="batch size for the data loader and optimizer",
                        type=str, default=64)
    parser.add_argument("--encoding",
                        help="which data encoding to use",
                        type=str, default="one_hot_AA",
                        choices=['one_hot_AA','aaindex','one_hot_aaindex','one_hot_nuc_uni','three_hot_nuc_uni','one_hot_nuc_tri',
                                 'METL-G-20M-1D','METL-G-20M-3D','METL-G-50M-1D','METL-G-50M-3D',
                                 'ESM_8M_6', 'ESM_35M_12', 'ESM_150M_30'])
    parser.add_argument("--num_pre_trained_layers_to_retrain",
                        help="how many layers of a pretrained model to retrain. Will start with terminal layers",
                        type=str, default=0)
    parser.add_argument("--pool_pretrained_representation",
                        help="whether to use global average pooling on last layer of pretrained model",
                        action="store_true")
    parser.add_argument("--max_epochs",
                        help="max epoch limit for training",
                        type=str, default=100)
    parser.add_argument("--ncomp",
                        help="number of components to use if encoding is aaindex or combination of aaindex",
                        type=str, default=6)
    parser.add_argument('--lr',
                        type=str,
                        default=0.0001)
    parser.add_argument("--scaling_weight",
                        help="weighting factor for using two classification and regression loss functions",
                        type=str, default=0.5)
    parser.add_argument("--augmentation_factor",
                        help="coefficient of augmentation to use for nucleotide embeddings",
                        type=str, default=1)
    parser.add_argument("--max_parameters",
                        help="if total parameters is above this threshold, training will terminate early (useful for hpopt)",
                        type=str, default=2000000)

    # output arguments
    parser.add_argument("--make_figs",
                        help="whether output figures at end of training",
                        action="store_true")


    # define hyperparameters for optuna hyperameter optimization
    parser.add_argument("--hp_batch_size",
                        help="options for batch size for the data loader and optimizer",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_encoding",
                        help="list of options for which data encoding to use",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_ncomp",
                        help="range of number of components to use if encoding is aaindex or combination of aaindex",
                        type=str, nargs="+", default=[None])
    parser.add_argument('--hp_lr',
                        help="range of learning rates to use for hp opt",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_scaling_weight",
                        help="range of weighting factor for using two classification and regression loss functions",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_augmentation_factor",
                        help="options for coefficient of augmentation to use for nucleotide embeddings",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_weight_decay",
                        help="options for weight decay to use for hp opt",
                        type=str, nargs="+", default=[None])

    parser.add_argument("--hp_n_dense",
                        help="minimum and maximum allowed dense/linear layers",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_n_conv",
                        help="minimum and maximum allowed convolutional layers",
                        type=str, nargs="+", default=[0,0])
    parser.add_argument("--hp_cnn_starting_filters",
                        help="first factor of expansion for CNN",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_cnn_filter_factor",
                        help="factor to increase filters by after each layer",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_ks",
                        help="options for CNN kernel sizes",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_fc_activation_function",
                        help="options for dense/linear activation function",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_cnn_activation_function",
                        help="options for CNN activation function",
                        type=str, nargs="+", default=[None])

    parser.add_argument("--hp_quality_treatment",
                        help="options for how to treat low quality data",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_loss_method",
                        help="method for loss training",
                        type=str, nargs="+", default=[None])
    parser.add_argument("--hp_weighted",
                        help="options for whether to do weighted regression",
                        type=str, nargs="+", default=[None])


    # add optional arguments
    # optional_parser = ArgumentParser()

    parser.add_argument("--fc_activation_function",
                        help="",
                        type=str, default='linear')#, choices=['linear','relu','gelu','leaky_relu','elu'])
    parser.add_argument("--n_dense",
                        help="",
                        type=str, default=0)
    parser.add_argument("--dense_dropout",
                        help="",
                        type=str, default=0.0)

    parser.add_argument("--cnn_activation_function",
                        help="",
                        type=str, default='relu')#, choices=['relu','gelu','leaky_relu','elu'])
    parser.add_argument("--cnn_filter_factor",
                        help="",
                        type=str, default=2)
    parser.add_argument("--cnn_starting_filters",
                        help="",
                        type=str, default=2)
    parser.add_argument("--n_conv",
                        help="",
                        type=str, default=0)
    parser.add_argument("--ks",
                        help="",
                        type=str, default=7)
    parser.add_argument("--cnn_dropout",
                        help="",
                        type=str, default=0.0)

    parser.add_argument("--weight_decay",
                        help="",
                        type=str, default=0.0)


    def retype(v):
        try:
            # retype ints, floats, bools
            if type(v) == list:
                vals = [eval(i) for i in v]
            else:
                vals = eval(v)

        except:
            try:
                # retype strings
                if type(v) == list:
                    vals = [eval(i) for i in v]
                else:
                    vals = eval("v")
            except:
                # value is None or already acceptable format
                vals = v

        return vals



    def retype_args(args):
        args = vars(args)
        for k,v in args.items():
            if 'hp_' in k:
                try:
                    vals = retype(v[0].split(' '))
                    args[k] = vals
                except : pass # use default as provided
                hpopt = True
            elif v == 'None':
                args[k] = None
            else:
                vals = retype(v)
                args[k] = vals


        # set hpoptimization flag to true if any hp flags were provided
        if hpopt:
            args['hpopt'] = True

        return argparse.Namespace(**args)

    # parse all args
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)
    args, unknown = parser.parse_known_args()

    # add dense unit information if it is present in unknown arguments
    for idx in range(0,len(unknown)):
        if '--n_dense_units_' in unknown[idx]:
            parser.add_argument(unknown[idx], type=int)

    # parse all args
    parser = ArgumentParser(parents=[parser], fromfile_prefix_chars='@', add_help=False)
    args, unknown = parser.parse_known_args()



    # reformat hp_opt args to have correct types
    args = retype_args(args)


    return args