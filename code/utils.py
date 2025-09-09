""" useful shared functions """
from itertools import groupby
import pandas as pd
import yaml
import uuid
from os.path import join, isfile, isdir
import os
import numpy as np
import shortuuid

import encode
import analysis

def load_ds_index():
    with open("data/datasets.yml", "r") as f:
        ds_index = yaml.safe_load(f)
    return ds_index


def load_ds(ds_fn, index_col=0):
    if ds_fn.split('.')[-1] == 'tsv':
        ds = pd.read_csv(ds_fn, index_col=index_col, sep='\t')
    elif ds_fn.split('.')[-1] == 'csv':
        ds = pd.read_csv(ds_fn)
    elif ds_fn.split('.')[-1] == 'xlsx' or ds_fn.split('.') == 'xls':
        ds = pd.read_excel(ds_fn,index_col=index_col)
    elif ds_fn.split('.')[-1] == 'pkl':
        ds = pd.read_pickle(ds_fn)
    return ds

def save_ds(ds, ds_fn, format='xlsx'):
    if format not in ds_fn:
        ds_fn += f".{format}"
    if format == 'tsv':
        ds.to_csv(ds_fn,sep='\t')
    elif format == 'csv':
        ds.to_csv(ds_fn)
    elif format == 'xlsx' or format == 'xls':
        ds.to_excel(ds_fn)
    elif format == 'pkl':
        ds.to_pickle(ds_fn)


def save_args(args_dict, hparam_dict, out_fn, ignore=None):
    """ save argparse arguments dictionary and model hyperparameters dict back to a file """
    args_dict.update(hparam_dict)
    with open(out_fn, "w") as f:
        for k, v in args_dict.items():
            # check if parameter has already been added
            # ignore these special arguments
            if (ignore is None) or (k not in ignore):
                # if a flag is set to false, dont include it in the argument file
                if (not isinstance(v, bool)) or (isinstance(v, bool) and v):
                    f.write(f"--{k}\n")
                    # if a flag is true, no need to specify the "true" value
                    if not isinstance(v, bool):
                        if isinstance(v, list):
                            for lv in v:
                                f.write(f"{lv}\n")
                        else:
                            f.write(f"{v}\n")


def load_lines(fn):
    """ loads each line from given file """
    lines = []
    with open(fn, "r") as f_handle:
        for line in f_handle:
            lines.append(line.strip())
    return lines


def all_equal(iterable):
    """ check if all list elements are equal (from itertools recipes)
        https://docs.python.org/3/library/itertools.html#itertools-recipes """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def log_dir_name(log_dir_base, my_uuid):
    """ log directory name, just the UUID """
    return join(log_dir_base, my_uuid)

def get_next_version(log_dir):
    """ see if there's an existing version_NUM directory and increment by one """
    existing_versions = []
    for d in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1

def gen_model_uuid():
    my_uuid = shortuuid.encode(uuid.uuid4())[:8]
    return my_uuid

def check_for_existing_log_dir(log_dir_base, my_uuid):
    existing_log_dir = False

    if not isdir(log_dir_base):
        return False, None

    # looking within the log_dir_base directory
    log_dirs = [join(log_dir_base, x) for x in os.listdir(log_dir_base) if isdir(join(log_dir_base, x))]

    # simply see if any of the log directory names contain the given UUID
    log_dir = None
    for ld in log_dirs:
        if my_uuid in ld.split("_"):
            print("Found existing log directory corresponding to given UUID: {}".format(ld))
            log_dir = ld
            existing_log_dir = True
            break

    return existing_log_dir, log_dir

def create_log_dir(log_dir_base, given_uuid, split_dir):
    if given_uuid == "condor":
        # this is a special value that instructs prep_condor_run to generate UUIDs
        # this should not actually make it to this script as an input
        raise ValueError("UUID 'condor' is a special value for prep_condor_run")

    # set up log directory & save the args file to it
    if given_uuid is None:
        # script was not given a custom UUID
        my_uuid = gen_model_uuid()
        print("Created model UUID: {}".format(my_uuid))
        log_dir = log_dir_name(log_dir_base, my_uuid)
        os.makedirs(log_dir, exist_ok=True)
        print("Created log directory: {}".format(log_dir))
    else:
        # script was given a custom UUID
        print("User gave model UUID: {}".format(given_uuid))
        my_uuid = given_uuid

        # check if a log directory already exists for this UUID
        existing_log_dir, log_dir = check_for_existing_log_dir(log_dir_base, my_uuid)
        if not existing_log_dir:
            # did not find an existing log directory, create our own using the supplied UUID
            print("Did not find existing log directory corresponding to given UUID: {}".format(my_uuid))
            log_dir = log_dir_name(log_dir_base, my_uuid)
            os.makedirs(log_dir, exist_ok=True)
            print("Created log directory: {}".format(log_dir))

    # at this point, my_uuid and log_dir are set to correct values, regardless of whether
    # uuid was passed in or created fresh in this script
    print("Final UUID: {}".format(my_uuid))
    print("Final log directory: {}".format(log_dir))

    return my_uuid, log_dir

def create_log_dir_version(log_dir):
    # figure out version number for this run (in case we are resuming a check-pointed run)
    version = get_next_version(log_dir)
    print("This is version: {}".format(version))

    # the log directory for this version
    log_dir_version = join(log_dir, "version_{}".format(version))
    os.makedirs(log_dir_version, exist_ok=True)

    print("Version-specific logs will be saved to: {}".format(log_dir_version))

    return version, log_dir_version


def get_checkpoint_path(log_dir):
    if isfile(join(log_dir, "checkpoints", "last.ckpt")):
        ckpt_path = join(log_dir, "checkpoints", "last.ckpt")
        print("Found checkpoint, resuming training from: {}".format(ckpt_path))
    else:
        ckpt_path = None
        print("No checkpoint found, training from scratch")
    return ckpt_path


def save_metrics_ptl(best_model_path, best_model_score, test_metrics, log_dir):
    """ save metrics to a txt file """
    with open(join(log_dir, "metrics.txt"), "w") as f:
        f.write("best_model_path,{}\n".format(best_model_path))
        f.write("best_model_score,{}\n".format(best_model_score))
        for k, v in test_metrics[0].items():
            f.write("{},{}\n".format(k, v))


def prepare_figures(args, log_dir_version, dm, model):
    # funciton to prepare figures according to model architecture and objectives
    if dm.classification:
        val_headers = ['val/val_loss', 'val/MSE_loss', 'val/BCE_loss']
    else:
        val_headers = ['val/val_loss']

    # load metrics and dataset info
    metrics_df = pd.read_csv(f'{log_dir_version}/metrics.csv')
    ds_info = load_ds_index()[dm.ds_name]

    # load and format all train, val, test datasets
    train_df = metrics_df[~metrics_df['train/train_loss'].isna()].drop(val_headers, axis=1)
    val_df = metrics_df[~metrics_df['val/val_loss'].isna()]
    val_df = val_df.loc[:, val_headers + ['epoch']]
    loss_df = pd.merge(val_df, train_df, on='epoch')

    # make figures for validation dataset to prevent bias
    test_df = load_ds(ds_info['ds_fn'])
    test_df = test_df.iloc[list(dm.split_idxs['test'])]
    seq_col_name = 'Sequence' if 'Sequence' in test_df.columns else 'sequence'

    # determine if df has weights columns
    weights_available = True if 'weight' in ' '.join(test_df.columns) else False

    # encode sequences
    encoded_seqs = np.squeeze(encode.encode(args.encoding, test_df[seq_col_name].tolist(),
                                        args.ds_name, args.frame_shift,
                                        args.ncomp,augmentation_factor=1, batch_converter=dm.batch_converter))
    if 'ESM' not in args.encoding:
        test_df[seq_col_name] = encoded_seqs.tolist()
    else:
        # additional processing for ESM
        test_df[seq_col_name] = [encoded_seqs[i] for i in range(encoded_seqs.shape[0])]

    # conduct analyses to compare model predicitons to true observations of test_df
    analysis.Analyze(model, test_df, dm.objective_names, dm.output_col_names, loss_df,
                    should_weight=weights_available, graph_quality='good', dm=dm, **vars(args))