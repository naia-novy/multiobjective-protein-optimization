""" split datasets and save splits to disk for reproducibility """
import argparse
from os.path import join, isfile, isdir, basename
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils


def save_split(split: Dict[str, List[int]], d: str):
    """ save a split to a directory """
    os.makedirs(d, exist_ok=True)
    for k, v in split.items():
        out_fn = join(d, "{}.txt".format(k))
        with open(out_fn, "w") as f_handle:
            for line in v:
                f_handle.write("{}\n".format(line))


def load_split(split_dir: str):
    """ load a split directory """
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if not f.startswith(".")]
    split = {}
    for f in fns:
        split_name = basename(f)[:-4]
        # pandas is a lot faster at reading csvs... only matters for very large splits
        split_data = pd.read_csv(f, header=None)[0].to_numpy()
        split[split_name] = split_data
    return split


def train_val_test(ds_len: int,
                   train_size: float,
                   val_size: float,
                   test_size: float,
                   seed: int = 0,
                   out_dir: str = None,
                   prefix: str = 'standard',
                   overwrite: bool = False):
    """ split data into train, val, and test sets """

    if round((train_size + val_size + test_size),3) != 1.000:
        # determine third split size from the other two if possible
        if train_size == 0.0:
            train_size = round(1.0 - val_size - test_size, 2)
        elif val_size == 0.0:
            val_size = round(1.0 - train_size - test_size, 2)
        elif test_size == 0.0:
            test_size = round(1.0 - val_size - train_size, 2)
        else:
            raise ValueError("train_size, val_size, and test_size must add up to 1. current values are "
                             "tr={}, tu={}, and te={}".format(train_size, val_size, test_size))

    # Set the random seed
    np.random.seed(seed)

    # Keep track of all the splits we make
    split = {}

    # Set up the indices that will get split
    idxs = np.arange(0, ds_len)

    # Set up the test dataset first so that all ensemble members will have the same test dataset
    if test_size > 0:
        if test_size == 1:
            split["test"] = idxs
        else:
            idxs, test_idxs = train_test_split(idxs, test_size=test_size, random_state=0)
            split["test"] = test_idxs

    if val_size > 0:
        adjusted_val_size = np.around(val_size / (1 - test_size), 5)
        if adjusted_val_size == 1:
            split["val"] = idxs
        else:
            idxs, val_idxs = train_test_split(idxs, test_size=val_size, random_state=seed)
            split["val"] = val_idxs

    if train_size > 0:
        adjusted_train_size = np.around(train_size / (1 - val_size - test_size), 5)
        if adjusted_train_size == 1:
            split["train"] = idxs
        else:
            idxs, train_idxs = train_test_split(idxs, test_size=adjusted_train_size, random_state=seed)
            split["train"] = train_idxs

    out_split_dir = None
    if out_dir is not None:
        out_dir_split = join(out_dir, "{}_tr{}_tu{}_te{}_r{}".format(prefix,train_size, val_size, test_size, seed))
        if isdir(out_dir_split) and not overwrite:
            raise FileExistsError("split already exists: {}".format(out_dir_split, out_dir))
        else:
            print("saving train-val-test split to directory {}".format(out_dir_split))
            save_split(split, out_dir_split)

    return split, out_dir_split


def main(args):
    # Load info for dataset we are making a split for
    if args.ds_path != None:
        ds = utils.load_ds(args.ds_path)
        dir = '/'.join(args.ds_path.split('/')[:-1])
        prefix = '.'.join(args.ds_path.split('/')[-1].split('.')[:-1])
    else:
        ds_info = utils.load_ds_index()[args.ds_name]
        ds = utils.load_ds(ds_info["ds_fn"])
        dir = ds_info["ds_dir"]
        prefix = args.prefix
    ds_len = ds.shape[0]

    # Create output directory
    out_dir = join(dir, "splits")
    os.makedirs(out_dir, exist_ok=True)

    # Create the split
    train_val_test(ds_len, args.train_size, args.val_size, args.test_size, args.seed, out_dir, prefix=prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@")

    parser.add_argument("--ds_name",
                        help="dataset name to create a split for (defined in datasets.yml)",
                        type=str)

    parser.add_argument("--train_size",
                        help="train set size",
                        type=float,
                        default=.8)
    parser.add_argument("--val_size",
                        help="val set size",
                        type=float,
                        default=.1)
    parser.add_argument("--test_size",
                        help="test set size",
                        type=float,
                        default=0.0)

    parser.add_argument("--seed",
                        help="random seed",
                        type=int,
                        default=1)
    parser.add_argument("--prefix",
                        help="prefix for output name",
                        type=str,
                        default='standard')
    parser.add_argument("--ds_path",
                        help="path for base dataset",
                        type=str,
                        default=None)


    main(parser.parse_args())
