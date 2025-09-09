from os.path import join
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch

import utils
from datamodules import DMSDatasetDataModule
import tasks
import parse_args




def main(args):
    # set seed for everything to the same seed used to generate splits
    seed = args.split_dir.split('_')[-1][1:] # first letter is r for random
    pl.seed_everything(seed)

    # get the uuid and log directory for this run
    my_uuid, log_dir = utils.create_log_dir(args.log_dir_base, args.uuid, args.split_dir)
    args.log_dir = log_dir # store for access to metrics file during training

    # get the version and version log directory for this run
    # the version log directory is contained within the main log directory
    # the version number starts at 0 on the first run for this UUID
    # a new version is created every time this model UUID run is restarted
    version, log_dir_version = utils.create_log_dir_version(log_dir)

    # are we resuming from checkpoint, and if so, what is the checkpoint path
    # assumes the latest checkpoint is called last.ckpt saved in the checkpoints directory
    ckpt_path = utils.get_checkpoint_path(log_dir_version)

    # set up loggers for training
    wandb_logger = WandbLogger(
        save_dir=log_dir_version,
        id=my_uuid,
        offline=not args.wandb_online,
        project=args.wandb_project,
        # disable creation of symlinks which causes problems w/ HTCondor file transfer
        settings=wandb.Settings(symlink=False))
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name='',
        version=version)

    loggers = [wandb_logger, csv_logger]

    # load full dataset
    ds_info = utils.load_ds_index()[args.ds_name]
    all_splits_df = utils.load_ds(ds_info['ds_fn'])

    # set up the datamodule
    dm = DMSDatasetDataModule(**vars(args))

    # select validation section of all_splits_df to allow for calculation of intermediate correlations
    val_df = all_splits_df.iloc[list(dm.split_idxs['val'])]

    # set up the task
    task = tasks.Task[args.task_name].cls(dm=dm,
                                          val_df=val_df,
                                          **vars(args))

    # set up callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor='val/val_loss',
        mode="min",
        dirpath=join(log_dir_version, "checkpoints"),
        save_last=False,
        save_top_k=1,
        every_n_epochs=1)
    early_stopping = EarlyStopping(monitor='val/val_loss', mode='min', min_delta=0.0, patience=10)

    callbacks.append(checkpoint_callback)
    callbacks.append(early_stopping)

    # set up the trainer from argparse args
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=log_dir,
                                            callbacks=callbacks,
                                            logger=loggers,
                                            accelerator="auto", devices="auto")#,
                                            # profiler='simple')

    # save args to the log directory for this version (these come from input args and logged hyperparameters)
    utils.save_args(vars(args), dict(task.model.hparams), join(log_dir_version, "args.txt"),
                    ignore=["cluster","process",'total_params','pre_dense_params','log_dir',
                            'dm','task','val_df','hpopt',
                            'aa_seq_len','aa_encoding_len','uuid',
                            'quality', 'classification', 'weighted_classification', 'threshold_regression'])

    # run training
    trainer.fit(task, datamodule=dm, ckpt_path=ckpt_path)

    # print out best checkpoint paths
    print('Info for best model:')
    print(checkpoint_callback.best_model_path)
    print('Validation score: ', checkpoint_callback.best_model_score)

    # prepare figures if args.make_figures == True, or just print out pearson R
    if args.make_figs:
        try: # try loading checkpoint file, if this doesn't exist then use the last epoch
            # loading models from checkpoints
            model = task.model.eval()
            checkpoint_file = checkpoint_callback.best_model_path
            checkpoint = torch.load(checkpoint_file)

            state_dict = checkpoint['state_dict'].copy()
            for k, v in checkpoint['state_dict'].items():
                state_dict['.'.join(k.split('.')[1:])] = checkpoint['state_dict'][k]
                del state_dict[k]

            print('Loading checkpoint...')
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            print('Checkpoint loading successful!')
        except:
            model = task.eval()
            print('Checkpoint loading unsuccessful, using last training epoch: beware of overfitting')

        utils.prepare_figures(args,log_dir_version,dm,model)


if __name__ == "__main__":
    # parse arguments from parameters file and save files to output
    args = parse_args.main()

    # force args.hpopt to be False, otherwise it will cause errors
    args.hpopt = False

    main(args)
