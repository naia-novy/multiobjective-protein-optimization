from os.path import join
import os
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import joblib

import utils
from datamodules import DMSDatasetDataModule_hpopt
import tasks
import parse_args

# set seaborn settings to produce high quality figures
sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
sns.set_style("ticks")


def objective(trial):
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
        settings=wandb.Settings(symlink=False)
    )
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name='',
        version=version
    )
    loggers = [wandb_logger, csv_logger]

    # load full dataset
    ds_info = utils.load_ds_index()[args.ds_name]
    all_splits_df = utils.load_ds(ds_info['ds_fn'])

    # set up the datamodule
    dm = DMSDatasetDataModule_hpopt(trial=trial, **vars(args))

    # select validation section of all_splits_df
    val_df = all_splits_df.iloc[list(dm.split_idxs['val'])]

    # set up the task
    # pass in arguments from the datamodule that are important for model construction
    # other important args, like model_name, learning_rate, etc., are in the args object
    task = tasks.Task[args.task_name].cls(dm=dm,
                                          val_df=val_df,
                                          **vars(args))

    # set up callbacks
    # callbacks = [RichModelSummary(max_depth=-1)]
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor='val/val_loss',
        mode="min",
        dirpath=join(log_dir_version, "checkpoints"),
        save_last=False,
        save_top_k=1,
        every_n_epochs=1)
    early_stopping = EarlyStopping(monitor='val/val_loss', mode='min', min_delta=0.0, patience=3)
    callbacks.append(checkpoint_callback)
    callbacks.append(early_stopping)


    # set up the trainer from argparse args
    trainer = pl.Trainer.from_argparse_args(args,
                                            default_root_dir=log_dir,
                                            callbacks=callbacks,
                                            logger=loggers,
                                            precision=16,
                                            accelerator="auto", devices="auto")

    # save args to the log directory for this version (these come from input args and logged hyperparameters
    utils.save_args(vars(args), dict(task.model.hparams), join(log_dir_version, "args.txt"),
                    ignore=["cluster","process",'total_params','pre_dense_params','log_dir',
                            'dm','task','val_df','hpopt',
                            'aa_seq_len','aa_encoding_len','uuid',
                            'quality', 'classification', 'weighted_classification',  'ReLURegressor'])


    try:
        # run training
        # this is under try because if model is too large training will fail
        trainer.fit(task, datamodule=dm, ckpt_path=ckpt_path)

        # prepare figures if args.make_figures == True, or just print out pearson R
        if args.make_figs:

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
            utils.prepare_figures(args,log_dir_version,dm,model)
            print('Figure preparation complete')

        # If model training fails (ex. too large), metrics will not be saved
        val_headers = ['val/val_loss', 'val/MSE_loss', 'val/BCE_loss', 'spearmans_r_weighted']
        metrics_df = pd.read_csv(f'{log_dir_version}/metrics.csv')
        train_df = metrics_df[~metrics_df['train/train_loss'].isna()].drop(val_headers, axis=1)
        val_df = metrics_df[~metrics_df['val/val_loss'].isna()]
        val_df = val_df.loc[:, val_headers + ['epoch']]
        loss_df = pd.merge(val_df, train_df, on='epoch')
        min_idx = loss_df[['val/val_loss']].idxmin()
        min_R = loss_df.loc[min_idx, 'spearmans_r_weighted'].values[0]
        wandb_state = 'finished'

    except ValueError:
        min_R = 0.0 # needs to be penalized if it is too big, otherwise it will get stuck (None causes it to get stuck)
        print('min_R', min_R)
        print(f'Model training failed! This is often because #params > {args.max_parameters}')
        wandb_state = 'too_large'

    if args.wandb_online:
        # report the final validation accuracy to wandb
        wandb_logger.experiment.config["R_final"] = min_R
        wandb_logger.experiment.config["state"] = wandb_state
        wandb_logger.experiment.finish()

    return min_R


def main(args):
    # set seed for everything to the same seed used to generate splits
    seed = args.split_dir.split('_')[-1][1:] # first letter is r for random [seed]
    pl.seed_everything(seed)

    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_startup_trials=0, n_min_trials=5), patience=3)
    n_trials = 50
    current_trial = 0 # initialize current trial with 0
    while current_trial < n_trials:
        # If using classification as an option, maximize the validation correlation, otherwise maximize the validation loss
        direction = 'maximize' if 'classification' in ' '.join(args.loss_method) or True in args.hp_weighted else 'minimize'


        # load study if it exists, otherwise make new study
        try: study = joblib.load(f"{args.log_dir_base}/hpopt.pkl")
        except: study = optuna.create_study(study_name='hpopt', direction=direction, pruner=pruner)
        print("\nTrials so far: ", len(study.trials))

        # optimize a single time then save to allow for checkpointing after each iteration
        study.optimize(objective, n_trials=1)
        joblib.dump(study, f"{args.log_dir_base}/hpopt.pkl")


        # determine the current number of trials that have been completed
        current_trial = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])  # select non pruned
        current_trial = len([i.value for i in current_trial if i.value != 0.0])  # select models that were not too big

        # prepare plots of parameter importance and objective funciton history
        if current_trial % 5 == 0 and current_trial > 0:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(f'{args.log_dir_base}/param_importance.pdf', bbox_inches='tight', dpi=400, facecolor='white', transparent=True)
            plt.show()
            plt.clf()
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(f'{args.log_dir_base}/optim_history.pdf', bbox_inches='tight', dpi=400, facecolor='white', transparent=True)
            plt.show()
            plt.clf()

    # get trial information
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    trial = study.best_trial
    best_params = pd.DataFrame(trial.params.items())

    # report optimization statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(" Best parameters: ", best_params)

    # make filepath and filename to save arguments of best model for easy loading later
    if not os.path.isdir('output/optimized_models'):
        os.mkdir('output/optimized_models')
    outdir = f'output/optimized_models/{args.log_dir_base.split("/")[2]}/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    now = datetime.now()
    fn = now.strftime(f"%Y%m%d_%H%M%S_{args.log_dir_base.split('/')[2]}_args.txt")

    # reformat determined biases for downstream training args input
    bias_cats = [trial.params[i] for i in trial.params.keys() if 'bias_' in i]
    bias_dict = dict(zip(list(set(args.output_bias[0].split(' '))), bias_cats))
    args.output_bias = [' '.join([str(bias_dict[i]) for i in args.output_bias[0].split(' ')])]
    # save best args (these come from input args and logged hyperparameters
    utils.save_args(vars(args), trial.params, join(outdir, fn),
                    ignore=["cluster","process",'total_params','pre_dense_params','log_dir',
                            'dm','task','val_df','hpopt',
                            'aa_seq_len','aa_encoding_len','uuid',
                            'quality', 'classification', 'weighted_classification',  'ReLURegressor']+bias_cats)


if __name__ == "__main__":
    # parse arguments from parameters file and # todo save files to output
    args = parse_args.main()

    # This is necessary for wandb 0.13.10 because of a bug that stalls training randomly
    os.environ["WANDB_DISABLE_SERVICE"] = "true"

    main(args)
