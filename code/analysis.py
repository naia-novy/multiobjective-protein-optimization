# Script for various analyses on trained model

# Import packages
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import wandb
from scipy.stats import rankdata

# WweightedCorr class from https://github.com/matthijsz/weightedcorr.git
# I confirmed that this script gives the same results as the
# R package wcorr for spearman and pearson (weighted or unweighted)

class WeightedCorr:
    def __init__(self, xyw=None, x=None, y=None, w=None, df=None, wcol=None):
        ''' Weighted Correlation class. Either supply xyw, (x, y, w), or (df, wcol). Call the class to get the result, i.e.:
        WeightedCorr(xyw=mydata[[x, y, w]])(method='pearson')
        :param xyw: pd.DataFrame with shape(n, 3) containing x, y, and w columns (column names irrelevant)
        :param x: pd.Series (n, ) containing values for x
        :param y: pd.Series (n, ) containing values for y
        :param w: pd.Series (n, ) containing weights
        :param df: pd.Dataframe (n, m+1) containing m phenotypes and a weight column
        :param wcol: str column of the weight column in the dataframe passed to the df argument.
        '''
        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError('No data supplied')
            if not ((isinstance(xyw, pd.DataFrame)) != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))):
                raise TypeError('xyw should be a pd.DataFrame, or x, y, w should be pd.Series')
            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (pd.to_numeric(xyw[i], errors='coerce').values for i in xyw.columns)
            self.df = None
        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError('df should be a pd.DataFrame and wcol should be a string')
            if wcol not in df.columns:
                raise KeyError('wcol not found in column names of df')
            self.df = df.loc[:, [x for x in df.columns if x != wcol]]
            self.w = pd.to_numeric(df.loc[:, wcol], errors='coerce')
        else:
            raise ValueError('Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)')

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])
        return self._wcov(x, y, [mx, my]) / np.sqrt(self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my]))

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv]+((counts + 1)/2 * (a/counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method='pearson'):
        '''
        :param method: Correlation method to be used: 'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        :return: if xyw, or (x, y, w) were passed to __init__ returns the correlation value (float).
                 if (df, wcol) were passed to __init__ returns a pd.DataFrame (m, m), the correlation matrix.
        '''
        if method not in ['pearson', 'spearman']:
            raise ValueError('method should be one of [\'pearson\', \'spearman\']')
        cor = {'pearson': self._pearson, 'spearman': self._spearman}[method]
        if self.df is None:
            return cor()
        else:
            out = pd.DataFrame(np.nan, index=self.df.columns, columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        out.loc[x, y] = cor(x=pd.to_numeric(self.df[x], errors='coerce'), y=pd.to_numeric(self.df[y], errors='coerce'))
                        out.loc[y, x] = out.loc[x, y]
            return out


class Analyzer:
    def __init__(self,
                 model,
                 dataset,
                 objective_names,
                 output_names,
                 loss_df,
                 ds_name,
                 log_dir,
                 classification,
                 quality_available,
                 weights_available,
                 graph_quality,
                 log_to_wandb):

        # set seaborn settings to produce high quality figures
        sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
        sns.set_style("ticks")

        # define parameters
        self.log_to_wandb = log_to_wandb
        self.objective_names = objective_names
        self.output_names = output_names
        self.log_dir = log_dir
        self.ds_name = ds_name
        self.classification = classification
        self.quality_available = quality_available
        self.output_names = self.output_names
        self.prediction_names = ['p' + n for n in self.output_names]
        # calculate weight based on inverse variance/SE
        # weighted here means something different than in training modules. If weighting is available in dataset
        # it will be used to calculate statistics and color plots, even if these weights weren't used in training
        self.weighted = weights_available

        # quality is determined by the number of sequencing counts for a given variant
        if quality_available:
            if graph_quality == 'good':
                self.quality = 1.0
            elif graph_quality == 'poor':
                self.quality = 0.0
            elif graph_quality == 'all':
                self.quality = 0.5
            else:
                raise Exception("Quality must be one of good, poor, or all")
        else:
            # If dataset doesnt contain information about quality, then graph everything
            self.quality = 0.5

        self.graph_quality = graph_quality

        # Initialize dfs
        self.plot_df = dataset
        self.loss_df = loss_df
        self.plot_df.loc[:, self.prediction_names] = [np.NaN] * len(self.prediction_names)

        # retrieve model predictions for each sequence and add to dataframe
        # Make predictions for each sequence in chosen dataset
        for index, row in self.plot_df.iterrows():
            try: pred = model.predict(row.Sequence)
            except: pred = model.predict(row.sequence)

            # Apply sigmoid function to classification outputs
            if self.classification:
                lower_bound = int(len(output_names)/2)
                # convert to numpy if necessary
                class_pred = pred[:, lower_bound:].sigmoid()
                score_pred = pred[:, :lower_bound]
                self.plot_df.loc[index, self.prediction_names] = score_pred.tolist()[0] + class_pred.tolist()[0]
            else:
                self.plot_df.loc[index, self.prediction_names] = pred.tolist()[0]

        # prep output directory
        os.makedirs(log_dir + '/analyses', exist_ok=True)

    def save_wandb(self, filename, ax):
        # log graphs to wandb if setup
        if self.log_to_wandb:
            wandb.log({filename: wandb.Image(ax.get_figure())})

    def plot_loss(self):
        # plot training and validation loss
        p = sns.lineplot(x=self.loss_df.epoch, y=self.loss_df['train/train_loss'])  # blue
        sns.lineplot(x=self.loss_df.epoch, y=self.loss_df['val/val_loss']).set_ylabel('Loss')  # orange
        plt.legend(labels=["train", "validation"], fontsize='small', title_fontsize='medium')

        filename = 'loss_train_val.pdf'
        self.save_wandb(filename, p)
        plt.savefig(f'{self.log_dir}/analyses/{filename}')
        plt.show()
        plt.clf()

        # plot BCE and MSE loss as well if model predicts classes and scores
        if self.classification:
            p = sns.lineplot(x=self.loss_df.epoch, y=self.loss_df['val/MSE_loss'])  # blue
            sns.lineplot(x=self.loss_df.epoch, y=self.loss_df['val/BCE_loss']).set_ylabel('Loss')  # orange
            plt.legend(labels=["MSE", "BCE"], fontsize='small', title_fontsize='medium')

            filename = 'loss_val_BCE_MSE.pdf'
            self.save_wandb(filename, p)
            plt.savefig(f'{self.log_dir}/analyses/{filename}')
            plt.show()
            plt.clf()

    def plot_bar_chart(self):
        # function to plot relevent statistics in a bar chart (useful for multiobjective)
        stats_df = pd.DataFrame(columns=['Objective', 'Value', 'Statistic'])
        og_stats_cols = stats_df.columns

        # calculate statistics for each objective
        avg_weights = []
        for obj_idx in range(len(self.objective_names)):
            if self.objective_names[obj_idx] != self.output_names[obj_idx]: # then likely classificaiton or T7RBD
                obj_score = self.output_names[obj_idx]
            else:
                obj_score = self.objective_names[obj_idx]
            obj = self.objective_names[obj_idx]

            cols = [f'p{obj_score}', obj_score]
            if self.quality_available: cols.append(f'{obj}_qual')
            if self.weighted: cols.append(f'{obj}_weight')
            if self.classification: cols.append(f'{obj}_class')

            temp_df = self.plot_df.copy()
            temp_df = temp_df.loc[:, cols]
            temp_df.replace([np.inf, -np.inf], np.NaN,inplace=True)
            temp_df.dropna(inplace=True)

            # set pred and true variables according to self.quality and self.weighted
            if self.quality == 0.5:
                X = temp_df[obj_score]
                Y = temp_df['p' + obj_score]
                if self.weighted:
                    W = temp_df[obj + '_weight']
                    W.reset_index(inplace=True, drop=True)
            else:
                X = temp_df[temp_df[f'{obj}_qual'] == self.quality][obj_score]
                Y = temp_df[temp_df[f'{obj}_qual'] == self.quality]['p' + obj_score]
                if self.weighted:
                    W = temp_df[temp_df[f'{obj}_qual'] == self.quality][obj + '_weight']
                    W.reset_index(inplace=True, drop=True)

            # reset index for X, Y, W otherwise WeightedCorr might return NaN
            X.reset_index(inplace=True, drop=True)
            Y.reset_index(inplace=True, drop=True)

            # this is used for determining a weighted average of the different objectives (if there are multiple)
            stats = []
            if len(X) == 0:
                # If there is not data, set stats to 0: occurs sometimes during downsampling analyses
                avg_weights.append(0.0001) # this is a psuedo count because it represents the number of observations
                MSE = 1.0
                pearsons_r = 0.0
                spearman_r = 0.0
                if self.weighted:
                    pearsons_r_weighted = 0.0
                    spearman_r_weighted = 0.0
            else:
                MSE = metrics.mean_squared_error(X, Y)

                pearsons_r = WeightedCorr(x=X,y=Y,w=pd.Series([1]*len(X)))(method='pearson')# round((np.cov(X, Y) / np.sqrt(np.cov(X, X) * np.cov(Y, Y)))[0, 1], 3)
                spearman_r = WeightedCorr(x=X,y=Y,w=pd.Series([1]*len(X)))(method='spearman')
                if self.weighted:
                    pearsons_r_weighted = WeightedCorr(x=X,y=Y,w=W)(method='pearson')# round((np.cov(X, Y, aweights=W) / np.sqrt(np.cov(X, X, aweights=W) * np.cov(Y, Y, aweights=W)))[0, 1], 3)
                    spearman_r_weighted = WeightedCorr(x=X, y=Y, w=W)(method='spearman')
                    avg_weights.append(sum(W))
                else:
                    avg_weights.append(len(X))

            # append stats to list of lists formatted for plotting
            stats.append([spearman_r, "spearman's_r"])
            stats.append([pearsons_r, "pearson's_r"])
            if self.weighted:
                stats.append([spearman_r_weighted, "spearman's_r_weighted"])
                stats.append([pearsons_r_weighted, "pearson's_r_weighted"])
            stats.append([MSE, 'MSE'])

            # convert stats list to dataframe
            stats_list = []
            for stat_idx in range(len(stats)):
                if np.isnan(stats[stat_idx][0]):
                    stats[stat_idx][0] = 0.0000
                train_stats = [obj, round(stats[stat_idx][0],4), stats[stat_idx][1]]
                stats_list.append(train_stats)

            train_stats = pd.DataFrame(stats_list, columns=og_stats_cols)
            stats_df = pd.concat([stats_df, train_stats], ignore_index=True, axis=0)

        # compute average if there are multiple training objectives
        if len(self.objective_names) >= 2:
            avg_weights = np.array(avg_weights) / sum(avg_weights)
            stats_list = []

            for stat in ["spearman's_r", "pearson's_r", "spearman's_r_weighted", "pearson's_r_weighted", 'MSE']:
                if not self.weighted and (stat == "pearson's_r_weighted" or stat == "spearman's_r_weighted"):
                    continue

                value = stats_df.loc[stats_df['Statistic'] == stat].loc[:, 'Value'].values
                value = np.sum(value * avg_weights)
                avg_stats = ['Average', value, stat]
                stats_list.append(avg_stats)

            avg_stats = pd.DataFrame(stats_list, columns=og_stats_cols)
            stats_df = pd.concat([stats_df, avg_stats], ignore_index=True, axis=0)

        # plot statistics as bar chart- maybe need to do this for each plot
        p = sns.barplot(data=stats_df,  x='Objective', y='Value', hue='Statistic', palette='tab10')
        p.set(title='Model Statistics', xlabel=None)
        p.legend(fontsize='small', title_fontsize='medium')

        filename = f'bar_stats_{self.graph_quality}.pdf'
        self.save_wandb(filename, p)
        plt.savefig(f'{self.log_dir}/analyses/{filename}')
        plt.show()
        plt.clf()

        stats_df.to_csv(f'{self.log_dir}/analyses/test_statistics_{self.graph_quality}.tsv', sep='\t', index=False)
        print(stats_df)

    def plot_all_objectives(self):
        # plot all data as scatterplot colored by objective
        for obj_idx in range(len(self.objective_names)):
            if self.objective_names[obj_idx] != self.output_names[obj_idx]: # then likely classificaiton or T7RBD
                obj_score = self.output_names[obj_idx]
            else:
                obj_score = self.objective_names[obj_idx]
            obj = self.objective_names[obj_idx]

            # set which data to plot based on self.quality
            if self.quality == 0.5:
                plot_df = self.plot_df.copy().loc[:, [f'p{obj_score}', obj_score]]
            else:
                plot_df = self.plot_df.copy()[self.plot_df[f'{obj}_qual'] == self.quality].loc[:, [f'p{obj_score}', obj_score]]
            plot_df.dropna(inplace=True)

            # plot each line individualy
            data, x, y = plot_df, obj_score,f'p{obj_score}'
            p = sns.scatterplot(data=data, x=x, y=y, s=10, palette='colorblind', alpha=0.6, edgecolor='white',
                                linewidth=.10)

        # format and save plot
        p.set(xlabel='True Score', ylabel='Predicted Score', title='Predicted vs True')
        p.legend(labels=self.objective_names, fontsize='small', title_fontsize='medium')

        filename = f'scatter_all_{self.graph_quality}.pdf'
        self.save_wandb(filename, p)
        plt.savefig(f'{self.log_dir}/analyses/{filename}')
        plt.show()
        plt.clf()

    def plot_each_objective(self):
        # individual scatterplot for each objective
        for obj_idx in range(len(self.objective_names)):
            if self.objective_names[obj_idx] != self.output_names[obj_idx]: # then likely classificaiton or T7RBD
                obj_score = self.output_names[obj_idx]
            else:
                obj_score = self.objective_names[obj_idx]
            obj = self.objective_names[obj_idx]

            # filter by quality
            if self.quality == 0.5:
                plot_df = self.plot_df.copy()
            else:
                plot_df = self.plot_df.copy()[self.plot_df[f'{obj}_qual'] == self.quality]

            plot_df.dropna(subset=f'{obj}_score', axis=0, inplace=True)

            # plot data according to whether weighting is availible
            data, x, y = plot_df, obj_score,f'p{obj_score}'
            if self.weighted:
                weight = obj + '_weight'

                norm = plt.Normalize(data[weight].min(), data[weight].max())
                sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
                sm.set_array([])

                ax = sns.scatterplot(data=data, x=x, y=y, s=20, hue=weight, palette='rocket', alpha=0.6,
                                     edgecolor='white', linewidth=.1)
                ax.set(xlabel='True Score', ylabel='Predicted Score', title=f'Predicted vs True ({obj})')
                ax.figure.colorbar(sm)

                # will be necessary to pass if there is no data (e.g. improper train-(val)-test-split
                try:
                    ax.get_legend().remove()
                except:
                    pass
            else:
                ax = sns.scatterplot(data=data, x=x, y=y, s=20, alpha=0.6, edgecolor='white', linewidth=.1)
                ax.set(xlabel='True Score', ylabel='Predicted Score', title=f'Predicted vs True ({obj})')

            # save each individual plot
            filename = f'scatter_PvT_{obj}_{self.graph_quality}.pdf'
            self.save_wandb(filename, ax)
            plt.savefig(f'{self.log_dir}/analyses/{filename}')
            plt.show()
            plt.clf()

    def plot_pr_curve(self):
        # Precision = TP/(TP+FP)
        # Recall = TPR = TP/(TP+FN)
        # plots precision recal curve using sklearn.metrics
        ax = plt.axes()
        AP_list = []
        len_sets = []

        # plot PR-curve for each classificaiton objective
        for obj in self.output_names[int(len(self.output_names) / 2):]:
            true = self.plot_df[obj]
            pred = self.plot_df['p' + obj]

            # remove datapoints where true or pred is null
            bad = ~np.logical_or(np.logical_or(pd.isnull(true), pd.isnull(pred)), pd.isnull(true))
            true = true[bad].astype(float)
            pred = pred[bad].astype(float)

            # if there is no data, add psuedocounts so rest of function can run properly
            if len(true) == 0 or len(pred) == 0:
                AP_list.append(0.0)
                len_sets.append(0.00001)
            # calculate PR-curve and display
            else:
                AP_list.append(metrics.average_precision_score(true, pred, average=None))
                len_sets.append(len(true))
                metrics.PrecisionRecallDisplay.from_predictions(true, pred, name=obj.split('_')[0], ax=ax)

        # format plot and save
        ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
        filename = 'PR_curve.pdf'
        self.save_wandb(filename, ax)
        plt.savefig(f'{self.log_dir}/analyses/{filename}')
        plt.show()
        plt.clf()

        # save classification stats to stats file created in plot_bar_chart()
        weights = np.array(len_sets)/sum(len_sets)
        mAP = sum(np.array(AP_list) * weights)

        stats_df = pd.read_csv(f'{self.log_dir}/analyses/test_statistics_good.tsv', sep='\t')
        for idx in range(len(self.objective_names)):
            stats_df = pd.concat([stats_df,pd.DataFrame([[self.objective_names[idx],AP_list[idx],'AP']],columns=stats_df.columns)])
        stats_df = pd.concat([stats_df,pd.DataFrame([['Average',mAP,'AP']],columns=stats_df.columns)])
        stats_df.reset_index(drop=True,inplace=True)

        stats_df.to_csv(f'{self.log_dir}/analyses/test_statistics_good.tsv', sep='\t')


    def plot_roc_curve(self):
        # TPR (sensitivity) = TP/(TP+FN) 
        # FPR (1-specificity) = FP/(TN+FP)
        # plot ROC-curve: this is better than PR-curve when classes are balanced

        ax = plt.axes()
        # determine ROC-curve for each objective
        for obj in self.output_names[int(len(self.output_names) / 2):]:
            true = self.plot_df[obj]
            pred = self.plot_df['p' + obj]

            # remove data that is null for true or pred
            bad = ~np.logical_or(np.logical_or(pd.isnull(true), pd.isnull(pred)),pd.isnull(true))
            true = true[bad].astype(float)
            pred = pred[bad].astype(float)

            # display ROC curve for current bjective
            if len(true) != 0 and len(pred) != 0:
                metrics.RocCurveDisplay.from_predictions(true, pred, name=obj.split('_')[0], ax=ax)

        # format and save
        ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC Curve')
        filename = 'ROC_curve.pdf'
        self.save_wandb(filename, ax)
        plt.savefig(f'{self.log_dir}/analyses/{filename}')
        plt.show()
        plt.clf()


    def plot_class_reg(self, true_score=False):
        # plot classification probabilities vs regression scores and show correlation
        cor_list = []
        # true_score states that the regression axis should represent the measured scores
        # as opposed to the predicted scores
        for obj in self.objective_names:
            if true_score:
                X = self.plot_df[f'{obj}_score']

                x_name = 'True'
            else:
                X = self.plot_df[f'p{obj}_score']
                x_name = 'Predicted'
            Y = self.plot_df[f'p{obj}_class']
            X.replace(0.0,np.NaN,inplace=True)

            if self.weighted:
                W = self.plot_df[f'{obj}_weight']
                bad = ~np.logical_or(pd.isnull(X), pd.isnull(W)) # Should be the same, but or added just in case
                W = W[bad].astype(float)
            else:
                bad = ~pd.isnull(X)

            X = X[bad].astype(float)
            Y = Y[bad].astype(float)

            if len(X) >= 2 and len(Y) >= 2:
                # pearsons_r = round((np.cov(X, Y) / np.sqrt(np.cov(X, X) * np.cov(Y, Y)))[0, 1], 3)
                spearman_r = round(WeightedCorr(x=X, y=Y, w=pd.Series([1]*len(X)))(method='spearman'), 3)

                if self.weighted:
                    spearman_r_weighted = round(WeightedCorr(x=X, y=Y, w=W)(method='spearman'), 3)
                    # pearsons_r_weighted = round(
                    #     (np.cov(X, Y, aweights=W) / np.sqrt(np.cov(X, X, aweights=W) * np.cov(Y, Y, aweights=W)))[0, 1], 3)

                    cor_list.append(f'{obj} (ρ = {spearman_r}, wρ = {spearman_r_weighted})')
                else:
                    cor_list.append(f'{obj} (ρ = {spearman_r}')

            ax = sns.scatterplot(data=self.plot_df, x=X, y=Y, s=10, alpha=0.7, edgecolor='white', linewidth=.2)
            ax.set(xlabel=f'{x_name} Score', ylabel='Predicted Class',
                   title='Correlation between classification and regression')

        # necessary for scenarios where there are no data to plot
        try:
            ax.legend(labels=cor_list, fontsize=8)
            filename = f'class_v_reg_{x_name}.pdf'
            self.save_wandb(filename, ax)
            plt.savefig(f'{self.log_dir}/analyses/{filename}')

            plt.show()
            plt.clf()
        except:
            pass

def calculate_hp_stats(model, dataset, ds_name, objective_names, output_names, log_dir, classification, quality_available):

    weights_available = True if 'weight' in ' '.join(dataset.columns) else False

    analyzer = Analyzer(model, dataset, objective_names, output_names, None, ds_name, log_dir,classification, quality_available, weights_available, graph_quality = 'all', log_to_wandb = False)

    stats_df = pd.DataFrame(columns=['Objective', 'Value', 'Statistic'])
    og_stats_cols = stats_df.columns

    avg_weights = []
    lengths = []
    rows = objective_names
    for row in rows:
        row_score =  f'{row}_score'
        if quality_available and graph_quality == 'good':
            temp_df = analyzer.plot_df.copy().loc[:, [f'p{row_score}', row_score, f'{row}_qual']]
            temp_df.dropna(inplace=True)
            X = temp_df[temp_df[f'{row}_qual'] == 1.0][row_score].copy()
            Y = temp_df[temp_df[f'{row}_qual'] == 1.0]['p' + row_score].copy()
            if weights_available:
                W = 1 / analyzer.plot_df.loc[analyzer.plot_df[f'{row}_qual'] == 1.0][row + '_weight'].copy()
                stat_name = 'spearmans_r_weighted'
        else:
            temp_df = analyzer.plot_df.copy().loc[:, [f'p{row_score}', row_score]]
            X = temp_df[row_score].copy()
            Y = temp_df['p' + row_score].copy()
            if weights_available:
                W = analyzer.plot_df[row + '_weight'].copy()
                stat_name = 'spearmans_r_weighted'

        if not weights_available:
            W = temp_df[row_score].copy()
            W.loc[~W.isna()] = 1.0
            W.rename('weight', inplace=True)
            stat_name = 'spearmans_r'

        avg_weights.append(np.nansum(W))
        lengths.append(len(X.dropna()))

        try: spearmans_r = WeightedCorr(x=X, y=Y, w=W)(method='spearman')
        except: spearmans_r = 0.0

        if np.isnan(spearmans_r):
            spearmans_r = 0.0

        stats_list = []
        for stat in [[spearmans_r, stat_name]]:
            train_stats = [row, stat[0], stat[1]]
            stats_list.append(train_stats)

        train_stats = pd.DataFrame(stats_list, columns=og_stats_cols)
        stats_df = pd.concat([stats_df, train_stats], ignore_index=True, axis=0)

    # compute weighted average of correlations by the number of observations for each objective
    avg_weights = np.array(lengths) / sum(lengths) #np.array(avg_weights) / sum(avg_weights)
    avg_weights = np.nan_to_num(avg_weights, posinf=0.0, neginf=0.0)
    stats_list = []
    value = stats_df.loc[stats_df['Statistic'] == stat_name].loc[:, 'Value'].values
    value = np.sum(value * avg_weights)
    avg_stats = ['Average', value, stat_name]
    stats_list.append(avg_stats)

    avg_stats = pd.DataFrame(stats_list, columns=og_stats_cols)
    stats_df = pd.concat([stats_df, avg_stats], ignore_index=True, axis=0)

    return stat_name, stats_df


def Analyze(model, dataset, objective_names, output_names, loss_df, should_weight, graph_quality, dm, **kwargs):

    # initialize analysis module
    analyzer = Analyzer(model, dataset, objective_names,output_names, loss_df, kwargs['ds_name'], kwargs['log_dir'],
                            dm.classification, kwargs['quality_available'], should_weight,
                            graph_quality=graph_quality, log_to_wandb=kwargs['wandb_online'])

    # analyses to conduct for all model types (reg)
    # outputs may vary depending on dataset
    analyzer.plot_loss()
    analyzer.plot_bar_chart()
    analyzer.plot_each_objective()

    if len(objective_names) >= 2:
        analyzer.plot_all_objectives()

    if dm.classification:
        analyzer.plot_pr_curve()
        analyzer.plot_roc_curve()
        analyzer.plot_class_reg(true_score=False)
        analyzer.plot_class_reg(true_score=True)

    # if quality_availible then the above would have produced good quality plots, if not, it would have defaulted to all
    if kwargs['quality_available'] and graph_quality != 'all':
        analyzer = Analyzer(model, dataset, objective_names,output_names, loss_df, kwargs['ds_name'], kwargs['log_dir'],
                            dm.classification, kwargs['quality_available'], should_weight,
                            graph_quality='all', log_to_wandb=kwargs['log_to_wandb'])
        analyzer.plot_bar_chart()
        analyzer.plot_each_objective()

        if len(objective_names) >= 2:
            analyzer.plot_all_objectives()


