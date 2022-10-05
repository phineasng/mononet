import os
import collections
import argparse
import shap
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.nn import init
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, Birch, SpectralClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp

# from MonoNet_class import *
from MonoNet_class import *
from MonoNet_train import MonoNet_train

from generation_interventional_data import generate_interv
from captum.attr import NeuronGradientShap, NeuronGradient, NeuronIntegratedGradients, NeuronConductance, \
    NeuronDeepLift, NeuronDeepLiftShap
from lime.lime_tabular import LimeTabularExplainer

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def get_class_distribution(obj):
    count_dict = collections.Counter(obj + 1)
    count_dict = dict(sorted(count_dict.items()))

    return count_dict


def plot_training_progress(model_):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(model_.loss_hist['val'], label="val")
    axs[0].plot(model_.loss_hist['train'], label="train")
    axs[0].set(xlabel="Epochs", ylabel="Loss", title="Training and Validation Loss")
    axs[0].legend()

    axs[1].plot(model_.accuracy_hist['val'], label="val")
    axs[1].plot(model_.accuracy_hist['train'], label="train")
    axs[1].set(xlabel="Epochs", ylabel="Accuracy", title="Training and Validation Accuracy")
    axs[1].legend()
    plt.savefig(save_path + '/loss_accuracy_training.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_ks_across_layers(df_):
    if not os.path.exists(save_path + '/Stat_ana_activation_patterns/'):
        os.makedirs(save_path + '/Stat_ana_activation_patterns/')
    plt.figure(figsize=(13, 9))
    ax = sns.pointplot(x="layer", y="value", hue="variable",
                       data=df_, ci=None)

    ax.set(xlabel='Biomarker', ylabel='Layer')
    plt.xticks(rotation=45)
    ax.set_title('Average KS score across layers')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/Stat_ana_activation_patterns/plots_KS_score across layers.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()


def analysis_by_neuron(h_values, idx_neuron, task, X_input, y_input, perc=0.2):
    i = idx_neuron
    if not os.path.exists(save_path + '/Stat_ana_activation_patterns/Violin_plots/'):
        os.makedirs(save_path + '/Stat_ana_activation_patterns/Violin_plots/')
    if not torch.is_tensor(h_values):
        h_values = torch.from_numpy(h_values).clone().to(torch.float32)
    h_i = h_values[:, i]
    sorted_h_i, indices = torch.sort(h_i)
    nb_obs = round(perc * len(h_i))
    idx_bottom = indices[0:nb_obs]
    idx_top = indices[-nb_obs:]

    input_values_bottom = X_input[idx_bottom]
    input_values_top = X_input[idx_top]

    top_bottom_labels = np.hstack(
        (np.full(nb_obs, str('bottom')), np.full(nb_obs, str('top'))))
    top_bottom_input = np.vstack((input_values_bottom, input_values_top))

    df_top_bottom = pd.DataFrame(
        data=np.hstack((top_bottom_input, top_bottom_labels.reshape((len(top_bottom_labels), 1)))),
        columns=list(df.columns[:-1]) + ['label'])

    if task == 'KS_stat':
        stats_KS = []
        pvalues_KS = []

        for feat in list(df.columns[:-1]):
            x_bottom = np.array(df_top_bottom[feat][0:nb_obs])
            x_top = np.array(df_top_bottom[feat][-nb_obs:])
            stat, pvalue = ks_2samp(x_bottom, x_top)
            stats_KS += [stat]
            pvalues_KS += [pvalue]

        return stats_KS, pvalues_KS

    if task == 'violin_plot':
        df_top_bottom_melt = pd.melt(df_top_bottom, id_vars=['label'], ignore_index=False)
        df_top_bottom_melt['label'] = df_top_bottom_melt['label'].astype("category")
        df_top_bottom_melt['variable'] = df_top_bottom_melt['variable'].astype("category")
        df_top_bottom_melt['value'] = df_top_bottom_melt['value'].astype('float64')

        plt.figure(figsize=(8, 5))
        sns.violinplot(x="variable", y="value", hue="label",
                       data=df_top_bottom_melt, palette="Set2", split=True,
                       scale="count")
        plt.legend(title='', fontsize=12)
        plt.xlabel('', fontsize=14)
        plt.ylabel('Biomarker distribution', fontsize=16)
        # plt.title('Interpretable Neuron {}'.format(i + 1), fontsize=16)
        plt.tick_params(axis='y', which='major', labelsize=12)
        plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
        plt.savefig(save_path + '/Stat_ana_activation_patterns/Violin_plots/violin_plot_neuron_{}.eps'.format(i + 1),
                    format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def plots_distance_measure(df_measure, measure, p_values):
    if not os.path.exists(save_path + '/Stat_ana_activation_patterns/KS_neuron/'):
        os.makedirs(save_path + '/Stat_ana_activation_patterns/KS_neuron/')

    d = pd.melt(df_measure, id_vars=['neuron'], ignore_index=False)
    idx_sort = np.argsort(df_measure.iloc[:, :-1], axis=1)
    idx_sort.columns = [i for i in range(13)]
    for i in range(NB_NEURON_INTER_LAYER):
        df_i = pd.Series(df_measure.iloc[i, :-1]).reindex(index=list(
            pd.Series(list(df_measure.iloc[i, :-1].index)).reindex(
                index=list(idx_sort.iloc[i, :].values)).values)).to_frame().reset_index()
        df_i['neuron'] = [i + 1] * 13
        df_i.columns = ['variable', 'value', 'neuron']
        plt.figure(figsize=(6, 5))
        sns.pointplot(x="variable", y="value", data=df_i, ci=None, join=False,
                      palette=sns.color_palette())
        plt.xlabel('', fontsize=16)
        plt.ylabel('KS score', multialignment='center', fontsize=16)
        plt.tick_params(axis='y', which='major', labelsize=12)
        plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + f'/Stat_ana_activation_patterns/KS_neuron/KS_scores_neuron_{i + 1}.eps', format='eps',
                    dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

    plt.figure(figsize=(7, 5))
    sns.pointplot(x="variable", y="value", data=d, join=False, palette=sns.color_palette())
    plt.xlabel('', fontsize=16)
    plt.ylabel('Average KS score', multialignment='center', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/Stat_ana_activation_patterns/KS_scores_average.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    palette_13 = sns.color_palette(
        list(sns.color_palette('tab10')) + [sns.color_palette('Set1')[5]] + [sns.color_palette('Set1')[8]] + [
            sns.color_palette('Accent')[7]])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.pointplot(ax=axs[0], x="variable", y="value", data=d, join=False, palette=sns.color_palette())
    axs[0].set_xlabel('Biomarker', fontsize=16)
    axs[0].set_ylabel('Average KS score', fontsize=16)
    axs[0].tick_params(axis='y', which='major', labelsize=12)
    axs[0].tick_params(axis='x', which='major', labelsize=14, labelrotation=45)

    sns.stripplot(ax=axs[1], x="neuron", y="value", hue="variable", data=d, jitter=0.1)
    axs[1].legend(title='Biomarkers', fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_xlabel('Neuron', fontsize=16)
    axs[1].set_ylabel('KS score', fontsize=16)
    axs[1].tick_params(axis='y', which='major', labelsize=12)
    axs[1].tick_params(axis='x', which='major', labelsize=14)

    plt.savefig(save_path + '/Stat_ana_activation_patterns/KS_scores_average_&_neuron.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()


def compute_ks(h_values_, X_input_, y_input_):
    df_stats_KS_ = pd.DataFrame(columns=list(df.columns[:-1]))
    df_pvalues_KS_ = pd.DataFrame(columns=list(df.columns[:-1]))
    for i in range(NB_NEURON_INTER_LAYER):
        stat_ks, pvalues_ks = analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='KS_stat',
                                                 X_input=X_input_, y_input=y_input_)
        df_stats_KS_.loc[i] = stat_ks
        df_pvalues_KS_.loc[i] = pvalues_ks
    df_stats_KS_['neuron'] = np.arange(8) + 1
    df_pvalues_KS_['neuron'] = np.arange(8) + 1
    return df_stats_KS_, df_pvalues_KS_


def violin_plots(h_values_, X_input_, y_input_):
    for i in range(NB_NEURON_INTER_LAYER):
        analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='violin_plot',
                           X_input=X_input_, y_input=y_input_)


def visualize_importances_all_neurons(feature_names, importances, title="Average Feature Importances",
                                      method="captum", axis_title="Biomarkers", save_path='.'):
    if not os.path.exists(save_path + '/feature_attributions/'):
        os.makedirs(save_path + '/feature_attributions/')

    if torch.is_tensor(importances[list(importances.keys())[0]]):
        importances = {k: v.detach().numpy() for k, v in importances.items()}
        print('change it to numpy')
    x_pos = (np.arange(len(feature_names)))

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))

    fig.suptitle(title + method)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, importances['neuron_1'].mean(axis=0))
    axs[0, 0].set_title('Neuron 1')
    axs[0, 1].bar(x_pos, importances['neuron_2'].mean(axis=0))
    axs[0, 1].set_title('Neuron 2')
    axs[0, 2].bar(x_pos, importances['neuron_3'].mean(axis=0))
    axs[0, 2].set_title('Neuron 3')
    axs[0, 3].bar(x_pos, importances['neuron_4'].mean(axis=0))
    axs[0, 3].set_title('Neuron 4')
    axs[1, 0].bar(x_pos, importances['neuron_5'].mean(axis=0))
    axs[1, 0].set_title('Neuron 5')
    axs[1, 1].bar(x_pos, importances['neuron_6'].mean(axis=0))
    axs[1, 1].set_title('Neuron 6')
    axs[1, 2].bar(x_pos, importances['neuron_7'].mean(axis=0))
    axs[1, 2].set_title('Neuron 7')
    axs[1, 3].bar(x_pos, importances['neuron_8'].mean(axis=0))
    axs[1, 3].set_title('Neuron 8')

    for ax in axs.flat:
        ax.set(xlabel=axis_title)
        plt.sca(ax)
        plt.xticks(rotation=45)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(save_path, "shap_values.png"), dpi=350)


def monotonic_f(h_and_class_, neuron_, target_class_, fixed_class_, model_):
    if not os.path.exists(save_path + '/Monotonic_block/Monotonic_function/'):
        os.makedirs(save_path + '/Monotonic_block/Monotonic_function/')

    # x_lim_min = h_and_class_[:, neuron_].min()
    # x_lim_max = h_and_class_[:, neuron_].max()

    x_ = np.arange(-50, 50, 0.01)
    x_ = np.arange(-1.5, 1.5, 0.01)
    x_ = np.arange(-1, 1, 0.01)
    # Extract values of the interpretable neurons for the fixed chosen class
    h_fixed_class = h_and_class_[h_and_class_[:, -1] == fixed_class_ - 1, :]
    # TODO: Do that for predicted class instead?
    # Calculate the mean for each interpretable neuron
    mean_class = np.mean(h_fixed_class, axis=0)
    h_mon = np.full([len(x_), h_values.size()[1]], None)
    for idx_n in range(NB_NEURON_INTER_LAYER):
        h_mon[:, idx_n] = np.repeat(mean_class[idx_n], len(x_))
    # Replace in the neuron of interest the vector x_
    h_mon[:, neuron_ - 1] = x_

    # Pass the h_values into the monotonic block
    y_ = model_.monotonic_block(torch.from_numpy(h_mon.astype('float32')))[:, target_class_ - 1].detach().numpy()

    # x_lim_min = np.min(np.where(np.abs(y_[:-1] - y_[1:]) > 0)[0])
    # x_lim_max = np.max(np.where(np.abs(y_[:-1] - y_[1:]) > 0)[0])
    x_ = x_  # [x_lim_min:x_lim_max]
    y_ = y_  # [x_lim_min:x_lim_max]

    # Calculate the density for the chosen interpretable neuron

    density = gaussian_kde(h_fixed_class[:, neuron_ - 1])
    xs = np.linspace(-1.5, 1.5, 500)
    density.covariance_factor = lambda: .25
    density._compute_covariance()

    plt.figure(figsize=(7, 5))
    plt.plot(x_, y_)
    plt.xlabel(f'Neuron {neuron_}', fontsize=14)
    plt.ylabel(f'Activation value of the output neuron\ncorresponding to {meta_df.loc[target_class_ - 1][1]}',
               multialignment='center', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/Monotonic_function/NEW_mono_f_neuron{}_targetclass{}_fixedclass{}.eps'
                .format(neuron_, target_class_ - 1, fixed_class_ - 1), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(x_, y_, color="tab:blue")
    # set x-axis label
    ax.set_xlabel('Neuron {} range for {}'.format(neuron_, meta_df.loc[fixed_class_ - 1][1], fontsize=12))

    # set y-axis label
    ax.set_ylabel('Class {} ({})'.format(target_class_, meta_df.loc[target_class_ - 1][1]), color="tab:blue",
                  fontsize=12)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(xs, density(xs), color="tab:orange")
    ax2.set_ylabel("Density", color="tab:orange", fontsize=12)
    plt.show()
    # save the plot as a file
    fig.savefig(save_path + '/Monotonic_function/mono_f_neuron{}_targetclass{}_fixedclass{}.eps'
                .format(neuron_, target_class_ - 1, fixed_class_ - 1), format='eps', dpi=300, bbox_inches='tight')
    plt.close()


def plot_clusterings(df_1, df_2):
    plt.figure(figsize=(7, 5.5))
    sns.lineplot(data=df_clustering_y_true, dashes=False,
                 palette=sns.color_palette([sns.color_palette('tab10')[0]] + [sns.color_palette('tab10')[2]]))
    sns.lineplot(data=df_clustering_y_pred, dashes=False, palette=sns.color_palette(
        sns.color_palette([sns.color_palette('tab10')[1]] + [sns.color_palette('tab10')[3]])))
    plt.legend(title='', fontsize=13)
    plt.xlabel("", fontsize=18)
    plt.ylabel("Adjusted mutual information score", fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=16, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/information_flow/adjusted_mutual_info_score.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()


def create_df_activation(dict_layers_mean, abs):
    average_act = []

    for name in name_layers:
        if abs == True:
            df_temp = pd.DataFrame(dict_layers_mean[name].iloc[:, :-1].abs().mean(axis=0))
        else:
            df_temp = pd.DataFrame(dict_layers_mean[name].iloc[:, :-1].mean(axis=0))

        df_temp.columns = ['value']
        df_temp['pred_cell_type'] = list(df_temp.index)
        df_temp['layer'] = [name] * len(df_temp)

        average_act.append(df_temp)

    return pd.concat(average_act, ignore_index=True)


def plot_activation_mean(df_average, abs, bool_legend=False):
    plt.figure(figsize=(7.5, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_average, ci=None)
    if bool_legend:
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend([], [], frameon=False)

    plt.ylim([-1, 1])

    plt.xlabel('', fontsize=16)
    if abs:
        plt.ylabel('Mean absolute activation', fontsize=18)
    else:
        plt.ylabel('Mean activation', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()
    if abs:
        plt.savefig(save_path + '/activation_across_layers/plots_mean_abs_activation_across_layers.eps', format='eps',
                    dpi=300,
                    bbox_inches='tight')
    else:
        plt.savefig(save_path + '/activation_across_layers/plots_mean_activation_across_layers.eps', format='eps',
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    plt.close()


def create_df_meas(meas_):
    measure_entropy = []
    for name in name_layers:
        index = pd.Index(y_pred_MonoNet, name='label')
        df_ = pd.DataFrame(dict_values_layers[name], index=index)
        groups = {k: v.values for k, v in df_.groupby('label')}
        measure_variability = []
        for label, values in groups.items():

            cov_mat = np.cov(np.transpose(values))
            if meas_ == 'trace':
                meas = np.trace(cov_mat)
            elif meas_ == 'det':
                meas = np.linalg.det(cov_mat)
            elif meas_ == 'eigen_max':
                meas = np.max(np.linalg.eig(cov_mat)[0]).real
            elif meas_ == 'eigen_min':
                meas = np.min(np.linalg.eig(cov_mat)[0])
            measure_variability += [meas]

        df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])

        df_.index = list(meta_df.iloc[classes_pred, 1])
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)

        measure_entropy.append(df_)
    return pd.concat(measure_entropy, ignore_index=True)


def plot_entropy(df_, meas_):
    if meas_ == 'trace':
        title_meas = 'Trace'
    elif meas_ == 'det':
        title_meas = 'Determinant'
    elif meas_ == 'eigen_max':
        title_meas = 'Max eigenvalue'
    elif meas_ == 'eigen_min':
        title_meas = 'Min eigenvalue'
    plt.figure(figsize=(9.6, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_, ci=None)

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('')
    plt.ylabel(f'{title_meas} of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + f'/activation_across_layers/{meas_}_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MonoNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', type=bool, default=False, help='Specify if one should load or train a MonoNet')
    parser.add_argument('--save_root', type=str, default='.', help='Root folder where all the results files/subfolders'
                                                                   'will be stored. Default: "./"')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root folder where to find the data.csv and metadata.csv for the single cell example.')
    parser.add_argument('--in_file_NN', type=str, help='Path to a pretrained model. '
                                                       'If given, no training will be performed.')
    # default='/Users/dam/Documents/GitHub/MonoNet_single_cell/nn_30_12_90_saved.pickle')
    # --in_file_NN
    # /Users/dam/Documents/GitHub/MonoNet_single_cell/nn_12_12_87_saved.pickle
    # /Users/dam/Documents/GitHub/MonoNet_single_cell/nn_15_12_86_saved.pickle
    parser.add_argument('--out_file_NN', type=str, required=False,
                        help='Where to store the model if one is trained. '
                             'If not provided, defaults to <save_root>/model.pth')
#                        default='/Users/dam/Documents/GitHub/MonoNet_single_cell/nn_15_12_saved.pickle')

    parser.add_argument('--verbose', type=int, default=0)

    parser.add_argument('--compute_shap_values', action='store_true')
    parser.add_argument('--analysis_with_violin_plots', action='store_true')
    parser.add_argument('--analysis_clustering', action='store_true')
    parser.add_argument('--information_analysis', action='store_true')

    ####################################################################################################################
    # --------------------------------------- PARSE ARGUMENTS ----------------------------------------------------------
    ####################################################################################################################
    args = parser.parse_args()
    out_file_nn = args.out_file_NN
    in_file_nn = args.in_file_NN
    verbose = args.verbose
    save_root = args.save_root
    compute_shap_values = args.compute_shap_values
    plot_violin = args.analysis_with_violin_plots
    clustering = args.analysis_clustering
    information_analysis = args.information_analysis

    ####################################################################################################################
    # --------------------------------------- DATA PREPARATION AND VISUALIZATION ---------------------------------------
    ####################################################################################################################

    # Loading data
    meta_df = pd.read_csv(os.path.join(args.data_root, 'metadata.csv'))
    df = pd.read_csv(os.path.join(args.data_root, 'data.csv'))

    # Plotting class distribution of the whole dataset
    if verbose > 0:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1.7, 1]})

        sns.countplot(x='category', data=df, ax=axs[0])
        axs[0].set(xlabel='Cell type', ylabel='Number of samples', title="Class distribution")
        axs[1].axis('tight')
        axs[1].axis('off')
        table = axs[1].table(cellText=meta_df.values.tolist(), loc='center', colWidths=[0.2, 0.8])
        table.set_fontsize(9)
        plt.savefig(os.path.join(save_root, 'class_distribution.eps'), format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

    # Encode the classes form 0 to 19 instead of 1-20
    df.category = df.category - 1

    # Separate predictors and classes values
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval,
                                                      random_state=21)

    # Rescaling of the x-values for each subset
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Plotting the class distribution for each subset
    if verbose > 0:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
        # Train
        sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x="variable", y="value",
                    hue="variable", ax=axes[0]).set_title('Class Distribution in Train Set')
        axes[0].legend(loc='upper right')
        # Validation
        sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x="variable", y="value",
                    hue="variable", ax=axes[1]).set_title('Class Distribution in Validation Set')
        axes[1].legend(loc='upper right')
        # Test
        sns.barplot(data=pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x="variable", y="value",
                    hue="variable", ax=axes[2]).set_title('Class Distribution in Test Set')
        axes[2].legend(loc='upper right')
        plt.close()
    # Prepare the subtests for pytorch
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # Use of WeightedRandomSampler to deal with class imbalance
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)

    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    ########################################################################################################################
    # --------------------------------------- TRAIN NEURAL NETWORK ---------------------------------------
    ########################################################################################################################

    # Hyperparameters for the Neural network
    EPOCHS = 80
    BATCH_SIZE = 40  # 40
    LEARNING_RATE = 0.00055  # 0.00045
    # with uniform weights init
    # 0.0002 not bad until 50 epochs ( reached 70% and another time 60% at 50 epochs) but then plateau
    # 0.0001 moins bien
    # 0.0003 moins bien reached 60% (until 50epochs)
    # 0.0008 reach rapidly 70% (epoch 27) then plateau
    # seems much better with normal weights (at least first epochs)
    # 0.0007 reach rapidly 70% but then plateau
    NUM_FEATURES = len(X.columns)
    NUM_CLASSES = len(np.unique(y_train))
    NB_NEURON_INTER_LAYER = 8

    # avec l'ancienne version 13 128 64 8 75 epochs 50 0.0002
    '''model = MonoNet_train(EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_FEATURES, NUM_CLASSES, 13,
                          train_dataset, val_dataset, test_dataset, class_weights, weighted_sampler, X_test, y_test,
                          out_file_nn)'''

    if in_file_nn is not None:
        model = torch.load(in_file_nn)
        saved_model = in_file_nn
    else:
        if out_file_nn is None:
            out_file_nn = os.path.join(save_root, 'model.pth')
        model = MonoNet_train(EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_FEATURES, NUM_CLASSES, NB_NEURON_INTER_LAYER,
                              "xavier_uniform", train_dataset, val_dataset, test_dataset, class_weights,
                              weighted_sampler, X_test, y_test, out_file_nn=out_file_nn)
        saved_model = in_file_nn

    # create results subfolder
    fname = os.path.basename(saved_model)
    fname_no_ext = os.path.splitext(fname)[0]
    save_path = os.path.join(args.save_root, fname_no_ext)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_training_progress(model)
    ####################################################################################################################
    # --------------------------------------- Data Preparation ---------------------------------------
    ####################################################################################################################

    # Selection of the data we will use
    subset = 'test'
    X_subset = eval('X_{}'.format(subset))
    y_subset = eval('y_{}'.format(subset))

    torch_X_subset = torch.from_numpy(X_subset).clone().to(torch.float32)

    # Compute values of the neurons in the interpretable layer (h_values)
    h_values = model.unconstrained_block(torch_X_subset)

    # Combine X_input and h values in one dataframe
    df_input_h = pd.DataFrame(data=np.hstack((X_subset, h_values.detach().numpy(), y_subset.reshape(len(y_subset), 1))),
                              columns=list(df.columns[:-1]) + [str('Neuron_') + str(i + 1) for i in
                                                               range(h_values.detach().numpy().shape[1])]
                                      + ['category'])

    df_input_h.to_csv('csv_input_h.csv')

    # Verify the values
    output = model.monotonic_block(h_values)
    output_2 = model.forward(torch_X_subset)
    if not torch.all(torch.eq(output_2, output)).item():
        print('There should be an error in the forward, unconstrained_block or monotonic_block functions')

    # Randomly select some of the sample (to reduce computational time)

    idx_by_class_reduced = {}
    for i in range(len(np.unique(y_subset))):
        temp_idx = np.where(y_subset == i)[0]
        np.random.shuffle(temp_idx)
        idx_by_class_reduced[str('class_') + str(i)] = list(temp_idx)[0:min(len(temp_idx), 50)]

    idx_all_class_redu = [item for sublist in list(idx_by_class_reduced.values()) for item in sublist]

    X_reduced = X_subset[idx_all_class_redu]
    y_reduced = y_subset[idx_all_class_redu]
    h_values_reduced = model.unconstrained_block(X_reduced)
    y_pred_reduced = model(X_reduced)
    y_pred_reduced_softmax = torch.log_softmax(y_pred_reduced, dim=1)
    _, y_pred_reduced_tags = torch.max(y_pred_reduced_softmax, dim=1)

    y_pred = model(X_subset)
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    #
    name_layers = ['input', 'linear_1', 'linear_2', 'interpretable', 'pre_monotonic', 'monotonic_1', 'monotonic_2',
                   'output']
    dict_values_layers = {}
    for name in name_layers:
        if name == "input":
            dict_values_layers[name] = X_subset
        elif name == "output":
            dict_values_layers[name] = model(X_subset).detach().numpy()
        # elif name == "Pre-monotonic":
        # dict_values_layers[name] = model.get_Premonotonic(X_subset).detach().numpy()
        else:
            fct = eval(f"model.get_{name}")
            dict_values_layers[name] = fct(X_subset).detach().numpy()
    dict_values_layers_mean = {}
    dict_values_layers_mean_positive = {}
    dict_values_layers_mean_abs = {}
    y_pred_MonoNet = y_pred_tags.detach().numpy().astype("int")

    classes_pred = list(np.unique(y_pred_MonoNet))

    ####################################################################################################################
    # --------------------------------------- Simple classifier ---------------------------------------
    ####################################################################################################################
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    rf = RandomForestClassifier(max_depth=13, random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(torch.from_numpy(y_pred_rf.astype('float32')), torch.from_numpy(y_test.astype('float32')))]
    acc = matches.count(True) / len(matches)

    ####################################################################################################################
    # --------------------------------------- Unconstrained block ---------------------------------------
    ####################################################################################################################

    # ---- Statistical analysis of activation patterns -----

    if plot_violin:
        dict_df_stats_KS = {}
        dict_df_pvalues_KS = {}
        all_stat_KS = []
        for name in name_layers:
            values_layers = dict_values_layers[name]
            dict_df_stats_KS[name], dict_df_pvalues_KS[name] = compute_ks(h_values_=values_layers, X_input_=X_subset,
                                                                          y_input_=y_subset)
            df_temp = pd.melt(dict_df_stats_KS[name], id_vars=['neuron'], ignore_index=False)[['variable', 'value']]
            df_temp['layer'] = [name] * len(df_temp)
            all_stat_KS.append(df_temp)

        df_all_stat_KS = pd.concat(all_stat_KS, ignore_index=True)

        plot_ks_across_layers(df_=df_all_stat_KS)

        # Violin plots for the top and bottom distribution of the input (the ordering is done wrt the h_values)
        violin_plots(h_values_=h_values, X_input_=X_subset, y_input_=y_subset)
        plots_distance_measure(df_measure=dict_df_stats_KS['interpretable'],
                               p_values=dict_df_pvalues_KS['interpretable'], measure='KS')

        # Is there a difference between the top and bottom distributions? (H_0 = the distributions are the same so
        # if p<0.05 we reject the hypothesis that the distributions are the same)

        df_diff_in_distributions_KS = dict_df_pvalues_KS['interpretable'] < 0.05
        df_diff_in_distributions_KS.index = list(range(1, 9, 1))

    # ------------ Feature attribution methods ------------

    if compute_shap_values:
        exp_shap = {}
        X_train_summary = shap.kmeans(X_train, 50)
        for i in range(NB_NEURON_INTER_LAYER):
            explainer_shap = shap.KernelExplainer(model=eval(f"model.unconstrained_block_{i}"),
                                                  data=X_train_summary)
            # TODO: Which dataset should we use?, what should we put there as reference ?
            exp_shap[str('neuron_') + str(i + 1)] = \
                explainer_shap.shap_values(shap.sample(X_test, 100))[
                    0]  # TODO should we take idx_all_class_redu
            # shap.force_plot(explainer_shap.expected_value[0], exp_shap[str('neuron_') + str(i + 1)], X_test)

            # shap.summary_plot(shap_values[0], X_test[0:50])
        visualize_importances_all_neurons(list(df.columns)[:-1], exp_shap,
                                          title="Average Feature Importances for ",
                                          method="Shapley Values approximation", save_path=save_path)


    # ------------------- Causal models --------------------

    ####################################################################################################################
    # --------------------------------------- Monotonic block ---------------------------------------
    ####################################################################################################################
    if not os.path.exists(save_path + '/Monotonic_block/'):
        os.makedirs(save_path + '/Monotonic_block/')

    matrix_sign = torch.matmul(model.pre_monotonic.weight.data.reshape((8, 1)),
                               torch.transpose(model.output.weight.data.reshape((20, 1)), 0,
                                               1)).detach().numpy()
    inverse_elem_wise = lambda x: 1 / x
    inv_matrix_sign = inverse_elem_wise(matrix_sign)
    data_fr = pd.DataFrame(data=inv_matrix_sign, columns=list(meta_df.iloc[:, 1]), index=[f'{i + 1}' for i in range(8)])

    plt.figure(figsize=(8, 5.5))
    ax = sns.heatmap(data_fr, square=True, cmap='coolwarm', cbar_kws={"shrink": .50})
    ax.set_ylabel('Interpretable neuron', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12, rotation=0)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path + '/Monotonic_block/sign_matrix_".eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.close()

    h_and_class = np.hstack((h_values.detach().numpy(), y_subset.reshape(len(y_subset), 1)))

    '''for i in range(NB_NEURON_INTER_LAYER):
        j = random.randint(0, 20)
        k = random.randint(0, 20)
        l = random.randint(0, 20)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=j, fixed_class_=j, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=k, fixed_class_=k, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=l, fixed_class_=l, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=j, fixed_class_=k, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=k, fixed_class_=j, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=j, fixed_class_=l, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=l, fixed_class_=j, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=k, fixed_class_=l, model_=model)
        monotonic_f(h_and_class_=h_and_class, neuron_=i, target_class_=l, fixed_class_=k, model_=model)'''
    ####################################################################################################################
    # ------------------------------ Where does the classification happen ? ----------------------------------
    ####################################################################################################################

    if information_analysis:

        # ----------- Information flow across layers -----------
        if clustering:
            nb_cluters = 20
            kmeans = KMeans(init="random", n_clusters=nb_cluters, n_init=10, max_iter=300, random_state=42)
            kmeans_fits = {}
            kmeans_adjusted_mutual_info_scores_y_true = {}
            kmeans_adjusted_mutual_info_scores_y_pred = {}
            for key, values_layer in dict_values_layers.items():
                kmeans_fits[key] = kmeans.fit(values_layer)
                kmeans_adjusted_mutual_info_scores_y_true[key] = adjusted_mutual_info_score(kmeans_fits[key].labels_,
                                                                                            y_subset)
                kmeans_adjusted_mutual_info_scores_y_pred[key] = adjusted_mutual_info_score(kmeans_fits[key].labels_,
                                                                                            y_pred_MonoNet)

            #
            birch = Birch(threshold=0.01, n_clusters=nb_cluters)
            birch_fits = {}
            birch_adjusted_mutual_info_scores_y_true = {}
            birch_adjusted_mutual_info_scores_y_pred = {}
            for key, values_layer in dict_values_layers.items():
                birch_fits[key] = birch.fit(values_layer)
                birch_adjusted_mutual_info_scores_y_pred[key] = adjusted_mutual_info_score(birch.predict(values_layer),
                                                                                           y_pred_MonoNet)
                birch_adjusted_mutual_info_scores_y_true[key] = adjusted_mutual_info_score(birch.predict(values_layer),
                                                                                           y_subset)

            df_clustering_y_true = pd.DataFrame(
                data=np.transpose(np.vstack((np.array(list(kmeans_adjusted_mutual_info_scores_y_true.values())),
                                             np.array(list(birch_adjusted_mutual_info_scores_y_true.values()))))),
                columns=['K means - true class', 'BIRCH - true class'])
            df_clustering_y_true.index = name_layers
            df_clustering_y_pred = pd.DataFrame(
                data=np.transpose(np.vstack((np.array(list(kmeans_adjusted_mutual_info_scores_y_pred.values())),
                                             np.array(list(birch_adjusted_mutual_info_scores_y_pred.values()))))),
                columns=['K means - predicted class', 'BIRCH - predicted class'])
            df_clustering_y_pred.index = name_layers

            plot_clusterings(df_1=df_clustering_y_true, df_2=df_clustering_y_pred)

        # --------------- Activation across layers -------------
        if not os.path.exists(save_path + '/activation_across_layers/'):
            os.makedirs(save_path + '/activation_across_layers/')

        for name in name_layers:
            df_ = pd.DataFrame(
                data=np.hstack((dict_values_layers[name], (y_pred_MonoNet).reshape(len(y_pred_MonoNet), 1))),
                columns=[str('neuron_') + str(i + 1) for i in range(dict_values_layers[name].shape[1])] + [
                    'y_pred']).groupby(by=['y_pred'], as_index=False).mean()

            df_ = df_.transpose()
            df_.columns = list(meta_df.iloc[classes_pred, 1])  # [str('cell_type_') + str(i + 1) for i in classes_pred]
            df_ = df_[1:]
            df_['neurons'] = list(df_.index)
            dict_values_layers_mean[name] = df_

            ax = dict_values_layers_mean[name].set_index('neurons').plot(kind='bar', stacked=True, title=name)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(save_path + f'/activation_across_layers/activation_{name}.eps', format='eps', dpi=300,
                        bbox_inches='tight')
            plt.close()

        df_average_abs_activation = create_df_activation(dict_layers_mean=dict_values_layers_mean, abs=True)
        df_average_activation = create_df_activation(dict_layers_mean=dict_values_layers_mean, abs=False)

        plot_activation_mean(df_average=df_average_activation, abs=False)
        plot_activation_mean(df_average=df_average_abs_activation, abs=True, bool_legend=True)

        entropy_measures = ['trace', 'det', 'eigen_max', 'eigen_min']

        df_measure_entropy_trace = create_df_meas(meas_='trace')
        df_measure_entropy_det = create_df_meas(meas_='det')
        df_measure_entropy_eigen_max = create_df_meas(meas_='eigen_max')
        df_measure_entropy_eigen_min = create_df_meas(meas_='eigen_min')

        plot_entropy(df_=df_measure_entropy_trace, meas_='trace')
        plot_entropy(df_=df_measure_entropy_det, meas_='det')
        plot_entropy(df_=df_measure_entropy_eigen_max, meas_='eigen_max')
        plot_entropy(df_=df_measure_entropy_eigen_min, meas_='eigen_min')

    '''    measure_entropy_trace = []
        for name in name_layers:
            index = pd.Index(y_pred_MonoNet, name='label')
            df_ = pd.DataFrame(dict_values_layers[name], index=index)
            groups = {k: v.values for k, v in df_.groupby('label')}
            measure_variability = []
            for label, values in groups.items():
                trace = np.trace(np.cov(np.transpose(values)))
                measure_variability += [trace]
    
            df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])

        df_.index = list(meta_df.iloc[classes_pred, 1])
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)

        measure_entropy_trace.append(df_)
    df_measure_entropy_trace = pd.concat(measure_entropy_trace, ignore_index=True)

    plt.figure(figsize=(9.6, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_measure_entropy_trace, ci=None)

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('')
    plt.ylabel('Trace of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/activation_across_layers/trace_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    measure_entropy_det = []
    for name in name_layers:
        index = pd.Index(y_pred_MonoNet, name='label')
        df_ = pd.DataFrame(dict_values_layers[name], index=index)
        groups = {k: v.values for k, v in df_.groupby('label')}
        measure_variability = []

        for label, values in groups.items():
            det = np.linalg.det(np.cov(np.transpose(values)))
            measure_variability += [det]

        df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])
        df_.index = list(meta_df.iloc[classes_pred, 1])  # [str('cell_type_') + str(i + 1) for i in classes_pred]
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_det.append(df_)

    df_measure_entropy_det = pd.concat(measure_entropy_det, ignore_index=True)

    plt.figure(figsize=(9, 5))
    ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                       data=df_measure_entropy_det, ci=None)

    ax.set(xlabel='predicted cell type', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Determinant of covariance matrix')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/activation_across_layers/det_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    measure_entropy_eigen_max = []
    for name in name_layers:
        index = pd.Index(y_pred_MonoNet, name='label')
        df_ = pd.DataFrame(dict_values_layers[name], index=index)
        groups = {k: v.values for k, v in df_.groupby('label')}
        measure_variability = []

        for label, values in groups.items():
            max_eig = np.max(np.linalg.eig(np.cov(np.transpose(values)))[0])
            measure_variability += [max_eig]

        df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])
        df_.index = list(meta_df.iloc[classes_pred, 1])
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_eigen_max.append(df_)
    df_measure_entropy_eigen_max = pd.concat(measure_entropy_eigen_max, ignore_index=True)

    plt.figure(figsize=(7.5, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_measure_entropy_eigen_max, ci=None, legend=False)
    plt.legend([], [], frameon=False)
    # plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('')
    plt.ylabel('Maximum eigenvalue \n of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()

    plt.savefig(save_path + '/activation_across_layers/max_eigen_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    measure_entropy_eigen_min = []
    for name in name_layers:
        index = pd.Index(y_pred_MonoNet, name='label')
        df_ = pd.DataFrame(dict_values_layers[name], index=index)
        groups = {k: v.values for k, v in df_.groupby('label')}
        measure_variability = []

        for label, values in groups.items():
            min_eig = np.min(np.linalg.eig(np.cov(np.transpose(values)))[0])
            measure_variability += [min_eig]

        df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])
        df_.index = list(meta_df.iloc[classes_pred, 1])
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_eigen_min.append(df_)
    df_measure_entropy_eigen_min = pd.concat(measure_entropy_eigen_min, ignore_index=True)

    plt.figure(figsize=(9, 5))
    ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                       data=df_measure_entropy_eigen_min, ci=None)

    ax.set(xlabel='predicted cell type', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Min eigenvalue of covariance matrix')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/activation_across_layers/min_eigen_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()'''
