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
    for i in range(NUM_CLASSES):
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
    for i in range(h_values_.shape[1]):
        stat_ks, pvalues_ks = analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='KS_stat',
                                                 X_input=X_input_, y_input=y_input_)
        df_stats_KS_.loc[i] = stat_ks
        df_pvalues_KS_.loc[i] = pvalues_ks
    df_stats_KS_['neuron'] = np.arange(h_values_.shape[1]) + 1
    df_pvalues_KS_['neuron'] = np.arange(h_values_.shape[1]) + 1
    return df_stats_KS_, df_pvalues_KS_


def violin_plots(h_values_, X_input_, y_input_):
    for i in range(NUM_CLASSES):
        analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='violin_plot',
                           X_input=X_input_, y_input=y_input_)


def visualize_importances_all_neurons(feature_names, importances, title="Average Feature Importances",
                                      method="captum",
                                      axis_title="Biomarkers"):
    if not os.path.exists(save_path + '/feature_attributions/'):
        os.makedirs(save_path + '/feature_attributions/')

    if torch.is_tensor(importances[list(importances.keys())[0]]):
        importances = {k: v.detach().numpy() for k, v in importances.items()}
        print('change it to numpy')
    x_pos = (np.arange(len(feature_names)))

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

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
    axs[0, 4].bar(x_pos, importances['neuron_5'].mean(axis=0))
    axs[0, 4].set_title('Neuron 5')
    axs[1, 0].bar(x_pos, importances['neuron_6'].mean(axis=0))
    axs[1, 0].set_title('Neuron 6')
    axs[1, 1].bar(x_pos, importances['neuron_7'].mean(axis=0))
    axs[1, 1].set_title('Neuron 7')
    axs[1, 2].bar(x_pos, importances['neuron_8'].mean(axis=0))
    axs[1, 2].set_title('Neuron 8')
    axs[1, 3].bar(x_pos, importances['neuron_9'].mean(axis=0))
    axs[1, 3].set_title('Neuron 9')
    axs[1, 4].bar(x_pos, importances['neuron_10'].mean(axis=0))
    axs[1, 4].set_title('Neuron 10')

    for ax in axs.flat:
        ax.set(xlabel=axis_title)
        plt.sca(ax)
        plt.xticks(rotation=45)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path + f'/feature_attributions/{ method.replace(" ", "_")}_1', format='eps', dpi=300,
                bbox_inches='tight')

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    fig.suptitle(title + method)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, importances['neuron_11'].mean(axis=0))
    axs[0, 0].set_title('Neuron 11')
    axs[0, 1].bar(x_pos, importances['neuron_12'].mean(axis=0))
    axs[0, 1].set_title('Neuron 12')
    axs[0, 2].bar(x_pos, importances['neuron_13'].mean(axis=0))
    axs[0, 2].set_title('Neuron 13')
    axs[0, 3].bar(x_pos, importances['neuron_14'].mean(axis=0))
    axs[0, 3].set_title('Neuron 14')
    axs[0, 4].bar(x_pos, importances['neuron_15'].mean(axis=0))
    axs[0, 4].set_title('Neuron 15')
    axs[1, 0].bar(x_pos, importances['neuron_16'].mean(axis=0))
    axs[1, 0].set_title('Neuron 16')
    axs[1, 1].bar(x_pos, importances['neuron_17'].mean(axis=0))
    axs[1, 1].set_title('Neuron 17')
    axs[1, 2].bar(x_pos, importances['neuron_18'].mean(axis=0))
    axs[1, 2].set_title('Neuron 18')
    axs[1, 3].bar(x_pos, importances['neuron_19'].mean(axis=0))
    axs[1, 3].set_title('Neuron 19')
    axs[1, 4].bar(x_pos, importances['neuron_20'].mean(axis=0))
    axs[1, 4].set_title('Neuron 20')

    for ax in axs.flat:
        ax.set(xlabel=axis_title)
        plt.sca(ax)
        plt.xticks(rotation=45)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path + f'/feature_attributions/{ method.replace(" ", "_")}_2', format='eps', dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':
    short_name = 'mor'

    parser = argparse.ArgumentParser(prog='MonoNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', type=bool, default=True, help='Specify if one should load or train a MonoNet')
    parser.add_argument('--in_file_NN', type=str,
                        default=f'/Users/{short_name}/Documents/GitHub/MonoNet/mlp_26_04_saved.pickle')
    # --in_file_NN
    # /Users/mor/Documents/GitHub/MonoNet_single_cell/nn_saved_04_11_50_epochs.pickle
    parser.add_argument('--out_file_NN', type=str,
                        default=f'/Users/{short_name}/Documents/GitHub/MonoNet/mlp_26_04_saved.pickle')

    parser.add_argument('--verbose', type=int, default=0)

    parser.add_argument('--analysis_captum_methods', type=bool, default=False)
    parser.add_argument('--analysis_with_violin_plots', type=bool, default=False)
    parser.add_argument('--compute_shap_values', type=bool, default=True)

    args = parser.parse_args()
    load_bool = args.load
    compute_shap_values = args.compute_shap_values
    out_file_nn = args.out_file_NN
    in_file_nn = args.in_file_NN
    verbose = args.verbose

    save_path = str(f'/Users/{short_name}/Desktop/Plots_MLP/') + in_file_nn.split('/')[-1].split('.')[0]
    if not os.path.exists(save_path + '/'):
        os.makedirs(save_path + '/')

    captum = args.analysis_captum_methods
    plot_violin = args.analysis_with_violin_plots

    # Loading data
    meta_df = pd.read_csv(f'/Users/{short_name}/Desktop/single_cell_data/metadata.csv')
    df = pd.read_csv(f'/Users/{short_name}/Desktop/single_cell_data/data.csv')

    print(df.head)
    print(meta_df)

    # Plotting class distribution of the whole dataset
    if verbose > 0:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1.7, 1]})

        sns.countplot(x='category', data=df, ax=axs[0])
        axs[0].set(xlabel='Cell type', ylabel='Number of samples', title="Class distribution")
        axs[1].axis('tight')
        axs[1].axis('off')
        table = axs[1].table(cellText=meta_df.values.tolist(), loc='center', colWidths=[0.2, 0.8])
        table.set_fontsize(9)
        plt.savefig(save_path + '/class_distribution.eps', format='eps', dpi=300,
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

    ####################################################################################################################
    # --------------------------------------- TRAIN NEURAL NETWORK ---------------------------------------
    ####################################################################################################################

    # Hyperparameters for the Neural network
    EPOCHS = 5  # 80
    BATCH_SIZE = 35  # 40
    LEARNING_RATE = 0.00045  # 0.00045 0.00055
    NUM_FEATURES = len(X.columns)
    NUM_CLASSES = len(np.unique(y_train))
    NB_NEURON_INTER_LAYER = 8
    init_mode = 'xavier_normal'
    # avec l'ancienne version 13 128 64 8 75 epochs 50 0.0002

    model = torch.load(in_file_nn)

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
    output_values = model.forward(torch_X_subset)

    # Combine X_input and h values in one dataframe
    df_input_output = pd.DataFrame(
        data=np.hstack((X_subset, output_values.detach().numpy(), y_subset.reshape(len(y_subset), 1))),
        columns=list(df.columns[:-1]) + [str('Neuron_') + str(i + 1) for i in
                                         range(output_values.detach().numpy().shape[1])]
                                      + ['category'])

    df_input_output.to_csv('csv_input_h.csv')

    # Randomly select some of the sample (to reduce computational time)

    idx_by_class_reduced = {}
    for i in range(len(np.unique(y_subset))):
        temp_idx = np.where(y_subset == i)[0]
        np.random.shuffle(temp_idx)
        idx_by_class_reduced[str('class_') + str(i)] = list(temp_idx)[0:min(len(temp_idx), 50)]

    idx_all_class_redu = [item for sublist in list(idx_by_class_reduced.values()) for item in sublist]

    X_reduced = X_subset[idx_all_class_redu]
    y_reduced = y_subset[idx_all_class_redu]
    output_values_reduced = model.forward(X_reduced)
    y_pred_reduced = model(X_reduced)
    y_pred_reduced_softmax = torch.log_softmax(y_pred_reduced, dim=1)
    _, y_pred_reduced_tags = torch.max(y_pred_reduced_softmax, dim=1)

    y_pred = model(X_subset)
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    #
    name_layers = ['input', 'linear_1', 'linear_2', 'output']
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
        violin_plots(h_values_=output_values, X_input_=X_subset, y_input_=y_subset)
        plots_distance_measure(df_measure=dict_df_stats_KS['output'],
                               p_values=dict_df_pvalues_KS['output'], measure='KS')

        # Is there a difference between the top and bottom distributions? (H_0 = the distributions are the same so
        # if p<0.05 we reject the hypothesis that the distributions are the same)

        df_diff_in_distributions_KS = dict_df_pvalues_KS['output'] < 0.05
        df_diff_in_distributions_KS.index = list(range(1, NUM_CLASSES+1, 1))

    # ------------ Feature attribution methods ------------

    if compute_shap_values:
        exp_shap = {}
        X_train_summary = shap.kmeans(X_train, 50)
        for i in range(NUM_CLASSES):
            explainer_shap = shap.KernelExplainer(model=eval(f"model.output_{i}"),
                                                  data=X_train_summary)
            # TODO: Which dataset should we use?, what should we put there as reference ?
            exp_shap[str('neuron_') + str(i + 1)] = \
                explainer_shap.shap_values(shap.sample(X_test, 100))[
                    0]  # TODO should we take idx_all_class_redu
            # shap.force_plot(explainer_shap.expected_value[0], exp_shap[str('neuron_') + str(i + 1)], X_test)

            # shap.summary_plot(shap_values[0], X_test[0:50])
        visualize_importances_all_neurons(list(df.columns)[:-1], exp_shap,
                                          title="Average Feature Importances for ",
                                          method="Shapley Values approximation")
