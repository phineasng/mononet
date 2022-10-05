import os
import collections
import argparse
import shap
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, Birch
from sklearn.metrics.cluster import adjusted_mutual_info_score

from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
# from MonoNet_class import *
from MonoNet_class import *
from MonoNet_train import MonoNet_train

from generation_interventional_data import generate_interv
from captum.attr import NeuronGradientShap, NeuronIntegratedGradients, NeuronConductance


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def ECDF(x):
    n = np.sum(x)
    counts = []

    for i in range(len(x)):
        counts += [counts[i - 1] + x[i] if len(counts) > 0 else x[0]]

    return counts / n


def KS_discrete_fct(sample_1, sample_2):  # TODO: look again at that not sure about the calculation
    nb_obs = len(sample_1)  # + len(sample_2)
    max_value = np.max((np.max(sample_1), np.max(sample_2)))
    min_value = np.min((np.min(sample_1), np.min(sample_2)))

    nb_bins = nb_obs / 10  # TODO: need to change the 10 when more sample (maybe put 50?)
    count_1, _ = np.histogram(sample_1, bins=round(nb_bins), range=(min_value, max_value), density=False)
    count_2, _ = np.histogram(sample_2, bins=round(nb_bins), range=(min_value, max_value), density=False)

    diff_ecdf = np.abs(ECDF(count_1) - ECDF(count_2))  # TODO: leave np.abs() ??

    return np.max(diff_ecdf)


def M_seg_fct(sample_1, sample_2):
    nb_obs = len(sample_1)  # + len(sample_2)
    max_value = np.max((np.max(sample_1), np.max(sample_2)))
    min_value = np.min((np.min(sample_1), np.min(sample_2)))

    nb_bins = round(nb_obs / 50)  # TODO: need to change the 10 when more sample (maybe put 50?)
    count_1, bin_edge_1 = np.histogram(sample_1, bins=nb_bins, range=(min_value, max_value), density=False)
    count_2, bin_edge_2 = np.histogram(sample_2, bins=nb_bins, range=(min_value, max_value), density=False)
    bin_width = bin_edge_1[1] - bin_edge_1[0]
    bin_width_2 = (max_value - min_value) / nb_bins
    ecdf_1 = ECDF(count_1)
    ecdf_2 = ECDF(count_2)
    mult_counts = ecdf_1 * ecdf_2
    m_seg = -np.sum(mult_counts) * bin_width

    return m_seg


def analysis_by_neuron(h_values, idx_neuron, task, X_input, y_input, perc=0.2):
    i = idx_neuron

    #  h_i = torch.abs(h_values[:, i])  # TODO: put abs or not ????
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

    if task == 'Discrete_KS':
        KS_discrete_stats = []
        KS_discrete_pvalues = []

        for feat in list(df.columns[:-1]):
            x_bottom = df_top_bottom[feat][0:nb_obs].to_numpy(dtype='float')
            x_top = df_top_bottom[feat][-nb_obs:].to_numpy(dtype='float')
            ks_stat = KS_discrete_fct(x_bottom, x_top)
            KS_discrete_stats += [ks_stat]

            ###########################################################################
            x_all = list(df_top_bottom[feat].to_numpy(dtype='float'))
            emp_ks = []
            for j in range(1000):
                sample_1 = random.sample(x_all, round(len(x_all) / 2))

                sample_2 = []
                for x_all_i, sample_1_i in zip(x_all, sample_1):
                    sample_2.append(x_all_i - sample_1_i)

                emp_ks += [KS_discrete_fct(sample_1, sample_2)]
            p_val = np.sum(np.array(emp_ks) > ks_stat) / len(emp_ks)
            KS_discrete_pvalues += [p_val]
        ###########################################################################

        return KS_discrete_stats, KS_discrete_pvalues

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

    if task == 'M_segregation':  # TODO: VERIFY !!!!!
        M_seg_stats = []
        p_values_seg = []
        for feat in list(df.columns[:-1]):
            x_bottom = df_top_bottom[feat][0:nb_obs].to_numpy(dtype='float')
            x_top = df_top_bottom[feat][-nb_obs:].to_numpy(dtype='float')
            m_seg = M_seg_fct(x_bottom, x_top)
            M_seg_stats += [m_seg]

            ###########################################################################
            x_all = list(df_top_bottom[feat].to_numpy(dtype='float'))
            emp_m_seg = []
            for j in range(1000):
                sample_1 = random.sample(x_all, round(len(x_all) / 2))

                sample_2 = []
                for x_all_i, sample_1_i in zip(x_all, sample_1):
                    sample_2.append(x_all_i - sample_1_i)

                emp_m_seg += [M_seg_fct(sample_1, sample_2)]
            p_val = np.sum(np.array(emp_m_seg) > m_seg) / len(emp_m_seg)
            p_values_seg += [p_val]
        ###########################################################################

        return M_seg_stats, p_values_seg

    if task == 'violin_plot':
        if not os.path.exists(save_path + '/Violin_plots/'):
            os.makedirs(save_path + '/Violin_plots/')
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
        #plt.title('Interpretable Neuron {}'.format(i + 1), fontsize=16)
        plt.tick_params(axis='y', which='major', labelsize=12)
        plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
        plt.savefig(save_path + '/Violin_plots/neuron_{}.eps'.format(i + 1), format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        '''plt.figure(figsize=(8, 4.8))
        ax = sns.violinplot(x="variable", y="value", hue="label",
                            data=df_top_bottom_melt, palette="Set2", split=True,
                            scale="count")
        ax.set(xlabel='Biomarker', ylabel='Biomarker value', title="Neuron {}".format(i + 1))
        plt.savefig(save_path + '/Violin_plots/neuron_{}.eps'.format(i + 1), format='eps', dpi=300, bbox_inches='tight')
        plt.show()'''


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


def compute_discrete_ks(h_values_, X_input_, y_input_):
    df_ks_discrete_stats_ = pd.DataFrame(columns=list(df.columns[:-1]))
    df_ks_discrete_pvalues_ = pd.DataFrame(columns=list(df.columns[:-1]))
    for i in range(NB_NEURON_INTER_LAYER):
        ks_discrete_stat_i, ks_discrete_pvalue_i = analysis_by_neuron(h_values=h_values_, idx_neuron=i,
                                                                      task='Discrete_KS', X_input=X_input_,
                                                                      y_input=y_input_)
        df_ks_discrete_stats_.loc[i] = ks_discrete_stat_i
        df_ks_discrete_pvalues_.loc[i] = ks_discrete_pvalue_i
    df_ks_discrete_stats_['neuron'] = np.arange(8) + 1
    df_ks_discrete_pvalues_['neuron'] = np.arange(8) + 1
    return df_ks_discrete_stats_, df_ks_discrete_pvalues_


def compute_m_seg(h_values_, X_input_, y_input_):
    df_M_seg_ = pd.DataFrame(columns=list(df.columns[:-1]))
    df_M_seg_pvalues_ = pd.DataFrame(columns=list(df.columns[:-1]))
    for i in range(NB_NEURON_INTER_LAYER):
        M_seg, pvalues_m_seg = analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='M_segregation',
                                                  X_input=X_input_, y_input=y_input_)
        df_M_seg_.loc[i] = M_seg
        df_M_seg_pvalues_.loc[i] = pvalues_m_seg
    df_M_seg_['neuron'] = np.arange(8) + 1

    return df_M_seg_, df_M_seg_pvalues_


def violin_plots(h_values_, X_input_, y_input_):
    for i in range(NB_NEURON_INTER_LAYER):
        analysis_by_neuron(h_values=h_values_, idx_neuron=i, task='violin_plot',
                           X_input=X_input_, y_input=y_input_)


def plots_distance_measure(df_measure, measure, p_values):


    plt.close()
    d = pd.melt(df_measure, id_vars=['neuron'], ignore_index=False)

    plt.figure(figsize=(6, 5))
    sns.pointplot(x="variable", y="value", data=d[d['neuron'] == 4], ci=None, join=False, palette=sns.color_palette())
    plt.xlabel('', fontsize=16)
    plt.ylabel('KS score', multialignment='center', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/KS_scores_neuron4.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.pointplot(x="variable", y="value", data=d, join=False, palette=sns.color_palette())
    plt.xlabel('', fontsize=16)
    plt.ylabel('Average KS score', multialignment='center', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/KS_scores_average.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    '''plt.figure(figsize=(6.74, 4.5))
    sns.stripplot(x="neuron", y="value", hue="variable", data=d, jitter=0.1, color=list(sns.color_palette('Paired')) + [
        sns.color_palette("RdGy", 10)[7]])
    plt.legend(title='Biomarkers', fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Interpretable Neurons', fontsize=16)
    plt.ylabel('KS score', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path + '/KS_scores_neurons.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()'''
    palette_13 = sns.color_palette(list(sns.color_palette('tab10'))+[sns.color_palette('Set1')[5]]+[sns.color_palette('Set1')[8]]+[sns.color_palette('Accent')[7]])
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

    plt.savefig(save_path + '/KS_scores_average_&_neuron.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    d_ = pd.melt(p_values, id_vars=['neuron'], ignore_index=False)
    ax = sns.stripplot(x="neuron", y="value", hue="variable", data=d_, jitter=0.1, palette=sns.color_palette("Paired"))
    ax.set(xlabel='Neuron', ylabel='{} p-values'.format(measure))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.close()


def high_level_means(h_, idx_neuron_, y_input_, type_1_, type_2_):
    h_i = torch.abs(h_values[:, idx_neuron_])

    idx_type_1 = np.where(y_input_ == type_1_)[0]
    idx_type_2 = np.where(y_input_ == type_2_)[0]
    mean_type_1_ = h_i[idx_type_1].mean().item()
    mean_type_2_ = h_i[idx_type_2].mean().item()

    return mean_type_1_, mean_type_2_


def comp_high_level_profiles(h, y_input, type_1, type_2):
    mean_type_1_by_neuron = []
    mean_type_2_by_neuron = []

    for i in range(NB_NEURON_INTER_LAYER):
        mean_type_1, mean_type_2 = high_level_means(h_=h, idx_neuron_=i, y_input_=y_input, type_1_=type_1,
                                                    type_2_=type_2)
        mean_type_1_by_neuron += [mean_type_1]
        mean_type_2_by_neuron += [mean_type_2]

    df_mean_type = pd.DataFrame(np.array([mean_type_1_by_neuron, mean_type_2_by_neuron]),
                                columns=['neuron_1', 'neuron_2', 'neuron_3', 'neuron_4', 'neuron_5',
                                         'neuron_6', 'neuron_7', 'neuron_8']).transpose()
    df_mean_type.columns = [meta_df.loc[type_1][1], meta_df.loc[type_2][1]]
    df_mean_type['Neuron'] = np.arange(8) + 1
    df_mean_type_melt = pd.melt(df_mean_type, id_vars=['Neuron'], ignore_index=False)

    plt.figure()
    ax = sns.pointplot(x="Neuron", y="value", hue="variable", data=df_mean_type_melt)
    ax.set(xlabel='Neuron', ylabel='Hidden Layer mean',
           title='Class: {} vs {}'.format(meta_df.loc[type_1][1], meta_df.loc[type_2][1]))

    plt.savefig(save_path + '/plots_high_level_profiles_{}_VS_{}.eps'.format(meta_df.loc[type_1][1],
                                                                                             meta_df.loc[type_2][1]), format='eps', dpi=300, bbox_inches='tight')
    plt.show()


def t_SNE_plots_mean(df_input_, df_h_values_):
    df_input_reduced = df_input_.sample(500)
    X_reduced = df_input_reduced.iloc[:, 0:-1]
    y_reduced = df_input_reduced.iloc[:, -1]
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X_reduced)

    new_df_cat = pd.DataFrame({"category": list(y_reduced)})
    df_X_2d = pd.DataFrame(X_2d, columns=['dim_1', 'dim_2'])
    df_X_2d['category'] = new_df_cat['category'].astype('category')
    df_X_2d_mean = df_X_2d.groupby(by=['category'], as_index=False).mean()

    plt.figure(figsize=(7, 7))
    ax = sns.scatterplot(x='dim_1', y='dim_2', hue='category', data=df_X_2d_mean, legend=False)

    for i in list(df_X_2d_mean['category']):
        ax.annotate(str(i), (df_X_2d_mean.loc[df_X_2d_mean['category'] == i]['dim_1'].values + 0.01,
                             df_X_2d_mean.loc[df_X_2d_mean['category'] == i]['dim_2'].values + 0.01))
    plt.savefig(save_path + '/t_SNE_input.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()

    df_h_reduced = df_h_values_.sample(500)
    h_reduced = df_h_reduced.iloc[:, 0:-1]
    h_class_reduced = df_h_reduced.iloc[:, -1]
    tsne = TSNE(n_components=2, random_state=0)
    h_2d = tsne.fit_transform(h_reduced)

    df_h_2d = pd.DataFrame(data=h_2d, columns=['dim_1', 'dim_2'])
    df_h_2d['category'] = new_df_cat['category'].astype('category')
    df_h_2d_mean = df_h_2d.groupby(by=['category'], as_index=False).mean()

    plt.figure(figsize=(7, 7))
    ax = sns.scatterplot(x='dim_1', y='dim_2', hue='category', data=df_h_2d_mean, legend=False)

    for i in list(df_h_2d_mean['category']):
        ax.annotate(str(i), (df_h_2d_mean.loc[df_h_2d_mean['category'] == i]['dim_1'].values + 0.01,
                             df_h_2d_mean.loc[df_h_2d_mean['category'] == i]['dim_2'].values + 0.01))
    plt.savefig(save_path + '/t_SNE_h.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()


def monotonic_f(h_and_class_, neuron_, target_class_, fixed_class_):
    if not os.path.exists(save_path + '/Monotonic_function/'):
        os.makedirs(save_path + '/Monotonic_function/')

    # x_lim_min = h_and_class_[:, neuron_].min()
    # x_lim_max = h_and_class_[:, neuron_].max()

    x_lim_min = -1.5
    x_lim_max = 1.5
    x_ = np.arange(-50, 50, 0.01)
    x_ = np.arange(-1.5, 1.5, 0.01)
    x_ = np.arange(-1, 1, 0.01)
    # Extract values of the interpretable neurons for the fixed chosen class
    h_fixed_class = h_and_class_[h_and_class_[:, -1] == fixed_class_-1, :]  # TODO: Do that for predicted class instead?
    # Calculate the mean for each interpretable neuron
    mean_class = np.mean(h_fixed_class, axis=0)
    h_mon = np.full([len(x_), h_values.size()[1]], None)
    for idx_n in range(NB_NEURON_INTER_LAYER):
        h_mon[:, idx_n] = np.repeat(mean_class[idx_n], len(x_))
    # Replace in the neuron of interest the vector x_
    h_mon[:, neuron_-1] = x_

    # Pass the h_values into the monotonic block
    y_ = model.monotonic_block(torch.from_numpy(h_mon.astype('float32')))[:, target_class_-1].detach().numpy()

    x_lim_min = np.min(np.where(np.abs(y_[:-1]-y_[1:]) > 0)[0])
    x_lim_max = np.max(np.where(np.abs(y_[:-1]-y_[1:]) > 0)[0])
    x_ = x_#[x_lim_min:x_lim_max]
    y_ = y_#[x_lim_min:x_lim_max]

    # Calculate the density for the chosen interpretable neuron

    density = gaussian_kde(h_fixed_class[:, neuron_-1])
    xs = np.linspace(-1.5, 1.5, 500)
    density.covariance_factor = lambda: .25
    density._compute_covariance()

    plt.figure(figsize=(7, 5))
    plt.plot(x_, y_)
    plt.xlabel(f'Neuron {neuron_}', fontsize=14)
    plt.ylabel(f'Activation value of the output neuron\ncorresponding to {meta_df.loc[target_class_ - 1][1]}', multialignment='center', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/Monotonic_function/NEW_mono_f_neuron{}_targetclass{}_fixedclass{}.eps'
                .format(neuron_, target_class_-1, fixed_class_-1), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(x_, y_, color="tab:blue")
    # set x-axis label
    ax.set_xlabel('Neuron {} range for {}'.format(neuron_, meta_df.loc[fixed_class_-1][1], fontsize=12))

    # set y-axis label
    ax.set_ylabel('Class {} ({})'.format(target_class_, meta_df.loc[target_class_-1][1]), color="tab:blue", fontsize=12)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(xs, density(xs), color="tab:orange")
    ax2.set_ylabel("Density", color="tab:orange", fontsize=12)
    plt.show()
    # save the plot as a file
    fig.savefig(save_path + '/Monotonic_function/mono_f_neuron{}_targetclass{}_fixedclass{}.eps'
                .format(neuron_, target_class_-1, fixed_class_-1), format='eps', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_importances_all_neurons(feature_names, importances, title="Average Feature Importances", method="captum",
                                      axis_title="Biomarkers"):
    if not os.path.exists(save_path + f'/feature_attributions/'):
        os.makedirs(save_path + '/feature_attributions/')

    if torch.is_tensor(importances[list(importances.keys())[0]]):
        importances = {k: v.detach().numpy() for k, v in importances.items()}
        print('change it to numpy')
    x_pos = (np.arange(len(feature_names)))

    merge_dict_one_array = []
    for i in range(8):
        merge_dict_one_array += [importances[f'neuron_{i+1}'].mean(axis=0)]
    df_importances = pd.DataFrame(data=np.abs(np.array(merge_dict_one_array)), columns=list(df.iloc[:,:-1].columns))
    df_importances['neuron'] = [i+1 for i in range(8)]

    d = pd.melt(df_importances, id_vars=['neuron'], ignore_index=False)
    if method == "Shapley Values approximation":
        plt.figure(figsize=(7, 5))
        sns.pointplot(x="variable", y="value", data=d, join=False, palette=sns.color_palette())
        plt.xlabel('', fontsize=16)
        plt.ylabel('Average Shapley Values\napproximation', multialignment='center', fontsize=16)
        plt.tick_params(axis='y', which='major', labelsize=12)
        plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + '/feature_attributions/summary_{}.eps'.format(method.replace(" ", "_")), format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(19, 9))
    # fig.suptitle(title + method)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, importances['neuron_1'].mean(axis=0))
    axs[0, 0].set_title('Neuron 1', fontsize=14)
    axs[0, 1].bar(x_pos, importances['neuron_2'].mean(axis=0))
    axs[0, 1].set_title('Neuron 2', fontsize=14)
    axs[0, 2].bar(x_pos, importances['neuron_3'].mean(axis=0))
    axs[0, 2].set_title('Neuron 3', fontsize=14)
    axs[0, 3].bar(x_pos, importances['neuron_4'].mean(axis=0))
    axs[0, 3].set_title('Neuron 4', fontsize=14)
    axs[1, 0].bar(x_pos, importances['neuron_5'].mean(axis=0))
    axs[1, 0].set_title('Neuron 5', fontsize=14)

    axs[1, 1].bar(x_pos, importances['neuron_6'].mean(axis=0))
    axs[1, 1].set_title('Neuron 6', fontsize=14)

    axs[1, 2].bar(x_pos, importances['neuron_7'].mean(axis=0))
    axs[1, 2].set_title('Neuron 7', fontsize=14)

    axs[1, 3].bar(x_pos, importances['neuron_8'].mean(axis=0))
    axs[1, 3].set_title('Neuron 8', fontsize=14)

    for ax in axs.flat:
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.tick_params(axis='x', which='major', labelsize=13, labelrotation=90)
        # ax.set(xlabel=axis_title)
        plt.sca(ax)
    #fig.supxlabel('Biomarker', fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path + '/feature_attributions/big_{}.eps'.format(method.replace(" ", "_")), format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(11.5, 6))
    # fig.suptitle(title + method)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, importances['neuron_1'].mean(axis=0))
    axs[0, 0].set_title('Neuron 1', fontsize=12)
    axs[0, 1].bar(x_pos, importances['neuron_2'].mean(axis=0))
    axs[0, 1].set_title('Neuron 2', fontsize=12)
    axs[0, 2].bar(x_pos, importances['neuron_3'].mean(axis=0))
    axs[0, 2].set_title('Neuron 3', fontsize=12)
    axs[0, 3].bar(x_pos, importances['neuron_4'].mean(axis=0))
    axs[0, 3].set_title('Neuron 4', fontsize=12)
    axs[1, 0].bar(x_pos, importances['neuron_5'].mean(axis=0))
    axs[1, 0].set_title('Neuron 5', fontsize=12)

    axs[1, 1].bar(x_pos, importances['neuron_6'].mean(axis=0))
    axs[1, 1].set_title('Neuron 6', fontsize=12)

    axs[1, 2].bar(x_pos, importances['neuron_7'].mean(axis=0))
    axs[1, 2].set_title('Neuron 7', fontsize=12)

    axs[1, 3].bar(x_pos, importances['neuron_8'].mean(axis=0))
    axs[1, 3].set_title('Neuron 8', fontsize=12)

    for ax in axs.flat:
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='x', which='major', labelsize=10, labelrotation=90)
        # ax.set(xlabel=axis_title)
        plt.sca(ax)
    #fig.supxlabel('Biomarker', fontsize=12)
    #fig.supylabel(method.replace(" ", "_"), fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path + '/feature_attributions/bigger_legend_{}.eps'.format(method.replace(" ", "_")), format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    fig, axs = plt.subplots(2, 4, figsize=(11.5, 6))
    # fig.suptitle(title + method)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, importances['neuron_1'].mean(axis=0))
    axs[0, 0].set_title('Neuron 1', fontsize=10)
    axs[0, 1].bar(x_pos, importances['neuron_2'].mean(axis=0))
    axs[0, 1].set_title('Neuron 2', fontsize=10)
    axs[0, 2].bar(x_pos, importances['neuron_3'].mean(axis=0))
    axs[0, 2].set_title('Neuron 3', fontsize=10)
    axs[0, 3].bar(x_pos, importances['neuron_4'].mean(axis=0))
    axs[0, 3].set_title('Neuron 4', fontsize=10)
    axs[1, 0].bar(x_pos, importances['neuron_5'].mean(axis=0))
    axs[1, 0].set_title('Neuron 5', fontsize=10)

    axs[1, 1].bar(x_pos, importances['neuron_6'].mean(axis=0))
    axs[1, 1].set_title('Neuron 6', fontsize=10)

    axs[1, 2].bar(x_pos, importances['neuron_7'].mean(axis=0))
    axs[1, 2].set_title('Neuron 7', fontsize=10)

    axs[1, 3].bar(x_pos, importances['neuron_8'].mean(axis=0))
    axs[1, 3].set_title('Neuron 8', fontsize=10)

    for ax in axs.flat:
        ax.tick_params(axis='y', which='major', labelsize=8)
        ax.tick_params(axis='x', which='major', labelsize=10, labelrotation=90)
        # ax.set(xlabel=axis_title)
        plt.sca(ax)
    #fig.supxlabel('Biomarker', fontsize=12)
    #fig.supylabel(method.replace(" ", "_"), fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path + '/feature_attributions/small_{}.eps'.format(method.replace(" ", "_")), format='eps', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_PC_IDA(feature_names, ida_df, title="IDA", axis_title="Biomarkers"):
    if not os.path.exists(save_path + f'/ida/'):
        os.makedirs(save_path + '/ida/')

    x_pos = (np.arange(len(feature_names)))

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))

    fig.suptitle(title)
    plt.setp(axs, xticks=x_pos, xticklabels=feature_names)
    axs[0, 0].bar(x_pos, np.array(R_PC_IDA.iloc[0]))
    axs[0, 0].set_title('Neuron 1')
    axs[0, 1].bar(x_pos,  np.array(R_PC_IDA.iloc[1]))
    axs[0, 1].set_title('Neuron 2')
    axs[0, 2].bar(x_pos,  np.array(R_PC_IDA.iloc[2]))
    axs[0, 2].set_title('Neuron 3')
    axs[0, 3].bar(x_pos,  np.array(R_PC_IDA.iloc[3]))
    axs[0, 3].set_title('Neuron 4')
    axs[1, 0].bar(x_pos,  np.array(R_PC_IDA.iloc[4]))
    axs[1, 0].set_title('Neuron 5')
    axs[1, 1].bar(x_pos,  np.array(R_PC_IDA.iloc[5]))
    axs[1, 1].set_title('Neuron 6')
    axs[1, 2].bar(x_pos,  np.array(R_PC_IDA.iloc[6]))
    axs[1, 2].set_title('Neuron 7')
    axs[1, 3].bar(x_pos,  np.array(R_PC_IDA.iloc[7]))
    axs[1, 3].set_title('Neuron 8')

    for ax in axs.flat:
        ax.set(xlabel=axis_title)
        plt.sca(ax)
        plt.xticks(rotation=45)
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path + '/ida/ida_PC.eps', format='eps', dpi=300, bbox_inches='tight')


def stacked_barplot_measure_distr(measure, opposite=False):
    measure_short = measure.split('_')[-1]
    df_barplot = eval(f"df_{measure}")

    adjacency_causality = pd.read_csv('KPC_adjacency.csv', header=0, index_col=0).iloc[13:21, 0:13]
    df_causal_struct = pd.DataFrame(data=np.array(df_barplot.iloc[:, :-1])*np.array(pd.read_csv('KPC_adjacency.csv', header=0, index_col=0).iloc[13:21, 0:13]),
                                    columns=list(df_barplot.columns)[0:13])
    df_causal_struct['neuron'] = [i+1 for i in range(8)]
    df_causal_struct = df_causal_struct.drop(columns=['CD123', 'CD90'])

    plt.figure(figsize=(7,5))
    x = ['1', '2', '3', '4', '5', '6', '7', '8']
    colors = list(sns.color_palette('Paired'))
    bot = np.zeros(8)
    for i in range(11):
        plt.bar(x, list(df_causal_struct.set_index('neuron').iloc[:, i]), color=colors[i], bottom=bot, width=0.5)
        bot = np.add(bot, list(df_causal_struct.set_index('neuron').iloc[:, i]))
    plt.xlabel('Interpretable neurons', fontsize=16)
    plt.ylabel(f'{measure_short} score', fontsize=16)
    plt.legend(list(df_causal_struct.columns)[:-1], fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path + f'/stacked_barplots/causal_{measure_short}_neurons_x.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    df_barplot.set_index('neuron')
    plt.figure(figsize=(7,5))
    x = ['1', '2', '3', '4', '5', '6', '7', '8']
    colors = list(sns.color_palette('Paired')) + [sns.color_palette("RdGy", 10)[7]]
    bot = np.zeros(8)
    for i in range(13):
        plt.bar(x, list(df_barplot.set_index('neuron').iloc[:, i]), color=colors[i], bottom=bot, width=0.5)
        bot = np.add(bot,list(df_barplot.set_index('neuron').iloc[:, i]))
    plt.xlabel('Interpretable neurons', fontsize=16)
    plt.ylabel(f'{measure_short} score', fontsize=16)
    plt.legend(list(df_barplot.columns)[:-1], fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path + f'/stacked_barplots/{measure_short}_neurons_x.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    df_barplot_t = df_barplot.transpose().iloc[0:13]
    df_barplot_t['biomarkers'] = list(df_barplot_t.index)

    plt.figure(figsize=(7, 5))
    x = list(df_barplot.columns)[:-1]
    colors = list(sns.color_palette('tab10'))[0:5]+[list(sns.color_palette('tab10'))[6]]+list(sns.color_palette('tab10'))[8:10]
    bot = np.zeros(13)
    for i in range(8):
        plt.bar(x, list(df_barplot_t.set_index('biomarkers').iloc[:, i]), color=colors[i], bottom=bot, width=0.5)
        bot = np.add(bot, list(df_barplot_t.set_index('biomarkers').iloc[:, i]))
    plt.xlabel('', fontsize=16)
    plt.ylabel(f'{measure_short} score', fontsize=16)
    plt.legend([f'Neuron {i+1}' for i in list(df_barplot_t.columns)[:-1]], fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path + f'/stacked_barplots/{measure_short}_biomarkers_x.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(7,5))
    df_barplot.set_index('neuron').plot(kind='bar', stacked=True, color=list(sns.color_palette('Paired')) + [
        sns.color_palette("RdGy", 10)[7]])
    # add overall title
    # add axis titles
    plt.xlabel('Interpretable neurons', fontsize=16)
    plt.ylabel(f'{measure_short} score', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, rotation=90)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path + f'/stacked_barplots/{measure_short}_neurons_x.eps', format='eps', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    df_barplot_t = df_barplot.transpose().iloc[0:13]
    df_barplot_t['biomarkers'] = list(df_barplot_t.index)

    plt.figure(figsize=(7,5))
    df_barplot_t.set_index('biomarkers').plot(kind='bar', stacked=True, color=list(sns.color_palette('Paired', 8)))
    # add axis titles
    plt.xlabel('')
    plt.ylabel(f'{measure_short} score', fontsize=16)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.4)
    # rotate x-axis labels
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, rotation=45)
    plt.savefig(save_path + f'/stacked_barplots/{measure_short}_biomarkers_x.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_training_progress(model_):
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(model_.loss_hist['val'], label="Validation")
    plt.plot(model_.loss_hist['train'], label="Training")
    plt.legend(fontsize=16)

    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path + '/loss_training.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(model_.accuracy_hist['val'], label="Validation")
    plt.plot(model_.accuracy_hist['train'], label="Training")
    plt.legend(fontsize=16)

    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path + '/accuracy_training.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

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


def tSNE_plots(values_layer_, name_layer):
    if not os.path.exists(save_path + '/t_SNE_perplex/'):
        os.makedirs(save_path + '/t_SNE_perplex/')

    perplexities = [20] #[10, 15, 20]
    for perplex in perplexities:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplex)
        values_2d = tsne.fit_transform(values_layer_)
        df_values_2d = pd.DataFrame(values_2d, columns=['dim_1', 'dim_2'])
        # df_values_2d['y_pred'] = pd.Series(y_pred_tags).astype('category')
        df_values_2d['y_pred'] = pd.Series(y_subset).astype('category')
        nb_color = len(np.unique(pd.Series(y_subset)))
        qualitative_colors_20 = sns.color_palette("Set3", nb_color)
        plt.figure(figsize=(10, 10))
        ax = sns.scatterplot(x='dim_1', y='dim_2', hue='y_pred', data=df_values_2d, palette=qualitative_colors_20)
        ax.set(title=f'tSNE plot for the values of the {name_layer}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(save_path + f'/t_SNE_perplex/t_SNE_{name}_perplex_{perplex}.eps', format='eps', dpi=300, bbox_inches='tight')
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

    parser.add_argument('--analysis_captum_methods', type=bool, default=True)
    parser.add_argument('--compute_shap_values', type=bool, default=False)
    parser.add_argument('--analysis_with_violin_plots', type=bool, default=False)
    parser.add_argument('--analysis_monotonic_fct_plots', type=bool, default=False)
    parser.add_argument('--analysis_tSNE_plots', type=bool, default=False)
    parser.add_argument('--analysis_per_layer', type=bool, default=False)
    parser.add_argument('--analysis_clustering', type=bool, default=False)

    ####################################################################################################################
    # --------------------------------------- PARSE ARGUMENTS ----------------------------------------------------------
    ####################################################################################################################
    args = parser.parse_args()
    out_file_nn = args.out_file_NN
    in_file_nn = args.in_file_NN
    verbose = args.verbose
    save_root = args.save_root
    captum = args.analysis_captum_methods
    compute_shap_values = args.compute_shap_values
    plot_violin = args.analysis_with_violin_plots
    mono_f = args.analysis_monotonic_fct_plots
    tsne_plots = args.analysis_tSNE_plots
    analysis_per_layer = args.analysis_per_layer
    clustering = args.analysis_clustering

    ####################################################################################################################
    # --------------------------------------- DATA PREPARATION AND VISUALIZATION ---------------------------------------
    ####################################################################################################################

    # Loading data
    meta_df = pd.read_csv(os.path.join(args.data_root, '../metadata.csv'))
    df = pd.read_csv(os.path.join(args.data_root, '../data.csv'))

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
                              train_dataset, val_dataset, test_dataset, class_weights, weighted_sampler, X_test, y_test,
                              out_file_nn=out_file_nn)
        saved_model = in_file_nn

    # create results subfolder
    fname = os.path.basename(saved_model)
    fname_no_ext = os.path.splitext(fname)[0]
    save_path = os.path.join(args.save_root, fname_no_ext)
    if not os.path.exists(save_path):
        os.makedirs(save_path)



#########
    print(model)
    matrix_sign = torch.matmul(model.layer_inter.weight.data.reshape((8,1)), torch.transpose(model.layer_inter_out.weight.data.reshape((20, 1)),0,1)).detach().numpy()
    inverse_elem_wise = lambda x: 1/x
    inv_matrix_sign = inverse_elem_wise(matrix_sign)
    data_fr = pd.DataFrame(data=inv_matrix_sign, columns=list(meta_df.iloc[:, 1]), index=[f'{i + 1}' for i in range(8)])

    plt.figure(figsize=(8, 5.5))
    ax = sns.heatmap(data_fr, square=True, cmap='coolwarm', cbar_kws={"shrink": .50})
    ax.set_ylabel('Interpretable neuron', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12, rotation=0)
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_path + '/sign_matrix_".eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.close()
########################################################################################################################

    generate_interv(data_=test_dataset, model_=model)

    plot_training_progress(model)
########################################################################################################################
    # --------------------------------------- INTERPRETABILITY ---------------------------------------
########################################################################################################################

    # Compute h values from the interpretable layer
    # TODO: What input values should I consider to compute h?? Is it correct how I compute them ?

    # Selection of the data we will use
    subset = 'test'  # TODO: Which subset should I use?
    X_subset = eval('X_{}'.format(subset))
    y_subset = eval('y_{}'.format(subset))

    torch_X_subset = torch.from_numpy(X_subset).clone().to(torch.float32)

    # Compute values of the neurons in the interpretable layer (h_values)
    h_values = model.unconstrainted_block(torch_X_subset)

    # Combine X_input and h values in one dataframe
    df_input_h = pd.DataFrame(data=np.hstack((X_subset, h_values.detach().numpy(), y_subset.reshape(len(y_subset), 1))),
                              columns=list(df.columns[:-1]) + list(np.arange(h_values.detach().numpy().shape[1]) + 1) +
                                      ['category'])

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
    h_values_reduced = model.unconstrainted_block(X_reduced)
    y_pred_reduced = model(X_reduced)
    y_pred_reduced_softmax = torch.log_softmax(y_pred_reduced, dim=1)
    _, y_pred_reduced_tags = torch.max(y_pred_reduced_softmax, dim=1)

    y_pred= model(X_subset)
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    # (y_pred_tags == torch.from_numpy(y_test)).float().sum() / len((y_pred_tags == torch.from_numpy(y_test)).float())
    # (y_pred_reduced_tags == torch.from_numpy(y_reduced)).float().sum() / len((y_pred_reduced_tags ==
    # torch.from_numpy(y_reduced)).float())
    # much lower accuracy probably some bias somewhere #TODO: look at that!
######################################################################################################################

    #name_layers = ['Input', 'Linear_1', 'Linear_2', 'Linear_3', 'Interpretable', 'Monotonic_1', 'Monotonic_2',
                 #  'Output']
    name_layers = ['Input', 'Linear_1', 'Linear_2', 'Interpretable', 'Pre-monotonic', 'Monotonic_1', 'Monotonic_2',
                   'Output']
    dict_values_layers = {}
    for name in name_layers:
        if name == "Input":
            dict_values_layers[name] = X_subset
        elif name == "Output":
            dict_values_layers[name] = model(X_subset).detach().numpy()
        elif name == "Pre-monotonic":
            dict_values_layers[name] = model.get_Premonotonic(X_subset).detach().numpy()
        else:
            fct = eval(f"model.get_{name}")
            dict_values_layers[name] = fct(X_subset).detach().numpy()
    dict_values_layers_mean = {}
    dict_values_layers_mean_positive = {}
    dict_values_layers_mean_abs = {}
    y_pred_MonoNet = y_pred_tags.detach().numpy().astype("int")

    classes_pred = list(np.unique(y_pred_MonoNet))

    measure_entropy_trace = []
    measure_entropy_trace_norm = []
    for name in name_layers:
        index = pd.Index(y_pred_MonoNet, name='label')
        df_ = pd.DataFrame(dict_values_layers[name], index=index)
        groups = {k: v.values for k, v in df_.groupby('label')}
        measure_variability = []
        for label, values in groups.items():
            trace = np.trace(np.cov(np.transpose(values)))
            measure_variability += [trace]

        df_ = pd.DataFrame(np.array(measure_variability), columns=['value'])
        df_norm = pd.DataFrame(MinMaxScaler().fit_transform(df_), columns=['value'])
        df_.index = list(meta_df.iloc[classes_pred,1])
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        df_norm.index = list(meta_df.iloc[classes_pred,1]) #[str('cell_type_') + str(i + 1) for i in classes_pred]
        df_norm['pred_cell_type'] = list(df_norm.index)
        df_norm['layer'] = [name] * len(df_norm)
        measure_entropy_trace.append(df_)
        measure_entropy_trace_norm.append(df_norm)
    df_measure_entropy_trace = pd.concat(measure_entropy_trace, ignore_index=True)
    if not os.path.exists(save_path + '/entropy_plots/'):
        os.makedirs(save_path + '/entropy_plots/')

    plt.figure(figsize=(9.6, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_measure_entropy_trace, ci=None)

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('')
    plt.ylabel('Trace of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/entropy_plots/trace_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    df_measure_entropy_trace_norm = pd.concat(measure_entropy_trace_norm, ignore_index=True)

    plt.figure(figsize=(9, 5))
    ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                       data=df_measure_entropy_trace_norm, ci=None)

    ax.set(xlabel='predicted cell type', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Trace of covariance matrix (normalized)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/entropy_plots/trace_covariance_norm.eps', format='eps', dpi=300,
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
        df_.index = list(meta_df.iloc[classes_pred,1])#[str('cell_type_') + str(i + 1) for i in classes_pred]
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_det.append(df_)

    df_measure_entropy_det = pd.concat(measure_entropy_det, ignore_index=True)
    if not os.path.exists(save_path + '/entropy_plots/'):
        os.makedirs(save_path + '/entropy_plots/')
    plt.figure(figsize=(9, 5))
    ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                       data=df_measure_entropy_det, ci=None)

    ax.set(xlabel='predicted cell type', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Determinant of covariance matrix')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/entropy_plots/det_covariance.eps', format='eps', dpi=300,
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
        df_.index = list(meta_df.iloc[classes_pred,1])# [str('cell_type_') + str(i + 1) for i in classes_pred]
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_eigen_max.append(df_)
    df_measure_entropy_eigen_max = pd.concat(measure_entropy_eigen_max, ignore_index=True)
    if not os.path.exists(save_path + '/entropy_plots/'):
        os.makedirs(save_path + '/entropy_plots/')

    plt.figure(figsize=(7.5, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_measure_entropy_eigen_max, ci=None, legend=False)
    plt.legend([], [], frameon=False)

    plt.xlabel('')
    plt.ylabel('Maximum eigenvalue \n of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()

    plt.savefig(save_path + '/entropy_plots/max_eigen_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    plt.figure(figsize=(9.6, 6))
    sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                  data=df_measure_entropy_eigen_max, ci=None, legend=False)
    # plt.legend([], [], frameon=False)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('')
    plt.ylabel('Maximum eigenvalue \n of the covariance matrix', fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
    plt.tight_layout()

    plt.savefig(save_path + '/entropy_plots/max_eigen_covariance_withlegend.eps', format='eps', dpi=300,
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
        df_.index =list(meta_df.iloc[classes_pred,1])# [str('cell_type_') + str(i + 1) for i in classes_pred]
        df_['pred_cell_type'] = list(df_.index)
        df_['layer'] = [name] * len(df_)
        measure_entropy_eigen_min.append(df_)
    df_measure_entropy_eigen_min = pd.concat(measure_entropy_eigen_min, ignore_index=True)
    if not os.path.exists(save_path + '/entropy_plots/'):
        os.makedirs(save_path + '/entropy_plots/')
    plt.figure(figsize=(9, 5))
    ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                       data=df_measure_entropy_eigen_min, ci=None)

    ax.set(xlabel='predicted cell type', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Min eigenvalue of covariance matrix')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(save_path + '/entropy_plots/min_eigen_covariance.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
######################################################################################################################
    if analysis_per_layer:
        if not os.path.exists(save_path + '/activation_per_layer/'):
            os.makedirs(save_path + '/activation_per_layer/')

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
            plt.savefig(save_path + f'/activation_per_layer/activation_{name}.eps', format='eps', dpi=300,
                        bbox_inches='tight')
            plt.close()

            dict_values_layers_mean_positive[name] = dict_values_layers_mean[name].copy()
            submat = dict_values_layers_mean_positive[name].iloc[:, :-1]
            submat[submat < 0] = np.nan
            dict_values_layers_mean_positive[name].iloc[:, :-1] = submat
            ax = dict_values_layers_mean_positive[name].set_index('neurons').plot(kind='bar', stacked=True, title=name)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(save_path + f'/activation_per_layer/pos_activation_{name}.eps', format='eps',
                        dpi=300,
                        bbox_inches='tight')
            plt.close()

        ###############################################################################################################
        average_abs_activation = []
        average_abs_activation_norm = []
        scaler_2 = MinMaxScaler()
        for name in name_layers:
            df_temp_abs = pd.DataFrame(dict_values_layers_mean[name].iloc[:, :-1].abs().mean(axis=0))
            X_ = np.array(dict_values_layers_mean[name].iloc[:, :-1].abs())
            X_one_column = X_.reshape([-1, 1])
            result_one_column = scaler_2.fit_transform(X_one_column)
            result = result_one_column.reshape(X_.shape)
            # normalized_abs_values = scaler_2.fit_transform(pd.DataFrame(dict_values_layers_mean[name].iloc[:, :-1].abs()))
            df_temp_abs_normalize = pd.DataFrame(
                pd.DataFrame(result).mean(axis=0))  # pd.DataFrame(pd.DataFrame(normalized_abs_values).mean(axis=0))
            df_temp_abs.columns = ['value']
            df_temp_abs['pred_cell_type'] = list(df_temp_abs.index)
            df_temp_abs['layer'] = [name] * len(df_temp_abs)
            df_temp_abs_normalize.columns = ['value']
            df_temp_abs_normalize['pred_cell_type'] = list(df_temp_abs.index)
            df_temp_abs_normalize['layer'] = [name] * len(df_temp_abs_normalize)
            average_abs_activation.append(df_temp_abs)
            average_abs_activation_norm.append(df_temp_abs_normalize)

        df_average_abs_activation = pd.concat(average_abs_activation, ignore_index=True)
        df_average_abs_activation_norm = pd.concat(average_abs_activation_norm, ignore_index=True)

        ###############################################################################################################
        average_activation = []
        average_activation_norm = []
        scaler_3 = MinMaxScaler()
        for name in name_layers:
            df_temp = pd.DataFrame(dict_values_layers_mean[name].iloc[:, :-1].mean(axis=0))
            X_ = np.array(dict_values_layers_mean[name].iloc[:, :-1])
            X_one_column = X_.reshape([-1, 1])
            result_one_column = scaler_3.fit_transform(X_one_column)
            result = result_one_column.reshape(X_.shape)
            # normalized_values = scaler_3.fit_transform(pd.DataFrame(dict_values_layers_mean[name].iloc[:, :-1]))
            df_temp_normalize = pd.DataFrame(pd.DataFrame(result).mean(axis=0))
            df_temp.columns = ['value']
            df_temp['pred_cell_type'] = list(df_temp.index)
            df_temp['layer'] = [name] * len(df_temp)
            df_temp_normalize.columns = ['value']
            df_temp_normalize['pred_cell_type'] = list(df_temp.index)
            df_temp_normalize['layer'] = [name] * len(df_temp_normalize)
            average_activation.append(df_temp)
            average_activation_norm.append(df_temp_normalize)

        df_average_activation = pd.concat(average_activation, ignore_index=True)
        df_average_activation_norm = pd.concat(average_activation_norm, ignore_index=True)

        ###############################################################################################################
        average_pos_activation = []
        average_pos_activation_norm = []
        scaler_4 = MinMaxScaler()
        for name in name_layers:
            df_temp_pos = pd.DataFrame(dict_values_layers_mean_positive[name].iloc[:, :-1].mean(axis=0))
            X_ = np.array(dict_values_layers_mean_positive[name].iloc[:, :-1])
            X_one_column = X_.reshape([-1, 1])
            result_one_column = scaler_4.fit_transform(X_one_column)
            result = result_one_column.reshape(X_.shape)
            # normalized_pos_values = scaler_4.fit_transform(pd.DataFrame(dict_values_layers_mean_positive[name].iloc[:, :-1]))
            df_temp_pos_normalize = pd.DataFrame(pd.DataFrame(result).mean(axis=0))
            df_temp_pos.columns = ['value']
            df_temp_pos['pred_cell_type'] = list(df_temp_pos.index)
            df_temp_pos['layer'] = [name] * len(df_temp_pos)
            df_temp_pos_normalize.columns = ['value']
            df_temp_pos_normalize['pred_cell_type'] = list(df_temp_pos.index)
            df_temp_pos_normalize['layer'] = [name] * len(df_temp_pos_normalize)
            average_pos_activation.append(df_temp_pos)
            average_pos_activation_norm.append(df_temp_pos_normalize)

        df_average_pos_activation = pd.concat(average_pos_activation, ignore_index=True)
        df_average_pos_activation_norm = pd.concat(average_pos_activation_norm, ignore_index=True)

        ###############################################################################################################
        plt.figure(figsize=(13, 9))
        ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                           data=df_average_abs_activation_norm, ci=None)

        ax.set(xlabel='predicted cell type', ylabel='Layers')
        plt.xticks(rotation=45)
        ax.set_title('Mean absolute activation across layers (normalized)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(save_path + '/plots_mean_abs_activation_across_layers_norm.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()
        ###############################################################################################################
        ###############################################################################################################

        plt.figure(figsize=(7.5, 6))
        sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                      data=df_average_activation, ci=None)
        plt.legend([], [], frameon=False)
        plt.ylim([-1, 1])

        plt.xlabel('', fontsize=16)
        plt.ylabel('Mean activation', fontsize=18)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + '/plots_mean_activation_across_layers.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(9.6, 6))
        sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                      data=df_average_activation, ci=None)

        plt.ylim([-1, 1])
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('', fontsize=16)
        plt.ylabel('Mean activation', fontsize=18)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + '/plots_mean_activation_across_layers_withlegend.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()
        plt.figure(figsize=(9.6, 6))
        sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                      data=df_average_abs_activation, ci=None)
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('')
        plt.ylim([0, 1])
        plt.ylabel('Mean absolute activation', fontsize=18)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
        plt.tight_layout()

        plt.savefig(save_path + '/plots_mean_abs_activation_across_layers.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

        ###############################################################################################################
        ###############################################################################################################
        plt.figure(figsize=(13, 9))
        ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                           data=df_average_activation_norm, ci=None)

        ax.set(xlabel='predicted cell type', ylabel='Layers')
        plt.xticks(rotation=45)
        ax.set_title('Mean activation across layers (normalized)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(save_path + '/plots_mean_activation_across_layers_norm.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

        ###############################################################################################################
        plt.figure(figsize=(13, 9))
        ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                           data=df_average_pos_activation_norm, ci=None)

        ax.set(xlabel='predicted cell type', ylabel='Layers')
        plt.xticks(rotation=45)
        ax.set_title('Mean activation (>0) across layers (normalized)')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(save_path + '/plots_mean_pos_activation_across_layers_norm.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(13, 9))
        ax = sns.pointplot(x="layer", y="value", hue="pred_cell_type",
                           data=df_average_pos_activation, ci=None)

        ax.set(xlabel='predicted cell type', ylabel='Layers')
        plt.xticks(rotation=45)
        ax.set_title('Mean activation (>0) across layers')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(save_path + '/plots_mean_pos_activation_across_layers.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()

        ###############################################################################################################
        # Just before applying softsign to last layer
        dict_values_layers_mean['last_layer_b'] = pd.DataFrame(
            data=np.hstack((model.last_layer(model.unconstrainted_block(X_subset)).detach().numpy(),
                            (y_pred_tags.detach().numpy().astype("int")).reshape(len(y_pred_tags), 1))),
            columns=[str('neuron_') + str(i + 1) for i in range(NUM_CLASSES)] + ['y_pred']).groupby(by=['y_pred'],
                                                                                                    as_index=False).mean()
        dict_values_layers_mean['last_layer_b'].set_index('y_pred').plot(kind='bar', stacked=True)
        dict_values_layers_mean['last_layer_b'][dict_values_layers_mean['last_layer_b'] < 0] = 0
        dict_values_layers_mean['last_layer_b'].set_index('y_pred').plot(kind='bar', stacked=True)
        ######################################################################################################################

        '''if not os.path.exists(save_path + '/boxplot_per_layer/'):
            os.makedirs(save_path + '/boxplot_per_layer/')
        for name in name_layers:
            df_ = pd.melt(pd.DataFrame(data=np.hstack((dict_values_layers[name], (y_pred_MonoNet).reshape(len(y_pred_MonoNet), 1))),
                                       columns=[str('neuron_') + str(i + 1) for i in range(dict_values_layers[name].shape[1])] + ['y_pred']), id_vars="y_pred")
            nb_neurons = dict_values_layers[name].shape[1]
            fig, axs = plt.subplots(2, int(nb_neurons/2), figsize=(12, 7))
            fig.suptitle(f'Distribution pattern for {name}')
            for i in range(int(nb_neurons/2)):
                sns.boxplot(x="variable", y="value", hue="y_pred", data=df_[df_['variable'] == 'neuron_'+str(i+1)],
                            ax=axs[0, i], fliersize=1)
                axs[0, i].legend_.remove()
                sns.boxplot(x="variable", y="value", hue="y_pred",
                            data=df_[df_['variable'] == 'neuron_' + str(int(nb_neurons / 2) + i)], ax=axs[1, i], fliersize=2)

                axs[1, i].legend_.remove()
            for ax in axs.flat:
                ax.set(ylabel='activation')
            for ax in fig.get_axes():
                ax.label_outer()

            fig.tight_layout()
            plt.savefig(save_path + f'/boxplot_per_layer/boxplots_{name}.eps', format='eps', dpi=300,
                        bbox_inches='tight')'''
    if clustering:
        nb_cluters = 20
        kmeans = KMeans(init="random", n_clusters=nb_cluters, n_init=10, max_iter=300,  random_state=42)
        kmeans_fits = {}
        kmeans_adjusted_mutual_info_scores_y_true = {}
        kmeans_adjusted_mutual_info_scores_y_pred = {}
        for key, values_layer in dict_values_layers.items():
            kmeans_fits[key] = kmeans.fit(values_layer)
            kmeans_adjusted_mutual_info_scores_y_true[key] = adjusted_mutual_info_score(kmeans_fits[key].labels_, y_subset)
            kmeans_adjusted_mutual_info_scores_y_pred[key] = adjusted_mutual_info_score(kmeans_fits[key].labels_,
                                                                                 y_pred_MonoNet)
        birch = Birch(threshold=0.01, n_clusters=nb_cluters)
        birch_fits = {}
        birch_adjusted_mutual_info_scores_y_true = {}
        birch_adjusted_mutual_info_scores_y_pred = {}
        for key, values_layer in dict_values_layers.items():
            birch_fits[key] = birch.fit(values_layer)
            birch_adjusted_mutual_info_scores_y_pred[key] = adjusted_mutual_info_score(birch.predict(values_layer), y_pred_MonoNet)
            birch_adjusted_mutual_info_scores_y_true[key] = adjusted_mutual_info_score(birch.predict(values_layer), y_subset)

        df_clustering_y_true = pd.DataFrame(data=np.transpose(np.vstack((np.array(list(kmeans_adjusted_mutual_info_scores_y_true.values())),
                                                                  np.array(list(birch_adjusted_mutual_info_scores_y_true.values()))))),
                                     columns=['K means - true class', 'BIRCH - true class'])
        df_clustering_y_true.index = name_layers
        df_clustering_y_pred = pd.DataFrame(data=np.transpose(np.vstack((np.array(list(kmeans_adjusted_mutual_info_scores_y_pred.values())),
                                                                  np.array(list(birch_adjusted_mutual_info_scores_y_pred.values()))))),
                                     columns=['K means - predicted class', 'BIRCH - predicted class'])
        df_clustering_y_pred.index = name_layers
        plt.figure(figsize=(8, 5.5))
        sns.lineplot(data=df_clustering_y_true)
        plt.legend(title='', fontsize=14)
        plt.xlabel("Layers", fontsize=18)
        plt.ylabel("Adjusted mutual information score", fontsize=18)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + '/adjusted_mutual_info_score_y_true.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(8, 6.5))
        sns.lineplot(data=df_clustering_y_pred)
        plt.legend(title='', fontsize=14)
        plt.xlabel("Layers", fontsize=18)
        plt.ylabel("Adjusted mutual information score", fontsize=18)
        plt.tick_params(axis='y', which='major', labelsize=14)
        plt.tick_params(axis='x', which='major', labelsize=18, labelrotation=45)
        plt.tight_layout()
        plt.savefig(save_path + '/adjusted_mutual_info_score_y_pred.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

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
        plt.savefig(save_path + '/adjusted_mutual_info_score_big.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    '''if clustering:
        nb_cluters = 20
        Spect_clustering = SpectralClustering(n_clusters=nb_cluters)
        Spect_clustering_fits = {}
        Spect_clustering_adjusted_mutual_info_scores = {}
        for key, values_layer in dict_values_layers.items():
            Spect_clustering_fits[key] = Spect_clustering.fit_predict(values_layer)
            Spect_clustering_adjusted_mutual_info_scores[key] = adjusted_mutual_info_score(Spect_clustering_fits[key], y_pred_MonoNet)

        plt.plot(Spect_clustering_adjusted_mutual_info_scores.values())
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], list(Spect_clustering_adjusted_mutual_info_scores.keys()), rotation=45)
        plt.xlabel("Layers")
        plt.ylabel("Adjusted mutual information score: Spectral clustering")
        plt.savefig(save_path + '/Spectral_clustering_adjusted_mutual_info_score.eps', format='eps',
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()'''
######################################################################################################################
    nb_color = len(np.unique(pd.Series(y_pred_tags)))
    name_layers2 = ['layer_monotonic_2',
                   'layer_inter_out']
    if tsne_plots:
        for name in name_layers:
            tSNE_plots(values_layer_=dict_values_layers[name], name_layer=name)
    nb_color = len(np.unique(pd.Series(y_pred_tags)))
######################################################################################################################
    # COMPARISON OF HIGH LEVEL PROFILES FOR SIMILAR or DIFFERENT CELL TYPES
    if verbose > 0:
        comp_high_level_profiles(h=h_values, y_input=y_subset, type_1=6, type_2=7)
        comp_high_level_profiles(h=h_values, y_input=y_subset, type_1=6, type_2=10)

######################################################################################################################
    # T_SNE plots of the inputs and the h_values

    '''df_subset = pd.DataFrame(data=np.hstack((X_subset, y_subset.reshape(len(y_subset), 1))),
                             columns=list(df.columns[:-1]) + ['category'])
    df_h_values = pd.DataFrame(data=np.hstack((h_values.detach().numpy(), y_subset.reshape(len(y_subset), 1))),
                               columns=list(np.arange(8) + 1) + ['category'])
    if verbose > 0:
        t_SNE_plots_mean(df_input_=df_subset, df_h_values_=df_h_values)'''

######################################################################################################################
    # Plots of the multivariate function of the monotonic block

    h_and_class = np.hstack((h_values.detach().numpy(), y_subset.reshape(len(y_subset), 1)))

    if mono_f:
        monotonic_f(h_and_class_=h_and_class, neuron_=4, target_class_=1, fixed_class_=1)
        monotonic_f(h_and_class_=h_and_class, neuron_=4, target_class_=5, fixed_class_=5)
        monotonic_f(h_and_class_=h_and_class, neuron_=4, target_class_=6, fixed_class_=6)
        monotonic_f(h_and_class_=h_and_class, neuron_=4, target_class_=8, fixed_class_=8)
        monotonic_f(h_and_class_=h_and_class, neuron_=4, target_class_=2, fixed_class_=2)
        # monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=3, fixed_class_=3)
        # monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=4, fixed_class_=4)
        # monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=5, fixed_class_=5)
        # monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=6, fixed_class_=6)
        # monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=7, fixed_class_=7)
        monotonic_f(h_and_class_=h_and_class, neuron_=1, target_class_=8, fixed_class_=8)
        # monotonic_f(h_and_class_=h_and_class, neuron_=5, target_class_=15, fixed_class_=15)
        # monotonic_f(h_and_class_=h_and_class, neuron_=5, target_class_=16, fixed_class_=16)
        # monotonic_f(h_and_class_=h_and_class, neuron_=5, target_class_=18, fixed_class_=18)
        # monotonic_f(h_and_class_=h_and_class, neuron_=5, target_class_=19, fixed_class_=9)
        # monotonic_f(h_and_class_=h_and_class, neuron_=6, target_class_=8, fixed_class_=8)
        # monotonic_f(h_and_class_=h_and_class, neuron_=6, target_class_=8, fixed_class_=9)
        monotonic_f(h_and_class_=h_and_class, neuron_=6, target_class_=8, fixed_class_=10)
        # monotonic_f(h_and_class_=h_and_class, neuron_=6, target_class_=8, fixed_class_=11)
        # monotonic_f(h_and_class_=h_and_class, neuron_=2, target_class_=8, fixed_class_=8)

    if not os.path.exists(save_path + '/Monotonic_function/'):
        os.makedirs(save_path + '/Monotonic_function/')

        # x_lim_min = h_and_class_[:, neuron_].min()
        # x_lim_max = h_and_class_[:, neuron_].max()
    neuron_ = 4
    target_class_ = 5
    fixed_class_ = 5
    x_ = np.arange(-0.5, 0.5, 0.01)
    # Extract values of the interpretable neurons for the fixed chosen class
    h_fixed_class = h_and_class[h_and_class[:, -1] == fixed_class_ - 1,:]
    # Calculate the mean for each interpretable neuron
    mean_class = np.mean(h_fixed_class, axis=0)
    h_mon = np.full([len(x_), h_values.size()[1]], None)
    for idx_n in range(NB_NEURON_INTER_LAYER):
        h_mon[:, idx_n] = np.repeat(mean_class[idx_n], len(x_))
    # Replace in the neuron of interest the vector x_
    h_mon[:, neuron_ - 1] = x_

    # Pass the h_values into the monotonic block
    y_ = model.monotonic_block(torch.from_numpy(h_mon.astype('float32')))[:, target_class_ - 1].detach().numpy()

    x_5 = x_
    y_5 = y_
    target_class_ = 6
    fixed_class_ = 6
    x_ = np.arange(-0.5, 0.5, 0.01)
    # Extract values of the interpretable neurons for the fixed chosen class
    h_fixed_class = h_and_class[h_and_class[:, -1] == fixed_class_ - 1,:]
    # Calculate the mean for each interpretable neuron
    mean_class = np.mean(h_fixed_class, axis=0)
    h_mon = np.full([len(x_), h_values.size()[1]], None)
    for idx_n in range(NB_NEURON_INTER_LAYER):
        h_mon[:, idx_n] = np.repeat(mean_class[idx_n], len(x_))
    # Replace in the neuron of interest the vector x_
    h_mon[:, neuron_ - 1] = x_

    # Pass the h_values into the monotonic block
    y_ = model.monotonic_block(torch.from_numpy(h_mon.astype('float32')))[:, target_class_ - 1].detach().numpy()
    x_6 = x_
    y_6 = y_

    plt.figure(figsize=(7, 5))
    plt.plot(x_5, y_5, color='tab:blue')
    plt.xlabel(f'Neuron {neuron_}', fontsize=16)
    plt.ylabel(f'Activation value of the output neuron\ncorresponding to {meta_df.loc[4][1]}',
               multialignment='center', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/mono_f_neuron_4_output_5.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(x_6, y_6, color='tab:red')
    plt.xlabel(f'Neuron {neuron_}', fontsize=16)
    plt.ylabel(f'Activation value of the output neuron\ncorresponding to {meta_df.loc[5][1]}',
               multialignment='center', fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.tick_params(axis='x', which='major', labelsize=14, labelrotation=45)
    plt.tight_layout()
    plt.savefig(save_path + '/mono_f_neuron_4_output_6.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    ########################################################################################################################
# --------------------------------------- Analysis unconstrained block ---------------------------------------
#######################################################################################################################

    # Analysis using violin plots

    if plot_violin:
        dict_df_stats_KS = {}
        dict_df_pvalues_KS = {}
        all_stat_KS = []
        for name in name_layers:
            values_layers = dict_values_layers[name]
            dict_df_stats_KS[name], dict_df_pvalues_KS[name] = compute_ks(h_values_=values_layers, X_input_=X_subset, y_input_=y_subset)
            df_temp = pd.melt(dict_df_stats_KS[name], id_vars=['neuron'], ignore_index=False)[['variable', 'value']]
            df_temp['layer'] = [name] * len(df_temp)
            all_stat_KS.append(df_temp)

        df_all_stat_KS = pd.concat(all_stat_KS, ignore_index=True)

        plt.figure(figsize=(13, 9))
        ax = sns.pointplot(x="layer", y="value", hue="variable",
                           data=df_all_stat_KS, ci=None)

        ax.set(xlabel='Biomarker', ylabel='Layer')
        plt.xticks(rotation=45)
        ax.set_title('Average KS score across layers')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(save_path + '/plots_KS_score across layers.eps', format='eps', dpi=300,
                    bbox_inches='tight')
        plt.show()

        # Violin plots for the top and bottom distribution of the input (the ordering is done wrt the h_values)
        violin_plots(h_values_=h_values, X_input_=X_subset, y_input_=y_subset)

        # Computation of several distances for the difference between top and bottom distributions
        df_stats_KS, df_pvalues_KS = compute_ks(h_values_=h_values, X_input_=X_subset, y_input_=y_subset)

        #df_ks_discrete_stats, df_ks_discrete_pvalues = compute_discrete_ks(h_values_=h_values_reduced, X_input_=X_reduced,
                                    #                                       y_input_=y_reduced)
        #df_M_Segregation, df_M_seg_pvalues = compute_m_seg(h_values_=h_values_reduced, X_input_=X_reduced, y_input_=y_reduced)

        # Plots of the previous distance measures
        #plots_distance_measure(df_measure=df_M_Segregation, measure='Segregation')
        #plots_distance_measure(df_measure=df_ks_discrete_stats, measure='KS discrete')
        plots_distance_measure(df_measure=df_stats_KS, measure='KS (continuous)', p_values=df_pvalues_KS)
        sns.heatmap(np.cov(np.array(df_stats_KS.iloc[:, :-1]), rowvar=True))
        # Is there a difference between the top and bottom distributions? (H_0 = the distributions are the same so
        # if p<0.01 we reject the hypothesis that the distributions are the same)

        df_diff_in_distributions_KS = df_pvalues_KS < 0.01
        df_diff_in_distributions_KS.index = list(range(1, 9, 1))

        #df_diff_in_distributions_Mseg = df_M_seg_pvalues < 0.01
        #df_diff_in_distributions_Mseg.index = list(range(1, 9, 1))

########################################################################################################################

        qualitative_colors = sns.color_palette("Paired", 13)
        if not os.path.exists(save_path + '/stacked_barplots/'):
            os.makedirs(save_path + '/stacked_barplots/')

        stacked_barplot_measure_distr(measure=str('stats_KS'))
        #stacked_barplot_measure_distr(measure=str('M_Segregation'))
        #stacked_barplot_measure_distr(measure=str('M_Segregation'), opposite=True)
########################################################################################################################

    # Loading results from FCI algo run on R
    R_file = pd.read_csv('/Users/dam/concat_mat_R.csv', index_col=0)
    all_neurons_no_constraints = R_file.iloc[0:8]
    one_by_one_no_constraints = R_file.iloc[8:16]
    one_by_one_constrainted = R_file.iloc[16:24]
    all_neurons_constrainted = R_file.iloc[24:]



    # Loading results from PC algo run on R
    R_PC_IDA = pd.read_csv('ida_results.csv', index_col=0)

    visualize_PC_IDA(list(df.columns)[:-1], R_PC_IDA)

    # Analysis using Shapley values and captum methods

    if captum:
        if compute_shap_values:
            exp_shap = {}
            X_train_summary = shap.kmeans(X_train, 50)
            for i in range(NB_NEURON_INTER_LAYER):
                explainer_shap = shap.KernelExplainer(model=eval(f"model.unconstrainted_block_{i}"),
                                                      data=X_train_summary)
                # TODO: Which dataset should we use?, what should we put there as reference ?
                exp_shap[str('neuron_') + str(i + 1)] = \
                explainer_shap.shap_values(shap.sample(X_test, 100))[0] # TODO should we take idx_all_class_redu
                # shap.force_plot(explainer_shap.expected_value[0], exp_shap[str('neuron_') + str(i + 1)], X_test)

                # shap.summary_plot(shap_values[0], X_test[0:50])
            visualize_importances_all_neurons(list(df.columns)[:-1], exp_shap,
                                              title="Average Feature Importances for ", method="Shapley Values approximation")

        # LIME # TODO: explainer works only for one instance!!!
        '''explainer = LimeTabularExplainer(X_train,  # TODO: which dataset should we use?
                                         feature_names=df.columns,
                                         class_names=['impact'],
                                         mode='regression')
        exp = explainer.explain_instance(X_test[50], model.unconstrainted_block_0,
                                         num_features=10)
        
        exp.as_pyplot_figure()
        
        exp_lime = {}
        explainer_lime = LimeTabularExplainer(X_train,  # TODO: which dataset should we use?
                                              feature_names=df.columns,
                                              class_names=['impact'],
                                              mode='regression')
        for i in range(NB_NEURON_INTER_LAYER):
            idx_sampled = random.sample(idx_all_class_redu, 2)
        
            for idx in idx_sampled:
                exp_lime_sample = explainer_lime.explain_instance(X_test[idx],
                                                                  eval(f"model.unconstrainted_block_{i}"),
                                                                  num_features=13)
                if str('neuron_')+str(i+1) in list(exp_lime.keys()):
                    exp_lime[str('neuron_')+str(i+1)] = np.vstack((exp_lime[str('neuron_')+str(i+1)], exp_lime_sample))
                else:
                    exp_lime[str('neuron_')+str(i+1)] = exp_lime_sample
        
        # TODO: increase the number (low for computational time for the moment)'''

######################################################################################################################

        # Captum Gradient Shap
        '''att_grad_shap = {}

        neuron_grad_shap = NeuronGradientShap(model, model.layer_inter)
        baseline = torch.zeros(100, 13, requires_grad=True) # torch.randn(100, 13, requires_grad=True)  # TODO: review which baseline we should use, didn't find something on that, either zero tensor or random tensor
        infidelities_grad_shap = {}

        def perturb_fn(inputs):
            noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
            return noise, inputs - noise

        for i in range(NB_NEURON_INTER_LAYER):
            att_grad_shap[str('neuron_') + str(i + 1)] = \
                neuron_grad_shap.attribute(inputs=torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                           neuron_selector=i, baselines=baseline, attribute_to_neuron_input=True)

            infidelities_grad_shap[str('neuron_') + str(i + 1)] = \
                infidelity(eval(f'model.unconstrainted_block_{i}_torch'), perturb_fn,
                torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                att_grad_shap[str('neuron_') + str(i + 1)])

            #  sens = sensitivity_max(neuron_grad_shap.attribute, torch.from_numpy(X_test[idx_all_class_redu].astype('float32')))
        visualize_importances_all_neurons(list(df.columns)[:-1], att_grad_shap,
                                          title="Average Feature Importances for ", method="Gradient Shap")

        # Captum Neuron Gradient
        att_grad = {}
        infidelities_grad={}
        neuron_grad = NeuronGradient(model, model.layer_inter)
        for i in range(NB_NEURON_INTER_LAYER):
            att_grad[str('neuron_') + str(i + 1)] = \
                neuron_grad.attribute(torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                      neuron_selector=i, attribute_to_neuron_input=True)
        infidelities_grad[str('neuron_') + str(i + 1)] = \
            infidelity(eval(f'model.unconfstrainted_block_{i}_torch'), perturb_fn,
            torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
            att_grad[str('neuron_') + str(i + 1)])

        visualize_importances_all_neurons(list(df.columns)[:-1], att_grad,
                                          title="Average Feature Importances for ", method="Gradient")'''

        # Captum Integrated Gradient
        att_int_grad = {}
        infidelities_int_grad = {}
        neuron_int_grad = NeuronIntegratedGradients(model, model.layer_inter)
        for i in range(NB_NEURON_INTER_LAYER):
            att_int_grad[str('neuron_') + str(i + 1)] = \
                neuron_int_grad.attribute(torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                          neuron_selector=i, attribute_to_neuron_input=True)
            '''infidelities_int_grad[str('neuron_') + str(i + 1)] = \
                infidelity(eval(f'model.unconstrainted_block_{i}_torch'), perturb_fn,
                torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                att_int_grad[str('neuron_') + str(i + 1)])'''

        visualize_importances_all_neurons(list(df.columns)[:-1], att_int_grad,
                                          title="Average Feature Importances for ", method="Integrated Gradient")

        # Captum Conductance
        att_conductance = {}
        infidelities_conductance ={}
        neuron_conductance = NeuronConductance(model, model.layer_inter)
        # TODO: what is target? target class
        for i in range(NB_NEURON_INTER_LAYER):
            conductances = np.empty((0, 13))
            X_redu = torch.from_numpy(X_test[idx_all_class_redu].astype('float32'))
            y_pred_redu = y_pred_tags[idx_all_class_redu]
            for idx in range(X_redu.shape[0]):
                cond = neuron_conductance.attribute(
                    inputs=X_redu[idx].reshape((1, 13)),
                    neuron_selector=i, target=int(y_pred_redu[idx]), attribute_to_neuron_input=True)
                conductances = np.append(conductances, np.array(cond), axis=0)
            '''for j in range(NUM_CLASSES): # Try to put instead the true/predicted class as the target
                cond = neuron_conductance.attribute(inputs=torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                                    neuron_selector=i, target=j, attribute_to_neuron_input=True)
                conductances += cond
            mean_cond = sum(conductances)/NUM_CLASSES'''
            att_conductance[str('neuron_') + str(i + 1)] = torch.from_numpy(conductances)

            '''infidelities_conductance[str('neuron_') + str(i + 1)] = \
                infidelity(eval(f'model.unconstrainted_block_{i}_torch'), perturb_fn,
                torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                att_conductance[str('neuron_') + str(i + 1)])'''

        visualize_importances_all_neurons(list(df.columns)[:-1], att_conductance,
                                          title="Average Feature Importances for ", method="Conductances")

        # Captum Neuron DeepLift
        '''att_deep_lift = {}
        infidelities_deep_lift = {}
        neuron_deep_lift = NeuronDeepLift(model, model.layer_inter)
        for i in range(NB_NEURON_INTER_LAYER):
            att_deep_lift[str('neuron_') + str(i + 1)] = \
                neuron_deep_lift.attribute(inputs=torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                           neuron_selector=i, attribute_to_neuron_input=True)

               infidelities_deep_lift[str('neuron_') + str(i + 1)] = \
                infidelity(eval(f'model.unconstrainted_block_{i}_torch'), perturb_fn,
                torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                att_deep_lift[str('neuron_') + str(i + 1)])

        visualize_importances_all_neurons(list(df.columns)[:-1], att_deep_lift,
                                          title="Average Feature Importances for ", method="DeepLift")

        # Captum Neuron DeepLiftShap

        att_deep_lift_shap = {}
        infidelities_deep_lift_shap = {}
        baseline = torch.randn(100, 13, requires_grad=True)
        neuron_deep_lift_shap = NeuronDeepLiftShap(model, model.layer_inter)
        for i in range(NB_NEURON_INTER_LAYER):
            X_Test_reduced = X_test[random.sample(idx_all_class_redu, 100)].astype('float32')
            att_deep_lift_shap[str('neuron_') + str(i + 1)] = \
                neuron_deep_lift_shap.attribute(
                    inputs=torch.from_numpy(X_Test_reduced), #X_test[random.sample(idx_all_class_redu, 100)].astype('float32')
                    neuron_selector=i, attribute_to_neuron_input=True, baselines=baseline) # TODO: Can we put more than 100?
                infidelities_deep_lift_shap[str('neuron_') + str(i + 1)] = \
                infidelity(eval(f'model.unconstrainted_block_{i}_torch'), perturb_fn,
                           torch.from_numpy(X_Test_reduced),
                           att_deep_lift_shap[str('neuron_') + str(i + 1)])

        visualize_importances_all_neurons(list(df.columns)[:-1], att_deep_lift_shap,
                                          title="Average Feature Importances for ", method="DeepLiftShap")'''

    # Captum Gradient Shap
    att_grad_shap_layers_max = []
    baseline = torch.randn(100, 13, requires_grad=True)
    for name_layer in name_layers[1:]:
        att_grad_shap_mean = {}
        neuron_grad_shap = NeuronGradientShap(model, eval(f'model.{name_layer}'))

        for i in range(eval(f'model.{name_layer}').in_features):

            att_grad_shap_mean[str('neuron_') + str(i + 1)] = \
                neuron_grad_shap.attribute(inputs=torch.from_numpy(X_test[idx_all_class_redu].astype('float32')),
                                           neuron_selector=i, baselines=baseline, attribute_to_neuron_input=True).mean(axis=0)
        arr = np.empty((0, 13), float)
        for keys, values in att_grad_shap_mean.items():
            arr = np.append(arr, np.abs(values.detach().numpy().reshape((1, 13))), axis=0)

        att_grad_shap_layers_max += [np.max(arr, axis=0)]
    df_att_grad_shap_layers_max = pd.DataFrame(data=np.array(att_grad_shap_layers_max), columns=list(df.columns[:-1]), index=name_layers[1:])
    scaler=MinMaxScaler()
    df_norm = pd.DataFrame(data=scaler.fit_transform(df_att_grad_shap_layers_max), columns=list(df.columns[:-1]), index=name_layers[1:])
    df_ = pd.melt(df_att_grad_shap_layers_max,  ignore_index=False)
    df_['layer'] = df_.index
    df_norm_ = pd.melt(df_norm,  ignore_index=False)
    df_norm_['layer'] = df_norm_.index
    plt.figure(figsize=(13, 9))
    ax = sns.pointplot(x="layer", y="value", hue="variable",
                       data=df_, ci=None)

    ax.set(xlabel='Biomarkers', ylabel='Layers')
    plt.xticks(rotation=45)
    ax.set_title('Mean absolute')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.savefig(save_path + '/plots_mean_abs_activation_across_layers_norm.eps', format='eps', dpi=300,
              #  bbox_inches='tight')
    plt.show()

    print('The end')
