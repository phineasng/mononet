from functools import partial
import os
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter

import collections
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# from MonoNet_class import *
from experimental_code.tabular_data.MonoNet_class import *
from experimental_code.tabular_data.MonoNet_train import MonoNet_train


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
    save_path = str(f'/Users/{short_name}/Desktop/Plots_MonoNet/') + in_file_nn.split('/')[-1].split('.')[0]

    plt.savefig(save_path + '/loss_accuracy_training.eps', format='eps', dpi=300,
                bbox_inches='tight')
    plt.close()


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def MonoNet_train(config, checkpoint_dir=None, data_dir=None):
    # Prepare subsets for training
    # TODO: maybe use load_data
    train_loader = DataLoader(dataset=config['train_dataset'],
                              batch_size=config['batch_size'],
                              sampler=config['weighted_sampler']
                              )
    val_loader = DataLoader(dataset=config['val_dataset'], batch_size=1)
    test_loader = DataLoader(dataset=config['test_dataset'], batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the model
    model = MonoNet(num_feature=config['num_features'], num_class=config['num_classes'],
                    nb_neuron_inter_layer=config['nb_neuron_inter_layer'])
    init_mode = config['init_mode']
    if init_mode == 'normal':
        model.apply(init_weights_normal)
    elif init_mode == 'uniform':
        model.apply(init_weights_uniform)
    elif init_mode == 'xavier_uniform':
        model.apply(init_weights_xavier_uniform)
    elif init_mode == 'xavier_normal':
        model.apply(init_weights_xavier_normal)
    elif init_mode == 'kaiming_uniform':
        model.apply(init_weights_kaiming_uniform)
    elif init_mode == 'kaiming_normal':
        model.apply(init_weights_kaiming_normal)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=config['class_weights'].to(device))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    ##################################################################################################################
    # --------------------------------------- TRAINING ---------------------------------------
    print("Begin training.")

    # for e in tqdm(range(1, EPOCHS + 1)):
    for e in range(1, config['epochs'] + 1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()

            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        # scheduler.step(val_epoch_loss)
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        with tune.checkpoint_dir(e) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: '
            f'{val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: '
            f'{val_epoch_acc / len(val_loader):.3f}')

    # Test accuracy
    y_test_pred = model(torch.from_numpy(config['X_test']).clone().to(torch.float32))

    test_loss = criterion(y_test_pred, torch.from_numpy(config['y_test']))
    test_acc = multi_acc(y_test_pred, torch.from_numpy(config['y_test']))
    print(
        f'For the test set after {config["epochs"]} epochs: Test Loss: {test_loss.detach().numpy():.5f} | Test Acc: {test_acc.detach().numpy():.3f}')

    model.accuracy_train = round(accuracy_stats['train'][-1], 2)
    model.accuracy_val = round(accuracy_stats['val'][-1], 2)
    model.accuracy_test = test_acc.item()
    model.accuracy_hist = accuracy_stats
    model.loss_hist = loss_stats
    tune.report(loss_hist=loss_stats, accuracy_hist=accuracy_stats,
                loss=(val_epoch_loss / len(val_loader)), accuracy=val_epoch_acc / len(val_loader))

    # return model


def main(num_samples=10):
    short_name = 'mor'
    save_path = str(f'/Users/{short_name}/Desktop/Plots_MonoNet/') + in_file_nn.split('/')[-1].split('.')[0]
    if not os.path.exists(save_path + '/'):
        os.makedirs(save_path + '/')

    # if not os.path.exists(f'/Users/{short_name}/Desktop/Plots_MonoNet/'):
    # os.makedirs(f'/Users/{short_name}/Desktop/Plots_MonoNet/')
    captum = args.analysis_captum_methods
    plot_violin = args.analysis_with_violin_plots

    # Loading data
    meta_df = pd.read_csv(f'/Users/{short_name}/Desktop/single_cell_data/metadata.csv')
    df = pd.read_csv(f'/Users/{short_name}/Desktop/single_cell_data/data.csv')

    print(df.head)
    print(meta_df)

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

    # Prepare the subtests for pytorch
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    # Plotting the class distribution for each subset
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

    # Hyperparameters for the Neural network
    EPOCHS = 20  # 80
    BATCH_SIZE = 35  # 40
    LEARNING_RATE = 0.00045  # 0.00045 0.00055
    NUM_FEATURES = len(X.columns)
    NUM_CLASSES = len(np.unique(y_train))
    NB_NEURON_INTER_LAYER = 8
    data_dir = os.path.abspath(f'/Users/{short_name}/Desktop/single_cell_data/')

    config = {
        "lr": tune.choice([0.00045]), # tune.quniform(0.001, 0.0014, 0.00005),# tune.choice([0.0007, 0.0005, 0.00055, 0.0009]),# tune.loguniform(1e-4, 1e-1),
        # tune.choice([0.00045, 0.0005, 0.00055, 0.0009]),
        "batch_size": tune.choice([40, 45]),
        "init_mode": tune.choice(
            ["xavier_uniform"]),
                 #tune.choice(
            #["normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"]),
        'epochs': EPOCHS,
        'num_features': NUM_FEATURES,
        'num_classes': NUM_CLASSES,
        'nb_neuron_inter_layer': tune.choice([4, 8, 16]),
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_weights': class_weights,
        'weighted_sampler': weighted_sampler,
        'X_test': X_test,
        'y_test': y_test,
        'out_file_nn': out_file_nn

    }

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(MonoNet_train, data_dir=data_dir),
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = MonoNet(best_trial.config['num_features'], best_trial.config['num_classes'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = best_trained_model.accuracy_test
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    short_name = 'mor'

    parser = argparse.ArgumentParser(prog='MonoNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', type=bool, default=False, help='Specify if one should load or train a MonoNet')
    parser.add_argument('--in_file_NN', type=str,
                        default=f'/Users/{short_name}/Documents/GitHub/MonoNet/nn_saved.pickle')
    # --in_file_NN
    # /Users/mor/Documents/GitHub/MonoNet_single_cell/nn_saved_04_11_50_epochs.pickle
    parser.add_argument('--out_file_NN', type=str,
                        default=f'/Users/{short_name}/Documents/GitHub/MonoNet/nn_26_04_saved.pickle')

    parser.add_argument('--verbose', type=int, default=0)

    parser.add_argument('--analysis_captum_methods', type=bool, default=False)
    parser.add_argument('--analysis_with_violin_plots', type=bool, default=True)
    parser.add_argument('--compute_shap_values', type=bool, default=False)

    args = parser.parse_args()
    load_bool = args.load
    compute_shap_values = args.compute_shap_values
    out_file_nn = args.out_file_NN
    in_file_nn = args.in_file_NN
    verbose = args.verbose

    main()
