import click
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from mononet.datasets.tabular_data import ClassifierDataset
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


@click.command()
@click.option('--data_root', required=True, help='Where to retrieve the data')
def run_train(data_root):
    data_fpath = os.path.join(data_root, 'data.csv')
    metadata_fpath = os.path.join(data_root, 'metadata.csv')

    meta_df = pd.read_csv(metadata_fpath)
    df = pd.read_csv(data_fpath)
    df.category = df.category - 1

    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

    # Split train into train-val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    #train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    #val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    #test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    scores = []
    depths = []
    reg_scores = []
    reg_depths = []
    for random_seed in [1231, 4353, 32141, 9875, 12345, 46321, 123, 5876]:
        dt = DecisionTreeClassifier(random_state=random_seed)
        dt.fit(X_train, y_train)
        scores.append(dt.score(X_test, y_test))
        depths.append(dt.get_depth())

        dt = DecisionTreeClassifier(random_state=random_seed, max_depth=5)
        dt.fit(X_train, y_train)
        reg_scores.append(dt.score(X_test, y_test))
        reg_depths.append(dt.get_depth())
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))
    print(np.mean(depths))
    print(reg_scores)
    print(np.mean(reg_scores))
    print(np.std(reg_scores))
    print(np.mean(reg_depths))


if __name__ == '__main__':
    run_train()