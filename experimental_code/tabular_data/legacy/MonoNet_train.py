import numpy as np
import pandas as pd
import seaborn as sns
import collections
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import init
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.functional import softsign
import math
from tqdm import tqdm
import time
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
from random import sample
from sklearn.manifold import TSNE
# from MonoNet_class import *

from MonoNet_class import *
import os


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def MonoNet_train(EPOCHS,BATCH_SIZE,  LEARNING_RATE, NUM_FEATURES, NUM_CLASSES, NB_NEURON_INTER_LAYER,
                  train_dataset, val_dataset, test_dataset, class_weights, weighted_sampler, X_test, y_test,
                  out_file_nn):

    print(f'The parameters of the model are: \n EPOCHS: {EPOCHS} | BATCH_SIZE: {BATCH_SIZE} | '
          f'LEARNING_RATE: {LEARNING_RATE} NUM_FEATURES: {NUM_FEATURES} | NUM_CLASSES: {NUM_CLASSES} | '
          f'NB_NEURON_INTER_LAYER: {NB_NEURON_INTER_LAYER}')

    # Prepare subsets for training
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the model
    model = MonoNet(num_feature=NUM_FEATURES, num_class=NUM_CLASSES, nb_neuron_inter_layer=NB_NEURON_INTER_LAYER)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, threshold=0.001, verbose=True)
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
    for e in range(1, EPOCHS + 1):
        #if e == 20:
           # LEARNING_RATE = LEARNING_RATE/2
        if e == 50:
            LEARNING_RATE = LEARNING_RATE/2
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
        #scheduler.step(val_loss)
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        #scheduler.step(sum(loss_stats['val'])/len(loss_stats['val']))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: '
            f'{val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: '
            f'{val_epoch_acc / len(val_loader):.3f}')

    # Test accuracy
    y_test_pred = model(torch.from_numpy(X_test).clone().to(torch.float32))

    test_loss = criterion(y_test_pred, torch.from_numpy(y_test))
    test_acc = multi_acc(y_test_pred, torch.from_numpy(y_test))
    print(
        f'For the test set after {EPOCHS} epochs: Test Loss: {test_loss.detach().numpy():.5f} | Test Acc: {test_acc.detach().numpy():.3f}')

    model.accuracy_train = round(accuracy_stats['train'][-1], 2)
    model.accuracy_val = round(accuracy_stats['val'][-1], 2)
    model.accuracy_test = test_acc.item()
    model.accuracy_hist = accuracy_stats
    model.loss_hist = loss_stats
    torch.save(model, out_file_nn)
    return model
