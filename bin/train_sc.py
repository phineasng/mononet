import pytorch_lightning as pl
import click
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning import LightningDataModule
from mononet.monotonic_mlp import SingleCellMonoNetExample
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def make_timestamp_folder(root='.'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(root, ts)
    return path, ts


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class SCDataModule(LightningDataModule):
    def __init__(self, data_path, metadata_path, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        # meta_df = pd.read_csv(metadata_path)
        df = pd.read_csv(data_path)

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
        self.train_dataset = ClassifierDataset(X_train.astype(np.single), y_train.astype(np.long))
        self.val_dataset = ClassifierDataset(X_val.astype(np.single), y_val.astype(np.long))
        self.test_dataset = ClassifierDataset(X_test.astype(np.single), y_test.astype(np.long))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)


@click.command()
@click.option('--results_path', default=None, help="Where to store/retrieve the results and the model")
@click.option('--data_path', required=True, help="Where to retrieve the dataset")
@click.option('--n_epochs', default=10, help="Number of training_epochs")
@click.option('--lr', default=0.001, help="Learning rate")
@click.option('--batch_size', default=64, help="Batch size")
@click.option('--optimizer', default='adam', help="Optimizer to use")
def main(results_path, data_path, n_epochs, lr, batch_size, optimizer):

    # preprocess results folder
    if results_path is None:
        results_path = os.getcwd()
    results_path, ts = make_timestamp_folder(results_path)
    os.makedirs(results_path, exist_ok=True)

    data = os.path.join(data_path, 'data.csv')
    metadata = os.path.join(data_path, 'metadata.csv')
    dm = SCDataModule(data, metadata, batch_size)

    checkpoint_callback = ModelCheckpoint(dirpath=results_path, every_n_epochs=5)
    trainer = pl.Trainer(max_epochs=n_epochs, check_val_every_n_epoch=5, log_every_n_steps=1,
                         default_root_dir=results_path, callbacks=[checkpoint_callback])
    model = SingleCellMonoNetExample(optimizer=optimizer, lr=lr)
    trainer.fit(model, datamodule=dm)
    model = model.eval()
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    main()
