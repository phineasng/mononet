import pytorch_lightning as pl
from mononet.datasets.vision_data import *
from mononet.monotonic_cnn import *
import medmnist
from medmnist.evaluator import Evaluator
from torchvision import transforms
import click
from torch.utils.data import DataLoader
from mononet.datasets.vision_data import get_data
from datetime import datetime
import torch
import os

DATASETS = [
    "pathmnist",
    "octmnist",
    "pneumoniamnist",
    "dermamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist"
]


def make_timestamp_folder(root='.'):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(root, ts)
    return path, ts


@click.command()
@click.option('--benchmark', is_flag=True, default=False, help="If to perform benchmark over all datasets")
@click.option('--data_flag', default='dermamnist', help=f"Dataset to benchmark, If the benchmark flag is not set. "
                                                        f"Available datasets: {DATASETS}")
@click.option('--results_path', default=None, help="Where to store/retrieve the results and the model")
@click.option('--data_path', required=True, help="Where to store/retrieve the dataset")
@click.option('--model_name', default='ResidualMonotonicCNN1', help="Model to train")
@click.option('--n_epochs', default=10, help="Number of training_epochs")
@click.option('--lr', default=0.001, help="Learning rate")
@click.option('--batch_size', default=64, help="Batch size")
@click.option('--optimizer', default='adam', help="Optimizer to use")
def main(benchmark, data_flag, results_path, data_path, model_name, n_epochs, lr, batch_size, optimizer):
    if benchmark:
        datasets = DATASETS
    else:
        datasets = [data_flag]

    # preprocess results folder
    if results_path is None:
        results_path = os.getcwd()
    results_path, ts = make_timestamp_folder(results_path)
    os.makedirs(results_path, exist_ok=True)

    for data_flag in datasets:
        path = os.path.join(results_path, data_flag)
        path = os.path.join(path, model_name)
        os.makedirs(path, exist_ok=True)

        BATCH_SIZE = batch_size

        ds, loaders, info = get_data(data_flag, data_path, BATCH_SIZE)

        train_loader, valid_loader, test_loader = loaders['train'], loaders['valid'], loaders['test']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        trainer = pl.Trainer(max_epochs=n_epochs, check_val_every_n_epoch=5, log_every_n_steps=1,
                             default_root_dir=path)

        model = globals()[model_name](nclasses=n_classes, nchannels=n_channels, optimizer=optimizer, lr=lr,
                                      with_monotonic=True)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        model = model.eval()
        trainer.test(model, dataloaders=test_loader)

        evaluator = Evaluator(data_flag, split='test', root=data_path)
        y_hat = []
        for x, _ in test_loader:
            y_hat.append(torch.softmax(model(x), dim=1).detach().cpu().numpy())
        y_hat = np.concatenate(y_hat, axis=0)
        metrics = evaluator.evaluate(y_hat, save_folder=results_path, run=ts)
        print(metrics)


if __name__ == '__main__':
    main()
