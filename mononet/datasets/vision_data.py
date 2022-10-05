import medmnist
from medmnist.dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageMode


MEDMNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

MEDMNIST_INVERSE_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[-1.], std=[2.]),
    transforms.ToPILImage()
])


def get_data(data_flag, data_path, batch_size):
    download = True

    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    data_transform = MEDMNIST_TRANSFORM
    train_dataset = DataClass(split='train', transform=data_transform, download=download, root=data_path)
    valid_dataset = DataClass(split='val', transform=data_transform, download=download, root=data_path)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, root=data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    return datasets, loaders, info
