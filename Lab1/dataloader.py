from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def get_mnist_dataset():

    # MNIST Transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST
    ds_train_full = MNIST(root="./data", train=True, download=True, transform=transform)
    ds_test = MNIST(root="./data", train=False, download=True, transform=transform)

    # Split into train/val
    val_size = 5000
    indices = np.random.permutation(len(ds_train_full))
    ds_val = Subset(ds_train_full, indices[:val_size])
    ds_train = Subset(ds_train_full, indices[val_size:])

    return ds_train, ds_val, ds_test


def get_cifar10_dataset():
    # CIFAR10 Transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR10
    ds_train_full = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    ds_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Split into train/val
    val_size = 5000
    indices = np.random.permutation(len(ds_train_full))
    ds_val = Subset(ds_train_full, indices[:val_size])
    ds_train = Subset(ds_train_full, indices[val_size:])

    return ds_train, ds_val, ds_test
