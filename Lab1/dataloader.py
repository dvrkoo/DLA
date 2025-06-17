from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset, Dataset
from PIL import Image

cifar_train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


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


def get_cifar10_dataset(distillation):
    # CIFAR10 Transform
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    # Load CIFAR10
    ds_train_full = CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    ds_test = CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    # Split into train/val
    val_size = 5000
    indices = np.random.permutation(len(ds_train_full))
    ds_val = Subset(ds_train_full, indices[:val_size])
    ds_train = Subset(ds_train_full, indices[val_size:])

    if distillation:
        # For distillation, we use the full training set as the training set
        ds_train_eval_full = CIFAR10(
            root="./data", train=True, download=True, transform=test_transform
        )
        ds_train_eval = Subset(ds_train_eval_full, indices[val_size:])
        return ds_train, ds_val, ds_test, ds_train_eval
    else:
        return ds_train, ds_val, ds_test


def get_dataset(name, distillation=False):
    if name == "FlexibleMLP":
        return get_mnist_dataset()
    else:
        return get_cifar10_dataset(distillation)


class DistillCIFAR10FromSubset(Dataset):
    """
    Wrapper to create distillation dataset from existing subset and teacher logits.
    This handles the case where we have Subset objects instead of full datasets.
    """

    def __init__(self, subset_dataset, teacher_logits, transform=None):
        self.subset = subset_dataset
        self.teacher_logits = teacher_logits
        self.transform = transform

        # Verify lengths match
        if len(self.subset) != len(self.teacher_logits):
            raise ValueError(
                f"Subset length ({len(self.subset)}) doesn't match teacher_logits length ({len(self.teacher_logits)})"
            )

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get image and label from subset
        img, label = self.subset[idx]

        # Apply additional transform if provided
        if self.transform is not None:
            # Get the original image index from the subset
            original_idx = self.subset.indices[idx]

            # Access the raw CIFAR10 data directly
            # The structure is: Subset -> CIFAR10, not Subset -> Dataset -> CIFAR10
            raw_data = self.subset.dataset.data[original_idx]

            # Convert numpy array to PIL Image (CIFAR10 stores as HWC numpy arrays)
            if isinstance(raw_data, np.ndarray):
                raw_img = Image.fromarray(raw_data)
            else:
                raw_img = raw_data

            # Apply the transform
            img = self.transform(raw_img)

        # Get corresponding teacher logits
        teacher_logits = self.teacher_logits[idx]

        return img, label, teacher_logits
