# Start with some standard imports.
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from models import SimpleMLP, ResidualMLP, SimpleCNN

# set seed
torch.manual_seed(0)

# Standard MNIST transform.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load MNIST train and test.
ds_train = MNIST(root="./data", train=True, download=True, transform=transform)
ds_test = MNIST(root="./data", train=False, download=True, transform=transform)

# Split train into train and validation.
val_size = 5000
I = np.random.permutation(len(ds_train))
ds_val = Subset(ds_train, I[:val_size])
ds_train = Subset(ds_train, I[val_size:])


# Simple function to plot the loss curve and validation accuracy.
def plot_validation_curves(losses_and_accs):
    losses = [x for (x, _) in losses_and_accs]
    accs = [x for (_, x) in losses_and_accs]
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Loss per Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}")


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(nin, nout)
                for (nin, nout) in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )

    def forward(self, x):
        return reduce(
            lambda f, g: lambda x: g(F.relu(f(x))), self.layers, lambda x: x.flatten(1)
        )(x)


"""Exercise 1.1"""
# Your code here.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, device=device):
    model.eval()
    predictions = []
    correct = 0
    running_loss = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
    avg_loss = loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy


def main():
    # Training hyperparameters.
    device = "mps"
    epochs = 10
    lr = 0.0001
    batch_size = 128

    # Architecture hyperparameters.
    input_size = 28 * 28
    hidden_size = 256
    num_classes = 10
    # width = 16
    # depth = 2

    # Dataloaders.
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size, shuffle=True, num_workers=4
    )
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size, shuffle=True, num_workers=4
    )

    # Instantiate model and optimizer.
    # model_mlp = SimpleMLP(input_size, num_classes, hidden_size, 4)
    # model_mlp = ResidualMLP(input_size, num_classes, hidden_size, 4)
    model_mlp = SimpleCNN(1, num_classes)
    model_mlp = model_mlp.to(device)
    opt = torch.optim.Adam(params=model_mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accs = []
    # Training loop.
    for epoch in range(epochs):
        loss = train_one_epoch(model_mlp, opt, criterion, dl_train, device=device)
        (val_loss, val_acc) = evaluate_model(
            model_mlp, dl_val, criterion, device=device
        )
        print(
            f"Epoch {epoch} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # And finally plot the curves.
    test_loss, test_acc = evaluate_model(model_mlp, dl_test, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
    # lets plot the loss and accuracy
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Loss per Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Best Accuracy = {np.max(val_accs)} @ epoch {np.argmax(val_accs)}")
    plt.show()


if __name__ == "__main__":
    main()
