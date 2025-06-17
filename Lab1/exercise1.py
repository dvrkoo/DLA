import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb

from models import FlexibleMLP, FlexibleCNN
from dataloader import get_dataset
from plot import gradient_norm_plot

# Device setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# Set random seed for reproducibility
torch.manual_seed(0)


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy


def plot_metrics(train_losses, val_losses, val_accs, save_dir):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


def get_model(
    input_channels,
    model_name,
    input_size,
    num_classes,
    hidden_size,
    depth=2,
    residual=False,
    norm=False,
    layers=[2, 2, 2, 2],
):
    if model_name == "FlexibleMLP":
        return FlexibleMLP(
            input_size, hidden_size, depth, residual=residual, use_batchnorm=norm
        )
    elif model_name == "FlexibleCNN":
        return FlexibleCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            block_type="basic",
            layers=layers,
            use_skip=residual,
            use_batchnorm=norm,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(args):
    ds_train, ds_val, ds_test = get_dataset(args.model)

    if args.model == "FlexibleMLP":
        run_name = f"{args.model}_lr{args.lr}_batch{args.batch_size}_hidden{args.hidden_size}_depth{args.depth}_residual{args.residual}_norm{args.norm}"
    else:
        run_name = f"{args.model}_lr{args.lr}_layers_{args.layers}_batch{args.batch_size}_residual{args.residual}_scheduler{args.scheduler}"

    save_dir = os.path.join("runs", args.model, run_name)
    os.makedirs(save_dir, exist_ok=True)
    project_name = "Lab1_DLA" + ("_MLP" if args.model == "FlexibleMLP" else "_CNN")

    wandb.init(project=project_name, name=run_name, config=vars(args))

    # Data loaders
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=4)
    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    input_size = 28 * 28 if args.model == "FlexibleMLP" else 32 * 32 * 3
    num_classes = len(dl_train.dataset.dataset.classes)
    input_channels = dl_train.dataset[0][0].shape[0]

    model = get_model(
        input_channels,
        args.model,
        input_size,
        num_classes,
        args.hidden_size,
        args.depth,
        args.residual,
        args.norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, dl_train, device)

        if args.scheduler:
            scheduler.step()

        val_loss, val_acc = evaluate_model(model, dl_val, criterion, device)

        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    test_loss, test_acc = evaluate_model(model, dl_test, criterion, device)
    print(f"Test Loss = {test_loss:.4f} | Test Accuracy = {test_acc:.4f}")

    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

    # Save model
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save plots and config
    if args.model == "FlexibleMLP":
        gradient_norm_plot(
            model, dl_train, device, args, f"{save_dir}/gradient_norms.png"
        )
    plot_metrics(train_losses, val_losses, val_accs, save_dir)
    with open(os.path.join(save_dir, "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["FlexibleMLP", "FlexibleCNN"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument(
        "--scheduler",
        action="store_true",
        default=False,
        help="Use learning rate scheduler",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 2, 2, 2])
    args = parser.parse_args()
    main(args)
