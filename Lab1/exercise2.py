import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from tqdm import tqdm

from models import FlexibleCNN

torch.manual_seed(0)


# ----- Distillation loss -----
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    student_logits: [B, C]
    teacher_logits: [B, C]
    labels: [B]
    T: temperature
    alpha: weight for distillation loss
    """
    # Hard loss
    ce_loss = F.cross_entropy(student_logits, labels)
    # Soft targets loss
    p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    kl_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)
    return alpha * kl_loss + (1 - alpha) * ce_loss, ce_loss, kl_loss


# ----- Dataset that includes precomputed teacher logits -----
class DistillCIFAR10(Dataset):
    def __init__(self, root, split, teacher_logits, transform=None):
        self.data = datasets.CIFAR10(
            root, train=(split == "train"), download=True, transform=transform
        )
        self.teacher_logits = teacher_logits  # tensor [N, C]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        tlog = self.teacher_logits[idx]
        return img, label, tlog


# ----- Utility to precompute teacher logits -----
def compute_teacher_logits(model, dataloader, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits.append(model(x).cpu())
    return torch.cat(logits, dim=0)


# ----- Training loops -----
def train_teacher(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_student(student, loader, optimizer, device, args, epoch):
    student.train()
    running = {"total": 0, "ce": 0, "kl": 0}
    for x, y, tlog in loader:
        x, y, tlog = x.to(device), y.to(device), tlog.to(device)
        out = student(x)
        loss, ce_l, kl_l = distillation_loss(out, tlog, y, args.temperature, args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running["total"] += loss.item()
        running["ce"] += ce_l.item()
        running["kl"] += kl_l.item()
    wandb.log(
        {
            "train/total_loss": running["total"] / len(loader),
            "train/ce_loss": running["ce"] / len(loader),
            "train/kl_loss": running["kl"] / len(loader),
            "epoch": epoch,
        }
    )
    return running


def evaluate(model, loader, device, prefix, epoch=None):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += F.cross_entropy(out, y).item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    avg_loss = loss_sum / len(loader)
    if epoch is not None:
        wandb.log({f"{prefix}/loss": avg_loss, f"{prefix}/acc": acc, "epoch": epoch})
    return avg_loss, acc


# ----- Main -----
def main(args):
    # WandB init
    wandb.init(project="distillation_cifar10", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    ds_train = datasets.CIFAR10(
        args.data_root, train=True, download=True, transform=transform
    )
    ds_test = datasets.CIFAR10(
        args.data_root, train=False, download=True, transform=transform
    )
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size)

    # Teacher
    teacher = FlexibleCNN(
        input_channels=3,
        num_classes=10,
        block_type="basic",
        layers=args.t_layers,  # e.g. [3,4,6,3] for a ResNet‑34 style
        use_skip=args.use_skip,  # True to enable residuals
        use_batchnorm=True,
        zero_init_residual=True,
    ).to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    # Train teacher
    for e in range(args.epochs):
        print(f"Epoch {e+1}/{args.epochs} - Training Teacher")
        train_teacher(teacher, loader_train, opt_t, device)
    # Save teacher
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(teacher.state_dict(), os.path.join(args.save_dir, "teacher.pth"))

    # Precompute teacher logits
    teacher_logits = compute_teacher_logits(teacher, loader_train, device)

    # Student dataset
    ds_distill = DistillCIFAR10(args.data_root, "train", teacher_logits, transform)
    loader_distill = DataLoader(ds_distill, batch_size=args.batch_size, shuffle=True)

    # Small student

    student = FlexibleCNN(
        input_channels=3,
        num_classes=10,
        block_type="basic",
        layers=args.s_layers,  # e.g. [2,2,2,2] for a much smaller net
        use_skip=False,  # turn off skips to see impact
        use_batchnorm=True,
    ).to(device)

    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Train student normally
    for e in range(args.epochs):
        print(f"Epoch {e+1}/{args.epochs} - Training Student Baseline")
        train_teacher(student, loader_train, opt_s, device)
        evaluate(student, loader_test, device, prefix="student_baseline")

    # Re-init student for distillation
    baseline_path = os.path.join(args.save_dir, "student_norm.pth")
    torch.save(student.state_dict(), baseline_path)
    student.load_state_dict(
        torch.load(os.path.join(args.save_dir, "student_norm.pth"), map_location=device)
    )
    opt_s = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Train student via distillation
    for e in range(args.epochs):
        print(f"Epoch {e+1}/{args.epochs} - Training Student with Distillation")
        train_student(student, loader_distill, opt_s, device, args, e)
        evaluate(student, loader_test, device, prefix="student_distill", epoch=e)

    # Save final student
    torch.save(student.state_dict(), os.path.join(args.save_dir, "student_distill.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./distill_runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--t_layers",
        nargs="+",
        type=int,
        default=[2, 2, 2, 2],
        help="Teacher network layer counts per block (e.g. 3 4 6 3)",
    )
    parser.add_argument(
        "--s_layers",
        nargs="+",
        type=int,
        default=[1, 1, 1, 1],
        help="Student network layer counts (should be smaller)",
    )
    parser.add_argument(
        "--use_skip",
        action="store_true",
        help="Enable residual (skip) connections in the teacher",
    )
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()
    main(args)
