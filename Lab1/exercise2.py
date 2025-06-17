import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataloader import get_dataset, DistillCIFAR10FromSubset, cifar_train_transform

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
    # Hard loss (standard cross-entropy)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft targets loss (knowledge distillation)
    p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    kl_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)

    # Combined loss
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return total_loss, ce_loss, kl_loss


# ----- Utility to precompute teacher logits -----
def compute_teacher_logits(model, dataloader, device):
    """Compute teacher logits for all training data"""
    model.eval()
    logits = []
    print("Computing teacher logits...")
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Teacher logits"):
            x = x.to(device)
            logits.append(model(x).cpu())
    return torch.cat(logits, dim=0)


# ----- Training loops -----
def train_teacher(model, loader, optimizer, device, epoch=None, prefix="teacher"):
    """Train teacher with proper tracking"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Training {prefix}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
        )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    # Log to wandb if epoch is provided
    if epoch is not None:
        wandb.log(
            {
                f"{prefix}/train_loss": avg_loss,
                f"{prefix}/train_acc": accuracy,
                "epoch": epoch,
            }
        )

    return avg_loss, accuracy


def train_student(student, loader, optimizer, device, args, epoch):
    """Train student with distillation"""
    student.train()
    running = {"total": 0, "ce": 0, "kl": 0}
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Distilling Epoch {epoch+1}")
    for x, y, tlog in pbar:
        x, y, tlog = x.to(device), y.to(device), tlog.to(device)

        optimizer.zero_grad()
        out = student(x)
        loss, ce_l, kl_l = distillation_loss(out, tlog, y, args.temperature, args.alpha)
        loss.backward()
        optimizer.step()

        # Track metrics
        running["total"] += loss.item()
        running["ce"] += ce_l.item()
        running["kl"] += kl_l.item()

        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {
                "Total": f"{loss.item():.4f}",
                "CE": f"{ce_l.item():.4f}",
                "KL": f"{kl_l.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%",
            }
        )

    # Calculate averages
    avg_metrics = {k: v / len(loader) for k, v in running.items()}
    train_acc = 100.0 * correct / total

    # Log to wandb
    wandb.log(
        {
            "student_distill/train_total_loss": avg_metrics["total"],
            "student_distill/train_ce_loss": avg_metrics["ce"],
            "student_distill/train_kl_loss": avg_metrics["kl"],
            "student_distill/train_acc": train_acc,
            "epoch": epoch,
        }
    )

    return avg_metrics, train_acc


def evaluate(model, loader, device, prefix, epoch=None):
    """Evaluate model with detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {prefix}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += F.cross_entropy(out, y).item()

            _, predicted = out.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            pbar.set_postfix({"Acc": f"{100.*correct/total:.2f}%"})

    accuracy = 100.0 * correct / total
    avg_loss = loss_sum / len(loader)

    print(f"{prefix} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    if epoch is not None:
        wandb.log(
            {
                f"{prefix}/test_loss": avg_loss,
                f"{prefix}/test_acc": accuracy,
                "epoch": epoch,
            }
        )

    return avg_loss, accuracy


def compare_models(teacher, student_baseline, student_distill, test_loader, device):
    """Compare all three models"""
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print("=" * 50)

    # Evaluate teacher
    teacher_loss, teacher_acc = evaluate(teacher, test_loader, device, "teacher_final")

    # Evaluate baseline student
    baseline_loss, baseline_acc = evaluate(
        student_baseline, test_loader, device, "student_baseline_final"
    )

    # Evaluate distilled student
    distill_loss, distill_acc = evaluate(
        student_distill, test_loader, device, "student_distill_final"
    )

    # Log comparison
    wandb.log(
        {
            "final/teacher_acc": teacher_acc,
            "final/student_baseline_acc": baseline_acc,
            "final/student_distill_acc": distill_acc,
            "final/distill_improvement": distill_acc - baseline_acc,
            "final/teacher_student_gap": teacher_acc - distill_acc,
        }
    )

    print(f"\nTeacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Student Distilled Accuracy: {distill_acc:.2f}%")
    print(f"Distillation Improvement: {distill_acc - baseline_acc:.2f}%")
    print(f"Teacher-Student Gap: {teacher_acc - distill_acc:.2f}%")


# ----- Main -----
def main(args):
    # WandB init with better config tracking
    config = vars(args)
    config.update(
        {
            "teacher_params": sum(
                p.numel()
                for p in FlexibleCNN(
                    3, 10, "basic", args.t_layers, args.use_skip, True
                ).parameters()
            ),
            "student_params": sum(
                p.numel()
                for p in FlexibleCNN(
                    3, 10, "basic", args.s_layers, False, True
                ).parameters()
            ),
        }
    )

    wandb.init(project="distillation_cifar10_improved", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_train, ds_val, ds_test, ds_train_eval = get_dataset("", distillation=True)

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    loader_train_eval = DataLoader(
        ds_train_eval, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    loader_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Teacher model
    print(
        f"\nTeacher architecture: {args.t_layers} layers, skip connections: {args.use_skip}"
    )
    teacher = FlexibleCNN(
        input_channels=3,
        num_classes=10,
        block_type="basic",
        layers=args.t_layers,
        use_skip=args.use_skip,
        use_batchnorm=True,
        zero_init_residual=True,
    ).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params:,}")

    # Train teacher
    print("\n" + "=" * 30 + " TRAINING TEACHER " + "=" * 30)
    opt_t = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_t = torch.optim.lr_scheduler.StepLR(
        opt_t, step_size=args.epochs // 3, gamma=0.1
    )

    best_teacher_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_teacher(
            teacher, loader_train, opt_t, device, epoch, "teacher"
        )
        test_loss, test_acc = evaluate(teacher, loader_test, device, "teacher", epoch)
        scheduler_t.step()

        if test_acc > best_teacher_acc:
            best_teacher_acc = test_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                teacher.state_dict(), os.path.join(args.save_dir, "teacher_best.pth")
            )

    # Load best teacher
    teacher.load_state_dict(torch.load(os.path.join(args.save_dir, "teacher_best.pth")))
    print(f"\nBest teacher accuracy: {best_teacher_acc:.2f}%")

    # Precompute teacher logits
    teacher_logits = compute_teacher_logits(teacher, loader_train_eval, device)

    # Student dataset for distillation

    ds_distill = DistillCIFAR10FromSubset(
        ds_train, teacher_logits, cifar_train_transform
    )
    loader_distill = DataLoader(
        ds_distill, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    # Student model
    print(f"\nStudent architecture: {args.s_layers} layers, no skip connections")
    student_baseline = FlexibleCNN(
        input_channels=3,
        num_classes=10,
        block_type="basic",
        layers=args.s_layers,
        use_skip=False,
        use_batchnorm=True,
    ).to(device)

    student_params = sum(p.numel() for p in student_baseline.parameters())
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.1f}x")

    # Train student baseline (without distillation)
    print("\n" + "=" * 30 + " TRAINING STUDENT BASELINE " + "=" * 30)
    opt_s_baseline = torch.optim.Adam(
        student_baseline.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler_s_baseline = torch.optim.lr_scheduler.StepLR(
        opt_s_baseline, step_size=args.epochs // 3, gamma=0.1
    )

    best_baseline_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_teacher(
            student_baseline,
            loader_train,
            opt_s_baseline,
            device,
            epoch,
            "student_baseline",
        )
        test_loss, test_acc = evaluate(
            student_baseline, loader_test, device, "student_baseline", epoch
        )
        scheduler_s_baseline.step()

        if test_acc > best_baseline_acc:
            best_baseline_acc = test_acc
            torch.save(
                student_baseline.state_dict(),
                os.path.join(args.save_dir, "student_baseline_best.pth"),
            )

    # Load best baseline
    student_baseline.load_state_dict(
        torch.load(os.path.join(args.save_dir, "student_baseline_best.pth"))
    )
    print(f"\nBest baseline student accuracy: {best_baseline_acc:.2f}%")

    # Create new student for distillation
    student_distill = FlexibleCNN(
        input_channels=3,
        num_classes=10,
        block_type="basic",
        layers=args.s_layers,
        use_skip=False,
        use_batchnorm=True,
    ).to(device)

    # Train student with distillation
    print("\n" + "=" * 30 + " TRAINING STUDENT WITH DISTILLATION " + "=" * 30)
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}")

    opt_s_distill = torch.optim.Adam(
        student_distill.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler_s_distill = torch.optim.lr_scheduler.StepLR(
        opt_s_distill, step_size=args.epochs // 3, gamma=0.1
    )

    best_distill_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_metrics, train_acc = train_student(
            student_distill, loader_distill, opt_s_distill, device, args, epoch
        )
        test_loss, test_acc = evaluate(
            student_distill, loader_test, device, "student_distill", epoch
        )
        scheduler_s_distill.step()

        if test_acc > best_distill_acc:
            best_distill_acc = test_acc
            torch.save(
                student_distill.state_dict(),
                os.path.join(args.save_dir, "student_distill_best.pth"),
            )

    # Load best distilled student
    student_distill.load_state_dict(
        torch.load(os.path.join(args.save_dir, "student_distill_best.pth"))
    )
    print(f"\nBest distilled student accuracy: {best_distill_acc:.2f}%")

    # Final comparison
    compare_models(teacher, student_baseline, student_distill, loader_test, device)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation on CIFAR-10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./distill_runs")
    parser.add_argument(
        "--epochs", type=int, default=70, help="Epochs for each training phase"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--t_layers",
        nargs="+",
        type=int,
        default=[2, 2, 2, 2],
        help="Teacher network layer counts per block",
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
    parser.add_argument(
        "--temperature", type=float, default=3.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for KL loss vs CE loss"
    )

    args = parser.parse_args()
    main(args)
