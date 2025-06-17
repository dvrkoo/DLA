import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
import wandb


def gradient_norm_plot(model, dataloader, device, args, save_path):
    data, labels = next(iter(dataloader))
    data, labels = data.to(device), labels.to(device)
    model.zero_grad()
    loss = F.cross_entropy(model(data), labels)
    loss.backward()

    # collect norms
    sums, counts = defaultdict(float), defaultdict(int)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # assume "layers.{i}.weight" or "layers.{i}.bias"
        idx = int(name.split(".")[1])
        sums[idx] += p.grad.norm().item()
        counts[idx] += 1

    avg_norms = [sums[i] / counts[i] for i in range(len(sums))]

    # plot
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(avg_norms)), avg_norms)
    plt.xlabel("Block index")
    plt.ylabel("Avg gradient L2 norm")
    plt.title("Gradient norms per depth level")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    wandb.log({"grad_norms": wandb.Image(save_path)})
