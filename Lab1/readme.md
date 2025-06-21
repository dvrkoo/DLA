# Deep Learning Architectures & Experiments

This repository contains code for two exercises exploring model design, training, and knowledge distillation on MNIST and CIFAR-10 datasets.

## 📂 Repository Structure

```
├── models/
│   ├── flexible_mlp.py       # FlexibleMLP implementation
│   ├── flexible_cnn.py       # FlexibleCNN implementation
│   └── custom_cnn.py         # Custom CNN variants (BasicBlock, Bottleneck)
├── dataloader.py             # Dataset loaders for MNIST & CIFAR-10
├── exercise1.py              # Main script for MLP/CNN experiments (training, eval, grads)
├── exercise2.py              # Main script for knowledge distillation on CIFAR-10
├── runs/                     # Output directory for model checkpoints & plots
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```


## 🏋️ Exercise 1: FlexibleMLP & FlexibleCNN Experiments

Train and evaluate MLPs and CNNs on MNIST/CIFAR-10 with various depths, widths, normalization, residual connections, and schedulers.

### MLP
MLP validation accuracy, please note that around 20 experiments were made, they can be seen at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/Lab1_DLA_MLP?nw=nwuserniccolomarini
![](plots/mlp.png)



### CNN

CNN validation accuracy, please note that around 20 experiments were made, they can be seen at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/Lab1_DLA_CNN?nw=nwuserniccolomarinis
![](plots/cnn.png)
### Usage

```bash
python3 exercise1.py --model FlexibleMLP \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 128 \
    --hidden_size 128 \
    --depth 20 \
    --residual \
    --norm \
    --scheduler
```

* **Arguments**:

  * `--model`: `FlexibleMLP` or `FlexibleCNN`
  * `--epochs`: # training epochs
  * `--lr`: learning rate
  * `--batch_size`
  * `--hidden_size`: for MLP only
  * `--depth`: # layers for MLP
  * `--layers`: list of 4 ints for CNN blocks
  * `--residual`: enable skip connections
  * `--norm`: enable BatchNorm
  * `--scheduler`: use cosine annealing LR

Results (check `runs/<model>/<run_name>/`) include:

* `model.pth` checkpoint
* `params.txt` hyperparameters
* `training_curves.png` loss/accuracy plots
* `gradient_norms.png` (for MLP)

## 🧪 Exercise 2: Knowledge Distillation on CIFAR-10

Experiment results can be seen at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/distillation_cifar10_improved?nw=nwuserniccolomarini
Distill a large (teacher) CNN into a smaller (student) one using Hinton et al. 2015.

### Usage

```bash
python3 exercise2.py \
    --data_root ./data \
    --save_dir ./distill_runs \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --t_layers 3 4 6 3 \
    --s_layers 1 1 1 1 \
    --use_skip \
    --temperature 3.0 \
    --alpha 0.5
```

* **Arguments**:

  * `--data_root`: data folder
  * `--save_dir`: output folder
  * `--epochs`, `--batch_size`, `--lr`
  * `--t_layers`: teacher block counts
  * `--s_layers`: student block counts
  * `--use_skip`: enable teacher residual
  * `--temperature`: distillation temp T
  * `--alpha`: weight for KL vs CE loss

Outputs in `distill_runs/`:

* `teacher.pth`, `student_baseline.pth`, `student_distill.pth`
* WandB logs for losses and accuracies
* Final comparison plots of teacher vs student performance

## 📈 Logging & Visualization

All experiments are instrumented with Weights & Biases (W\&B). Configure your API key:

```bash
wandb login
```

Metrics and model parameters are logged under the project names `Lab1_DLA_MLP`, `Lab1_DLA_CNN`, and `distillation_cifar10`.

## 🔧 Extending & Customizing

* **Add new optimizers**: modify `exercise1.py` or `exercise2.py` to support `--opt sgd|adam`.
* **Advanced schedulers**: integrate CyclicalLR, StepLR, or CosineAnnealingWarmRestarts.
* **Data augmentation**: enhance `dataloader.py` with transforms (CutMix, MixUp, etc.).
* **Hyperparameter sweeps**: use W\&B Sweeps to explore depth, width, α, T, etc.

## 📚 References

* Hinton, Vinyals & Dean. *Distilling the Knowledge in a Neural Network*, NeurIPS 2015.
* He, Zhang, Ren & Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016.
* PyTorch documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
