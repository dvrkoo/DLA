python3 main.py --finetune --wandb --epochs 10 --lr 2e-4 --batch_size 16 
python3 main.py --finetune --wandb --epochs 10 --lr 2e-5 --batch_size 16 

lora experiments
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-5 --batch_size 16 --lora_r 8 --lora_alpha 32
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-5 --batch_size 16 --lora_r 16 --lora_alpha 64

python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-4 --batch_size 16 --lora_r 8 --lora_alpha 32
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-4 --batch_size 16 --lora_r 16 --lora_alpha 64

# new dataset experiments
# 1. Run the SVM baseline on ag_news
# 1. Establish the baseline for sst2
python3 main.py --baseline --wandb --dataset_name sst2

# 2. Run your best full fine-tuning configuration
python3 main.py --finetune --wandb --dataset_name sst2 --epochs 5 --lr 2e-5 --batch_size 16

# 3. Run a comparable LoRA configuration
python3 main.py --finetune --wandb --peft --dataset_name sst2 --epochs 5 --lr 2e-5 --batch_size 16 --lora_r 16 --lora_alpha 32
# How does a very parameter-efficient LoRA perform?
python3 main.py --finetune --wandb --peft --dataset_name sst2 --epochs 5 --lr 2e-5 --batch_size 16 --lora_r 8 --lora_alpha 16
# Can we close the gap with full fine-tuning by giving LoRA more capacity?
python3 main.py --finetune --wandb --peft --dataset_name sst2 --epochs 5 --lr 2e-5 --batch_size 16 --lora_r 32 --lora_alpha 64
