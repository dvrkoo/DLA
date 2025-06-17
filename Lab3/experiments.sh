# python3 main.py --finetune --wandb --epochs 10 --lr 2e-4 --batch_size 16 
# python3 main.py --finetune --wandb --epochs 10 --lr 2e-5 --batch_size 16 
#
# lora experiments
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-5 --batch_size 16 --lora_r 8 --lora_alpha 32
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-5 --batch_size 16 --lora_r 16 --lora_alpha 64

python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-4 --batch_size 16 --lora_r 8 --lora_alpha 32
python3 main.py --finetune --wandb --peft --epochs 10 --lr 2e-4 --batch_size 16 --lora_r 16 --lora_alpha 64

