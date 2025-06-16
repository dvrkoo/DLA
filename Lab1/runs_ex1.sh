# # wnorm vs no norm
# # 40
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 40 --hidden_size 128
# python3 exercise1.py --model FlexibleMLP --epochs 50 --depth 40 --hidden_size 128
# # 20
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 20 --hidden_size 128
# python3 exercise1.py --model FlexibleMLP --epochs 50 --depth 20 --hidden_size 128
# # 10
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 10 --hidden_size 128
# python3 exercise1.py --model FlexibleMLP --epochs 50 --depth 10 --hidden_size 128
#
# # residual vs non-residual
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 20 --hidden_size 128 --residual
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 20 --hidden_size 128
#
# # adding scheduler
# python3 exercise1.py --model FlexibleMLP --norm --epochs 50 --depth 20 --hidden_size 128 --scheduler
#
#
# # ~1M params each
# python3 exercise1.py --model FlexibleMLP --depth 5  --hidden_size 512
# python3 exercise1.py --model FlexibleMLP --depth 20 --hidden_size 128
# python3 exercise1.py --model FlexibleMLP --depth 40 --hidden_size 64
#

# 40 vs 20 vs 10



# CNN
python3 exercise1.py --model FlexibleCNN --layers 2 2 2 2
python3 exercise1.py --model FlexibleCNN --layers 2 2 2 2 --residual

python3 exercise1.py --model FlexibleCNN --layers 3 4 6 3
python3 exercise1.py --model FlexibleCNN --layers 3 4 6 3 --residual

python3 exercise1.py --model FlexibleCNN --layers 4 8 12 6
python3 exercise1.py --model FlexibleCNN --layers 4 8 12 6 --residual


