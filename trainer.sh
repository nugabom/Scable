#CUDA_VISIBLE_DEVICES=0 python3 train.py app:app/mbv2-local.yml
CUDA_VISIBLE_DEVICES=1 python3 train.py app:app/mbv2-global.yml
