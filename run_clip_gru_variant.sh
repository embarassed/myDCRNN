#!/bin/bash
#  exp1 2 3 
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=72 --clip=3
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=96 --clip=4
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=120 --clip=5
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=144 --clip=6
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=168 --clip=7 --batch_size=32 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=192 --clip=8 --batch_size=32 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=216 --clip=9 --batch_size=32 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=240 --clip=10 --batch_size=32 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=264 --clip=11 --batch_size=32 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python ltd_clip_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=288 --clip=12 --batch_size=32 --learning_rate=0.003