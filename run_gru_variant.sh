#!/bin/bash
#  exp1 2 3 
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=48
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=60
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=72
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=84
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=96

# # --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

# # --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

# # --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

# # --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_vdcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
