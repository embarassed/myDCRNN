#!/bin/bash
#  exp6
# flow, occ, speed
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ,speed' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# flow, speed
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,speed' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# flow, occ
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python ltd_dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

