#!/bin/bash
# --------------------------------------seq_len=12 -----------------------------------------
# --------------FNN----------
CUDA_VISIBLE_DEVICES=1 python fnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --seq_len=12
# --------------GRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
# --------------stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
# --------------non-stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=2 DCGRU-----------
#CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12


# --------------------------------------seq_len=24 -----------------------------------------
# --------------FNN----------
CUDA_VISIBLE_DEVICES=1 python fnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --seq_len=24
# --------------GRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=24
# --------------stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24
# --------------non-stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24


# --------------------------------------seq_len=36 -----------------------------------------
# --------------FNN----------
CUDA_VISIBLE_DEVICES=1 python fnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --seq_len=36
# --------------GRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=36
# --------------stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36
# --------------non-stack: K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcfnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36
# --------------K=4 DCGRU-----------
#CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=1 python dcrnn_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36





# -----------------------------------------------More-----------------------------------------------------
# --------------K=5 DCFNN seq_len=12----------
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12
# --------------K=5 DCGRU-----------
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12

# --------------K=5 DCFNN seq_len=24----------
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=24
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=24
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=24
# --------------K=5 DCGRU-----------
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=24
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=24
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=24
# --------------K=5 DCFNN seq_len=36----------
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=36
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=36
#CUDA_VISIBLE_DEVICES=1 python dcfnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=36
# --------------K=5 DCGRU-----------
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=36
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=36
#CUDA_VISIBLE_DEVICES=1 python dcrnn_stack_flow_occ_speed_train_lixiang.py --data='flow,occ' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=36
