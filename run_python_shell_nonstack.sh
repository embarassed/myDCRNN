#!/bin/bash
# --------------------------------------seq_len=12 -----------------------------------------
# --------------FNN----------
#CUDA_VISIBLE_DEVICES=0 python fnn_train_lixiang.py --data='flow' --seq_len=12
# --------------GRU-----------
#CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12
# --------------K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# --------------------------------------seq_len=24 -----------------------------------------
# --------------FNN----------
#CUDA_VISIBLE_DEVICES=0 python fnn_train_lixiang.py --data='flow' --seq_len=24
# --------------GRU-----------
#CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=24
# --------------K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=24
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=24
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=24 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=24
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=24 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=24
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=24

# --------------------------------------seq_len=36 -----------------------------------------
# --------------FNN----------
#CUDA_VISIBLE_DEVICES=0 python fnn_train_lixiang.py --data='flow' --seq_len=36
# --------------GRU-----------
#CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=36
# --------------K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=36
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=36
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=36 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=36
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36
# --------------K=4 DCGRU-----------
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=36
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=36 --learning_rate=0.003
CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=36


# -----------------------------------------------More-----------------------------------------------------
# --------------K=5 DCFNN seq_len=12----------
#CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12
# --------------K=5 DCGRU-----------
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12

# --------------K=5 DCFNN seq_len=24----------
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=24
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=24
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=24
# --------------K=5 DCGRU-----------
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=24
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=24
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=24
# --------------K=5 DCFNN seq_len=36----------
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=36
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=36
# CUDA_VISIBLE_DEVICES=0 python dcfnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=36
# --------------K=5 DCGRU-----------
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=36
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=36
# CUDA_VISIBLE_DEVICES=0 python dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=36
