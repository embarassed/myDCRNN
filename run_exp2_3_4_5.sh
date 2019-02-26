#!/bin/bash
#  exp2 3 
# --------------K=1 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=1 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# --------------K=2 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=2 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# --------------K=3 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=3 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12 --learning_rate=0.003
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# --------------K=4 DCFNN----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
# --------------K=4 DCGRU-----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
CUDA_VISIBLE_DEVICES=1 python ltd_vdcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12 --learning_rate=0.003
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# --------------K=5 DCFNN----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcfnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12
# # --------------K=5 DCGRU-----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=5 --seq_len=12 --learning_rate=0.003
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=5 --seq_len=12 --learning_rate=0.003
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=5 --seq_len=12 --learning_rate=0.003




#######################################################################################################################################################
# exp 4 5
#-----------------------------------------------newgraph-------------------------------------------------------------
# --------------K=1 DCGRU-----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=1 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=1 --seq_len=12
# # --------------K=2 DCGRU-----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=2 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=2 --seq_len=12
# # --------------K=3 DCGRU-----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=3 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=3 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=3 --seq_len=12
# # --------------K=4 DCGRU-----------
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_stack_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12

# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=4 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='random_walk' --max_diffusion_step=4 --seq_len=12
# CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_newgraph_train_lixiang.py --data='flow' --filter_type='laplacian' --max_diffusion_step=4 --seq_len=12
