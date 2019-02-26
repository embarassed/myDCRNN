#!/bin/bash
# --------------------------------------seq_len=12 -----------------------------------------
# --------------FNN----------
#CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=12
# --------------GRU-----------
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=12

# --------------------------------------seq_len=24 -----------------------------------------
# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=24
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=24

# --------------------------------------seq_len=36 -----------------------------------------
# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=36
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=36

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=48
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=48

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=60
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=60

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=72
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=72

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=84
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=84

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=96
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=96

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=108 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=108

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=120 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=120

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=132 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=132

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=144 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=144

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=156 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=156

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=168 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=168

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=180 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=180

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=192 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=192

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=204 --batch_size=32
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=204

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=216 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=216

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=228 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=228 --batch_size=48 --learning_rate=0.003

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=240 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=240 --batch_size=48 --learning_rate=0.003

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=252 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=252 --batch_size=48 --learning_rate=0.003

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=264 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=264 --batch_size=48 --learning_rate=0.003

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=276 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=276 --batch_size=48 --learning_rate=0.003

# CUDA_VISIBLE_DEVICES=1 python ltd_fnn_train_lixiang.py --data='flow' --seq_len=288 --batch_size=16
CUDA_VISIBLE_DEVICES=1 python ltd_dcrnn_noseq2seq_train_lixiang.py --data='flow' --filter_type='dual_random_walk' --max_diffusion_step=0 --seq_len=288 --batch_size=48 --learning_rate=0.003
