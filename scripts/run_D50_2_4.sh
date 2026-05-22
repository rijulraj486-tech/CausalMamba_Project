#!/bin/bash
cd ../source_code

CUDA_VISIBLE_DEVICES=0 python train_pure_bimamba.py \
  --model causal_mamba \
  --data_path ../data/nclt_1hz.pt \
  --d_model 96 \
  --n_layers 2 \
  --seed 42 \
  --train_samples_per_traj 64 \
  --gps_filter none \
  --gps_fusion_mode fixed \
  --gps_anchor_alpha 0.03 \
  --eval_gps_anchor_alpha 0.03 \
  --D 50 \
  --k 2 \
  --w 4 \
  --lr 2e-5 \
  --epochs 30 \
  --gps_dropout 0.40 \
  --output_dir ../evaluation_results/D50_2_4
