#!/bin/bash


./pretrain_expr.py \
  --fasta ../../data/train_sequences.fa.gz \
  --resume \
  --batch-size 128 \
  --device 0 \
  --num-hidden-layers 3 \
  --fraction-per-epoch 0.2 \
  --num-workers 3 \
  -lr 0.0001 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --flank 200 \
  -o bert.exprtop20.480bp.hidden512 \
  --top 0.20