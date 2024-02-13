#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 ./test_expr.py \
  -f ../../data/test_sequences.fa.gz \
  -o ./test_bert.exprtop20.300bp.sample100.hidden512.clu50.nonfreeze \
  --flank 150 \
  --model_path ./train_bert.exprtop20.300bp.sample100.hidden512.clu50.nonfreeze \
  --batch_size 128
