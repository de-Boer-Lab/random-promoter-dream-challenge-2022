#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 ./submit.py \
  -o ./pred_bert.exprtop20.300bp.sample100.hidden512.cdhit95.nonfreeze \
  --test_path ./test_bert.exprtop20.300bp.sample100.hidden512.cdhit95.nonfreeze