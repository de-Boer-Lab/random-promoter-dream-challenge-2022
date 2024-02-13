#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2 ./train_expr_group.py \
	-f ../../data/train_sequences.fa.gz \
	-o ./tmp \
	--flank 150 \
	--batch_size 64 \
	--model_path ./bert.exprtop20.300bp.hidden512 \
	--sample 1.0
