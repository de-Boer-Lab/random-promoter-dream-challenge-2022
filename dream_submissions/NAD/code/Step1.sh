#!/bin/bash
# Build glove
cd glove && make
cd ..
# Generate embedding for each token
python3 glove_embedding.py
cd glove && bash demo.sh
cd ..
cp ./glove/vectors.txt ./data/vectors.txt