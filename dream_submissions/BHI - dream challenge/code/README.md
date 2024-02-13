# BHI-DREAM-Challenge
Training/inference pipeline submission for DREAM Challenge 2022 **Predicting gene expression using millions of random promoter sequences** by team BHI.

## Overview
```
├── ckpts                      <- Model weights
│   ├── our_trained.ckpt             <- Weights of our trained model
│   ├── (~~~~.ckpt)                  <- When you reproduce our model, we recommend you to save additional ckpt files in this path
│
├── data                       <-  Project data
│   ├── (test_sequences.txt)         <- Please save data
│   ├── (train_sequence.txt)         <- Please save data
│
├── prediction                 <-  Prediction files
│   ├── Final_Predictions.txt                 <- Predicted results for final submission in tab-delimited text format
│   ├── LB_final_submission_9045.json         <- Predicted results for leaderboard submission in json file 
│   ├── Predictions_before_Postprocess.txt    <- Predicted results before postprocessing, in tab-delimited text format
│   ├── sample_submission.json                <- Sample submission file used in LB
│   ├── (~~~~.txt)                            <- If you make other prediction files with reproduced models, it is recommended to save here
│
├── src                        <- Source code
│   ├── data.py                      <- Torch dataset
│   ├── infer.py                     <- Run inference on test data and produce submission json file
│   ├── infer_txt.py                 <- Run inference on test data and produce tab-delimited text file
│   ├── net.py                       <- Our model is implemented in torch.nn.Module
│   ├── postprocess.py               <- Run post-processing of prediction values for submission json file
│   ├── postprocess_9045.py          <- Run post-processing of prediction values for submission json file, for only 9045 sequences (as sample_submission.json)
│   ├── postprocess_txt.py           <- Run post-processing of prediction values for tab-delimited prediction file
│   ├── train_full.py                <- Run training with whole train data
│   ├── train.py                     <- Run training for cross-validation
│   └── util.py                      <- Utility scripts
│
├── .gitignore                       <- List of files ignored by git
├── environment.yaml                 <- Conda environment file
├── report.pdf                       <- Final report file
└── README.md
```

## Predicted results for final submission
As requested, we provide the 71,103 test sequences with their final expression predictions in `prediction/Final_Predictions.txt` as a tab-delimited text format.

## Requirements

- CUDA=11.2
- python=3.9.12
- cuDNN=8.2.0

*Please note that the reproducibility of the pipeline is tested only for GPU setting since we observed drastic slowdown of the training when TPUs (with PyTorch XLA) were used. This may be due to an uncharted issue of PyTorch XLA for the TPU-compatibility of RNN implementation, but we were not sure what the exact problem is, so we decided to develop and finalize our model in GPU setting.*

## How to run

### Clone the repo

```
git clone git@github.com:Sangyeup/BHI-DREAM-Challenge.git && cd BHI-DREAM-Challenge
```

### Set up conda environment
This will create conda environment named `DREAM2022`.
```
conda env create --file environment.yaml --prefix /your_prefix/
```

### Activate conda environment
```
conda activate DREAM2022
```

### Training (cross validation)
*NOTE:* Training requires `train_sequences.txt` file properly saved in `data` directory.
```
python src/train.py \
-m DeepGXP -e {EXP_NAME} -b 512 \
--optimizer AdamW --init-lr 0.0015 --weight-decay 0.01 --loss Huber \
--num-epochs 15 --fold {FOLD} --seed 40 \
--output ckpts/{CKPT_NAME}.ckpt
```

### Training (with whole training data)
*NOTE:* Training requires `train_sequences.txt` file properly saved in `data` directory.
```
python src/train_full.py \
-m DeepGXP -e {EXP_NAME} -b 512 \
--optimizer AdamW --init-lr 0.0015 --weight-decay 0.01 --loss Huber \
--num-epochs 15 --seed 40 \
--output ckpts/{CKPT_NAME}.ckpt
```

### Inference on test data
*NOTE:* Inference requires `test_sequences.txt` file properly saved in `data` directory.
```
python src/infer_txt.py \
-i ckpts/{CKPT_PATH}.ckpt -s 42 -m DeepGXP --tta 5 --output prediction/{FILE_NAME}.txt
```

### Postprocessing
```
python src/postprocess_txt.py -i prediction/{INPUT_FILE}.txt -o prediction/{OUTPUT_FILE}.txt
```

## Contributors

If you have any inquiries or there are any problems reproducing the training/inference pipeline, please contact any of our team members:

- [Dohoon Lee](https://github.com/dohlee) (apap7@snu.ac.kr, dohlee.bioinfo@gmail.com)
- [Danyeong Lee](https://github.com/DanYeong-Lee) (ldy9381@snu.ac.kr)
- [Dohyeon Kim](https://github.com/dohyeon-scott-kim) (scottkdh@snu.ac.kr)
- [Nayeon Kim](https://github.com/ny1031) (ny_1031@snu.ac.kr)
- [Sangyeup Kim](https://github.com/Sangyeup) (sang2668@snu.ac.kr)
- [Yeojin Shin](https://github.com/syjssj95) (syjssj95@snu.ac.kr)
