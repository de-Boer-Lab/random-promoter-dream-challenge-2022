# *autosome.org* solution for DREAM 2022 Promoter Expression Challenge

## Environment
To ensure that the results of evaluating scripts explained below will deviate as little as possible from the results submitted to the leaderboard of the *DREAM 2022 Challenge*, we strongly advise using the **conda** environment provided in the `environment.yml` file. The environment can be initialized via
```
> conda env create -f environment.yml
```
And then activated with 
```
> conda activate dream_autosome
```
Please note that we trained the model on a GPU, and it is likely that the conda environment doesn't accommodate fully for particular software/hardware combinations e.g. when using TPU.

## Input data

Before you proceed, make sure that `test_sequences.txt` file is in the root directory of the repository. If you aim to re-train the model from scratch, make sure that the same applies for `train_sequences.txt` file, which is not provided with the repository due to file size limitations.

## Our submission

The 'submission' folder contains the predictions of our model that correspond to the following result from the leaderboard

|#|Competitor|Submission Time|Score PearsonR^2|Score Spearman|PearsonR^2|Spearman|
|-|----------|---------------|----------------|--------------|----------|--------|
|**1**|autosome.org|2022-07-31T19:39:12+03:00|0.829|0.860|0.952|0.979|

Namely, the files are `results.txt` and `results.json`. They differ only in format and the latter is the file that was actually uploaded to the leaderboard (here, we provide both variants for convenience).

## Test-time evaluation

To reproduce predictions *almost* exactly (up to computational arithmetic inconsistencies due to different GPUs) as they were present at the *DREAM* leaderboard,  run the `test.sh` bash script. The resulting predictions will be saved to the `results.txt` file in the root directory of the repository. If you want to obtain predictions in `json` format as it was required by the leaderboard system, change `--output_format tsv` and `--output results.txt` to `--output_format json` and `--output results.json`, respectively.

## Training

You can re-train the model with `train.sh` script. Chances are, that your results will differ from those obtained at our local machine, however, we do not expect a drastic difference in the resulting performance scores. Changing the training seed might allow to better evaluate the quality of our model:
- `--seed` -- the seed for pseudo-random numbers generator (we use 42 as a default);
- `--model_dir` -- directory where the trained models will be stored (here, the script will save models to `model_1`).

Note that the models are saved each epoch, i.e. given 80 training epochs, your final model of interest will be the last one (`model_80.pth`).

Predictions from newly trained models can be ontained by changing the following arguments in the `test.sh` script:
- `--output` -- a path to a file where predictions will be stored;
- `--model` -- a path to the saved model `.pth`-file.

## Troubleshooting

Depending on particular machine, the script may fail due to the system limit applied to the number of open files. 
Please consider setting `ulimit -n` to a higher value on the user- or system level, e.g.
```
> ulimit -n 1000000
```
