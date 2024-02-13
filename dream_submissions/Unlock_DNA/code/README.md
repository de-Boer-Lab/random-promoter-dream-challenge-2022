# DREAM Challenge 2022
## Predicting gene expression using millions of random promoter sequences by Team `Unlock_DNA`

* Model checkpoint

`https://s3.msi.umn.edu/gongx030/projects/dream_PGE/notebooks_msi/m20220727e/tf_ckpts.tar`

* Notebook for converting training sequences to a `tf.data` object

`https://github.com/gongx030/dream_PGE/blob/main/prepare_tfdatasets.ipynb`

* Notebook for model training and prediction

`https://github.com/gongx030/dream_PGE/blob/main/mode_training.ipynb`

* The Conda environment file:

`https://github.com/gongx030/dream_PGE/blob/main/tf26_py37_a100.yml`

* The JSON file for prediction:

`https://s3.msi.umn.edu/gongx030/projects/dream_PGE/predictions/m20220727e/pred.json`

* The tsv file for prediction:

`https://s3.msi.umn.edu/gongx030/projects/dream_PGE/predictions/m20220727e/pred.tsv`

* Final report

`https://github.com/gongx030/dream_PGE/blob/main/report.pdf`

## The guide to training the model

1. Setup the hardware and the conda environment accroding according to the [yml](https://github.com/gongx030/dream_PGE/blob/main/tf26_py37_a100.yml) file. 
2. Run notebook [prepare_tfdatasets.ipynb](https://github.com/gongx030/dream_PGE/blob/main/prepare_tfdatasets.ipynb) to generate a `tf.data` file for all training data. The resulting `tf.data` file can be found at `./s3.msi.umn.edu/gongx030/projects/dream_PGE/training_data/pct_ds=1/`. 
3. Run notebook [mode_training.ipynb](https://github.com/gongx030/dream_PGE/blob/main/mode_training.ipynb) to train the model on the training data and make predictions on the testing data. The model was original trained on a machine with 4 A100 GPU with cuda version of 11.7. 
4. The checkpoint should be found at `./s3.msi.umn.edu/gongx030/projects/dream_PGE/predictions/m20220727e/tf_ckpts`. 
5. The final output file should be found at `./s3.msi.umn.edu/gongx030/projects/dream_PGE/predictions/m20220727e/pred.tsv`. 

