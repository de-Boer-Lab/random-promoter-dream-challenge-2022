# DREAM random promoter expression prediction 

Please see our final report `DREAM_promoter_2022_Peppa.docx` or `DREAM_promoter_2022_Peppa.pdf`. 

Please find our final submission file `final_submission.txt`

This README file mainly aims to provide an introduction on how to run our code.

## Aim

The input data is `train_sequences.txt`. (Please download it yourself from the DREAM challenge website.) It is a two-col tsv file where the first column is the random promoter sequences, most of them are 110bp and the first 18bp and last 15bp are always the same. The second column is the observed expression value.

The goal is to create a deep learning model that can predict the gene expression given any DNA sequences.

The competition is to see which team performs the best on the testing data set `test_sequences.txt`.

User submit the prediction as a json file in the leaderboard phase and submit the prediction as a tsv file in the final phase. Not sure why they change the format.

## Model

Our model is entirely based on the Enformer model from DeepMind. We were able to get 0.717 weighted pearsonR score (final evaluation score by the competition) in the leaderboard. We made some modifications to reduce the complexity and the number of parameters has been reduced to less than 2M. See our model structure below:

![m](enform_XS.png)

## Steps

#### 1. One-hot encoding of DNA sequences (~10h)

```
python Step1_generate_DNA_matrix.py

```

This script assumes the input files `train_sequences.txt` and `test_sequences.txt` are in the current dir. It outputs `all_train_data.h5` and `test_sequences.h5`.

#### 2. Train the enformer model and evaluate the model. (several days)

To skip step1, download the h5 file from https://drive.google.com/file/d/1U2u9QPAQViT-bu9tq2VRCwgPBT7UC0qU/view?usp=sharing

```
python Step2_train_eval.py

```

This script takes in `all_train_data.h5` and split it into 10K evaluation and the rest for training. Data augmention is based on +/- 1bp shifted sequences.

This script also takes in `test_sequences.h5` and when `pearson_eval>0.73`, it will make predictions on the testing set and generate the json file for leaderboard submission.

#### 3. Generate final prediction based on the final submission format (5 min)

```
python final_submission.py epoch_14_model

```

The final submission is a tsv file where the first column is the original DNA sequence and the second column is the prediction. So this script first reads in `test_sequences.h5`, make prediction, and then append this prediction to the sequences in `test_sequences.txt`

Unfortunately, for my final submission, I have to submit the results from my very early model, `epoch_14_model`, (this one has 0.711 pearsonR score in leaderboard) because in the best model (`final_model`), I forgot to change the input signature in the original Enformer code. Default required input size is `(None, 196608, 4)`, but mine is `(None, 85, 4)`, making this script (`final_submission.py`) fail. Supposely, I could just use the trained variables from this final model and make a new model based on these variables, but I couldn't get it to work, mainly because Enformer is using the `Sonnet` library and I'm not familiar with it. Also, I just found this issue today 8/7/2022, there is no time for me to retrain and save the correct model, but if you run `python Step2_train_eval.py`. You should be able to get better results than my old model, epoch_14_model.

## How I set up the conda env

```
module load conda3/202011
conda create -n keras python=3.9
module load cuda11/toolkit/11.3
module load cudnn/8.2.0.53
nvidia-smi # check GPU usage
echo $CONDA_PREFIX # check conda dir

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

pip install tensorflow
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda install -c conda-forge jupyterlab
conda install -c conda-forge swifter
conda install -c anaconda seaborn
# run jupyter lab

export XDG_RUNTIME_DIR=""
mkdir /scratch_space/yli11/68985
export JUPYTER_RUNTIME_DIR=/scratch_space/yli11/68985
jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --no-browser --port 44444



# install janggu
# pysam install has an error in my system, so I used conda to install it
pip install --no-deps janggu[tf2_gpu]
# install those dependences manually
pip install biopython==1.77

vim ~/.conda/envs/keras/lib/python3.9/site-packages/janggu/layers.py
# change from keras.layers.wrappers import Wrapper to
# from tensorflow.keras.layers import Wrapper

```
