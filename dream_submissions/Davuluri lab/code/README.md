# Report -- Davuluri lab

## 1. Report
The detailed report for creating training dataset and the detailed model description of our group can be found [here](https://github.com/Zhihan1996/Dream_2022/blob/master/DREAM%20Challenge%20Report_Davuluri_Lab.pdf). 



## 2. Model Training

#### 2.1 Download the DNABERT model

Please download the DNABERT model [here](https://drive.google.com/drive/folders/1AQUeVwDjDzA9Ft9YWrtsGaadlFwBGy-i?usp=sharing) and put it in the current directory.



#### 2.2 Download and Processed Data

For simplicity, we provide processed data. It is exactly the same as the provided one, but we silghtly modify the data format so that it can be fed with our model. We randomly select 100k samples as validation.

Please download the data [here](https://drive.google.com/drive/folders/1ucFFNwBWWxALtihKHt7roF3dTtjfVuVA?usp=sharing) and put it in `data/` folder.



#### 2.3 Setup environment

We do not support TPU training in the current code base. We use GPUs for model training.



```
conda create -n dream python=3.8
conda activate dream
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -y -c pytorch
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt
```





#### 2.4 Model Training

```
export MODEL_DIR=DNABERT/ # please set this as the absolute directory of the downloaded DNABERT model
export OUTPUT_DIR=output/
export DATA_DIR=data/

# we assume there are 4 GPUs available. If not, please change line 5 and 21 accordingly.
python -m torch.distributed.launch \
    --nproc_per_node 4 run.py \
    --dna_model_type dnakmer \
    --bert \
    --is_regression \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $MODEL_DIR \
    --train_file $DATA_DIR/train.csv \
    --validation_file $DATA_DIR/dev_10w.csv \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 118 \
    --output_dir output  \
    --use_fast_tokenizer \
    --save_steps 4000 \
    --warmup_steps 5000 \
    --logging_steps 300 \
    --ddp_find_unused_parameters True 
```



If the distributed training fails. Try the following. This might be slower.



```
export MODEL_DIR=DNABERT/ # please set this as the absolute directory of the downloaded DNABERT model
export OUTPUT_DIR=output/
export DATA_DIR=/home/zhihan/data/yeast/finetune/

# we assume there are 4 GPUs available. If not, please change line 5 and 21 accordingly.
python run.py \
    --dna_model_type dnakmer \
    --bert \
    --is_regression \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name $MODEL_DIR \
    --train_file $DATA_DIR/train.csv \
    --validation_file $DATA_DIR/dev_10w.csv \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 10000000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 118 \
    --output_dir output  \
    --use_fast_tokenizer \
    --save_steps 50000 \
    --warmup_steps 5000 \
    --logging_steps 500 \
    --ddp_find_unused_parameters True 
```





#### 2.5 Make Prediction for Test File

We expect a txt file as the test file, where each line contains one DNA sequence.



```
export TEST_FILE=data/sample_test.txt # please change this accordingly for real test

python make_prediction.py \
     --output_dir results/ \
     --model_dir output/ \
     --test_file $TEST_FILE
```





#### 2.6 The Model we Trained

To use the model we trained for evaluation, please download it from [here](https://drive.google.com/drive/folders/1Ik_As8zYss3AfnClAiIfW5zioi-uRhUx?usp=sharing). It can be used with the exactly same command as above by setting the `model_dir` as it directory.

