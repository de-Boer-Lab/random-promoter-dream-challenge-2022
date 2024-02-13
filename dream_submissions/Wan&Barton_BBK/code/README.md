# Introduction
This repository contains the implementation of predicting gene expression using millions of random promoter sequences for the Dream challenge 2022. The implementaion provides data preprocessing, data loaders, model building and training. The character-level Temporal Convolutional Network (TCN) described in https://arxiv.org/abs/1803.01271 is adopted.

# Usage
The impelementation is customised to run on a TPU virtual machine (tpu-vm) with PyTorch/XLA, it was tested on a tpu-vm v2-8 with Pytorch version 1.11. To run this impelementation on GCP yon need to do the following steps:

1\. Create a tpu-vm instance with Pytorch version 1.11 installed

`gcloud compute tpus tpu-vm create [INSTANCE NAME]
--zone=[SELECTED ZONE]
--accelerator-type=v2-8
--version=tpu-vm-pt-1.11`

2\. SSH to the tpu-vm instance

`gcloud compute tpus tpu-vm ssh [INSTANCE NAME]
  --zone [SELECTED ZONE]`
  
3\. Install dependencies using the given `requirements.txt` file

`pip3 install -r requirements.txt`

4\. Copy the dataset and the code from your GCP bucket to the instance

`gsutil -m cp gs://[BUCKET NAKE]/[FOLDER NAME]/* .`

Assuming all required files are in a single folder.

5\. Run `train.py` to train the model using all train sequences.

`python3 train.py`

`train.py` contains data preprocessing and model training, for data preprocessing details please see the report.

# Submission files
To create the final submission file `submission.txt`, run `python3 Final_submission.py`. The final saved model should be renamed to `model.pt` before generating the required file. Regarding the final predictions, we don’t specify weights for different evaluation metrics.

# Acknowledgements
We would like to thank DREAM Challenges for organising this competition, and Google Research for providing TPU resources to make training the model possible. In addition, we would like to thank the support by the Department of Computer Science and Information System and the Birkbeck GTA programme.
