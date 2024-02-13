# Dream2022

## Predicting gene expression using millions of random promoter sequences


    .
    ├── dream_competition_2022.py                   # Model training code
    ├── prediction.py                               # Model testing code
    ├── model.py                                    # Model construction code
    ├── Dream Challenge Final Report.pdf            # Report
    ├── saved_model/                                # Saved pretrained model folder
    ├── data/                                       # training and testing data folder
    ├── submission/                                 # Saved prediction result for submission
    ├── requirements.txt                            # required python packages
    └── README.md



This repo contains code and report for the Dream Challenge submission

- To setup environment on TPU vm:
    
    1. initiate a tpu-vm with tensorflow 2.8.0 preinstalled:
     
            gcloud alpha compute tpus tpu-vm create tpu_v2 --zone=asia-east1-c --accelerator-type=v2-8 --version=tpu-vm-tf-2.8.0

    2. install supporting packages:

            pip install -r requirements.txt

- To train model (copy the train_sequences.txt under *"/data"* folder): 
    1. on tpu (about 4 hours to converge with tpuv2-8):
    
            python3 dream_competition_2022.py
    
        trained model will be saved on folder *"/saved_model"*


    **pre-trained model can be downloaded from one drive:[link](https://anu365-my.sharepoint.com/:f:/g/personal/u5541558_anu_edu_au/ElF5VcuH2BxLlveqfBcIxDsB6M361TuXoKfqIgMDos_c6w?e=HPLatp)**

- To predict pre-trained model on test sequence (copy the pre-trained model to the "saved_model/" folder):
    1. on tpu 

            python3 prediction.py
    
    2. on local gpu machine

            python3 prediction.py
        or 

            run the jupyter notebook on folder "/prediction_notebook"
        prediction output file *submission.txt* will be saved on folder *"/submission"*