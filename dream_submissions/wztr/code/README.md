# DREAM-challenge-2022  (Team Name: wztr)
* ```final_prediction.txt``` is our prediction for test data.
* ```final_report.pdf``` is our final report.
* ```best_checkpoint.ckpt``` is the model checkpoint that is used to make prediction for test data
* If you want to retrain our model, follow the instructions below:

## Setup the environment

Install dependencies

```bash
conda env create --name seq2exp --file environment.yaml
```

Install the seq2exp package we will use for this project.

```bash
conda activate seq2exp
pip install -e .
```

## Download Data

* Create a new folder ```data``` under the root directory. 
* Download two data files ```train_sequences.txt``` and ```test_sequences.txt``` under the ```data``` folder from the competition website https://www.synapse.org/#!Synapse:syn28469146/files/.


## Train Model

* Run ```scripts/train_model.py``` in terminal: ```python scripts/train_model.py >> output.out```
* After training is done, a txt file ```final_prediction_2.txt``` will be generated under root directory.
