# Camformers Final Submission
Final submission for the DREAM 2022 [Predicting gene expression using millions of random promoter sequences](https://www.synapse.org/#!Synapse:syn28469146/wiki/) challenge.

## Submission details
Final submission file is available at [Output/submission_file.csv](Output/submission_file.csv)
This submission was created using the process described below.

Some required files (including the final model parameters) are too large to upload, these can be found here:

https://content.cruk.cam.ac.uk/ghlab/Susanne/DREAM/final_rerun/

These files will be downloaded through the instructions described below.

## Run the model

The submission is fully reproducible using a TPU v3-8 tpu-vm-pt-1.11 with rdm_seed set to 42.
Change the rdm_seed parameter in [camformers_submission.py](https://github.com/FredrikSvenssonUK/camformers_submission/blob/main/script/camformers_submission.py#L48) to get different results.

### Create the TPU
```
#Creating the TPU:
gcloud alpha compute tpus tpu-vm create Camformers --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-1.11

# Connect to the TPU
gcloud alpha compute tpus tpu-vm ssh Camformers --zone=europe-west4-a
```

### Initital setup (Google Cloud TPU)
Perform the following commands on the TPU to set up for calculations.
```
# Download required packages
pip3 install scipy
pip3 install sklearn
pip3 install pandas

# Clone GitHub repository
git clone https://github.com/FredrikSvenssonUK/camformers_submission.git

# Go into folder
cd camformers_submission

# Download training sequences (too large for GitHub)
wget https://content.cruk.cam.ac.uk/ghlab/Susanne/DREAM/train_sequences.txt -O Data/train_sequences.txt

# Download the final model parameter file
wget https://content.cruk.cam.ac.uk/ghlab/Susanne/DREAM/final_rerun/trained_model.pt -O Models/trained_model.pt
```

### Train the model
The main script camformers_submission.py will read and pre-process the data, one-hot-encode, train the model, and output a submission file for the leader board.

#### Note that this will replace the current model file!
```
python3 script/camformers_submission.py
```
Expect this to take about 16 hours from end-to-end. Example output is provided in the Output folder, [stdout](Output/stdout.txt) and [stderr](Output/stderr.txt).

### Create final submission file from saved model
Use a trained model (loading the parameter file) to generate the final submission csv file containing all the sequences in the test set.
#### Note that this will replace the current submission file!
```
python3 script/predict_test.py
```

### Delete the TPU
```
# Delete the Camformers TPU
gcloud alpha compute tpus tpu-vm delete Camformers --zone=europe-west4-a

# List all TPUs to make sure it has been deleted
gcloud compute tpus tpu-vm list --zone=europe-west4-a
```

### Running on other platforms
If training on GPU or CPU is desired, comment out [xm.mark_step()](https://github.com/FredrikSvenssonUK/camformers_submission/blob/main/script/camformers_submission.py#L383) in the train function (two places).
The script will run preferentially on TPU>GPU>CPU if the device is detected. To force a CPU run, use `--cpu`.

# Authors

* Susanne Bornelöv - [susbo](https://github.com/susbo)
* Fredrik Svensson - [FredrikSvenssonUK](https://github.com/FredrikSvenssonUK)
* Maria-Anna Trapotsi - [marianna-trapotsi](https://github.com/marianna-trapotsi)

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
