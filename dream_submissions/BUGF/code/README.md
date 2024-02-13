# BUGF-DREAM22
Final submission package for DREAM 2022 Challenge

Software requirements:

1. PyTorch 1.12
2. Einops 0.4.1 (pip install einops)


STEP 1 - Training run:

python3 pytorch_xla_train.py &>> log &

This assumes that training_set.txt and validation_set.txt are in current directory.


STEP 2 - Rename final weights file ready for inference:

mv exp_train.pt FINAL_exp_train.pt


STEP 3 - Generating predictions for challenge sequences:

python3 pytorch_challenge.py test_sequences.txt > challenge.out

python3 pytorch_challenge.py training_seq.txt -q > trainresults.out

(adding -q skips the target sequence output to output just the predicted values)


NOTE: Larger files, including NN weights are available as a zip file on Google Drive: https://drive.google.com/file/d/1tPN9JMFE1WDMGD-VYJYvbdhMgpNA9jYl/view?usp=sharing
