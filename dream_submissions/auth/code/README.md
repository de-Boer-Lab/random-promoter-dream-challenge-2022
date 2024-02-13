# auth_dream_challenge_2022

I've shared the model on Google Drive with abdulmuntakimrafirafi@gmail.com. Here is the link: https://drive.google.com/file/d/1mQvb2v_HowUO-Yde8AegVhjgCWmrHmx2/view?usp=sharing

Predictions are available as 'test_predictions.txt'

The requested report is available as 'report.pdf'.

Instructions on how to run the code:

  1. Run edited.py. Running this srcipt generates 2 files ('seq_onehot.h5' & 'expression.h5') that will be used for training.
  2. Run train.py to train the model. The model is saved as 'model.h5'. 
  3. Run test_edited.py to process test sequences. The file 'test_seq_onehot.h5' is saved after running this script.
  4. Run test.py to test the model on the challenge's test data. The predictions are saved as 'test_predictions.txt'
