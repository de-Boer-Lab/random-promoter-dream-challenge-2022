To replace the TF model from ChromBPNet with the DREAM models (or any other model), you'll need to clone the original [ChromBPNet repository](https://github.com/kundajelab/chrombpnet).

Follow the [ChromBPNet wiki](https://github.com/kundajelab/chrombpnet/wiki) to preprocess your data, train bias models, and train bias-factorized ChromBPNet. Once you are familiar with the chrombpnet repo, how data is being processed and saved at each step, how models are being trained and tested, if you want to replace the TF model of bias-factorized ChromBPNet with DREAM-RNN model you can run train_dream_rnn.py. 

Place train_modified.py (modified trainer function) under chrombpnet/training/ <br />
Place dream_rnn.py (DREAM-RNN architecture) under chrombpnet/training/models/ <br />
Place dream_rnn_with_bias_model.py (TF model (DREAM-RNN) combined with the bias model) under chrombpnet/training/models/ <br />
Place train_dream_rnn.py under the cloned repo

We will update this section if ChromBPNet repo starts supporting using different architectures for the TF model.