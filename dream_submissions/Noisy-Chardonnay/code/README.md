### Submission entry for `Noisy Chardonnay` team

This repo contains the source code and instructions to reproduce our submission entry for the [DREAM Challenge 2022](https://www.iscb.org/rsgdream2022). We trained a CNN-BiLSTM neural network on expression levels measured in yeast across millions of random promoter sequences to predict expression levels for a held-out test set.

Final model and weights are saved in the `model` folder and can be readily used to make predictions given a one-hot encoded input sequence of 110 length.

You can reproduce the predictions for the test set by running the following command:

```python
python predict.py --model_path model/noisy_chardonnay_model.h5 --test_seq processed/test_seq.h5 --output model/predictions.txt
```

#### Design choices considered

We picked the final architecture after extensive experimentation across various layers, parameters and preprocessing strategies. While there are promising attribution methods to benchmark competing models (or architectures) for binary prediction tasks (e.g., motif discovery), it is very challenging to relate model parameters and design choices to meaningful biological features or representations for sequence-to-expression models. Indeed, we have not been able to identify a set of design choices that consistently improve prediction accuracy.

- [Koo and Ploenzke](https://doi.org/10.1038/s42256-020-00291-x) showed that using exponential activation in the first layer (and first layer alone) is helpful for better model interpretation. This strategy did not help with prediction accuracy in regression setting.

- Similar to the motivation for using exponential activation in the first layer (i.e., better discrimination between distributed vs local representation), we tried different pooling strategies. Notably, we tested whether concatanating `max pooling` and `average pooling` together in an adaptive fashion would help ([AdaptiveConcatPool2d](https://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600)). It did not. Examples where this strategy worked may be limited to certain image datasets and capture inherently different hierarchical representations.

- It is conceivable to think that most promoters favor one strand over the other. Therefore, we explored alternative ways to incorporate reverse complement sequences. Stacking training sequences and their reverse complement counterparts did not help training. As one of the design choices, we considered augmenting the training sequences by stacking their reverse complement sequences (instead of, for example, using a reverse complement aware convolutional layers to process forward and reverse strand separately). Stacking strategy did not improve the predictive accuracy as measured by validation loss during training; therefore, we trained our model without any reverse complement sequences included. 


- We leveraged the experiment tracking and hyperparameter tuning ('sweep') features on [Weights & Biases](https://wandb.ai) platform and selected the parameter combination with the lowest validation loss.

- #TODO One of the participants suggested that providing longer flanking sequences may be helpful. [Here](https://github.com/cx0/noisy-chardonnay/blob/main/data/flanking_sequences.fa) are the distal and proximal sequences around the 80-bp insert for future experimentation. We would like to use attention-based models to quantify the relative importance of distal and proximal regions.

- #TODO `k-mer` encoding instead of `one-hot` encoding has been shown to improve prediction accuracy in certain settings ([Gunasekaran et al 2021](https://doi.org/10.1155/2021/1835056)). Given the relatively short input sequences at hand, it is not clear if encoding choice would make a significant difference.

- #TODO We did not perform any test time augmentation. In principle, averaging the predictions for test sequences and their reverse complements may reduce prediction error, however, this requires accurate identification of subset of promoters with informative strand-specific behavior. In a two-stage experimental setup, deep learning models with particularly strong performance for motif discovery (or related binary tasks) can be leveraged for choosing the most informative reverse complement sequences.
