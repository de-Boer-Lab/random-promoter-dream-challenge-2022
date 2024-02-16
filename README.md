# Random Promoter Dream Challenge 2022

![alt text](images/random_promoter_dream_challenge_2022.png)

To address the lack of standardized evaluation and continual improvement of genomics models, we organized the [Random Promoter DREAM Challenge 2022](https://www.synapse.org/#!Synapse:syn28469146/wiki/617075). Here, we asked the participants to design sequence-to-expression models and train them on expression measurements of random promoter sequences. The models would receive regulatory DNA sequence as input and use it to predict the corresponding gene expression value. We designed a separate set of sequences to test the limits of the models and provide insight into model performance.

The top-performing solutions in the challenge exceeded performance of all previous state-of-the-art models for similar data. The top-performing models included features inspired by the nature of the experiment and the state-of-the-art models from computer vision and NLP, while incorporating novel training strategies that are better suited to genomics sequence data. To determine how individual design choices affect performance, we created a Prix Fixe framework that enables modular testing of individual model components, revealing further performance gains.

Finally, we benchmarked the DREAM models on Drosophila and human datasets, including predicting expression and open chromatin from DNA sequence, where they consistently surpassed existing state-of-the-art performances. Overall, we demonstrate that high-quality gold-standard genomics datasets can drive significant progress in model development.

## Setting Up Your Environment

The `environment.yml` file specifies all required libraries for this project. If you're managing virtual environments with Anaconda, you can directly install these dependencies using the command:

`conda env create -n dream -f environment.yml`