# DREAM 2022: Predicting gene expression using millions of random promoter sequences

[![Snakemake](https://img.shields.io/badge/snakemake-≥7.8.3-brightgreen.svg)](https://snakemake.bitbucket.io)

## Table of contents

- [DREAM 2022: Predicting gene expression using millions of random promoter sequences](#dream-2022-predicting-gene-expression-using-millions-of-random-promoter-sequences)
    - [Table of contents](#table-of-contents)
    - [Abstract](#abstract)
    - [Authors](#authors)
    - [Workflow usage](#workflow-usage)
    - [Manual usage](#manual-usage)
    - [Final model](#final-model)
    - [Download prediction files](#download-prediction-files)
    - [Contact](#contact)

## Abstract

Understanding gene regulation is an important step towards the interpretation of sequence alterations causing disease. The regulatory activity can be measured through functional assays but are limited in their throughput given the large universe of hundreds of millions potential variants. Therefore, predictive models are built that tries to learn the regulatory property of a DNA sequence. Here we present a simple convolutional neural network that can be trained with expression data of high-throughput assay and predict the potential activity of a short sequence. Because such experimental data is biased towards GC content, we implemented a unique GC correction step on the training data so that the model can focus its decisions on motifs within the sequence rather than the general nucleotide composition. This makes the model more interpretable towards learned motifs and enables the ability to measure sequence alterations, a step forward understanding diseases.

## Authors

* Max Schubach (@visze), Pyaree Mohan Dash (@pyareedash), Sebastian Röner (@sroener) , Berlin Institute of Health (BIH), [Computational Genome Biology](https://kircherlab.bihealth.org)

---------------------------------------------------

## Workflow usage

The following section describes the usage of the provided snakemake workflow to reproduce the training of the final model.

### Step 1: Obtain a copy of this workflow

[Clone](https://help.github.com/en/articles/cloning-a-repository) this repository to your local system, into the place where you want to perform the data analysis.

Once cloned, it is recommended to perform following steps from the main directory of this repository i.e. ```DREAM_predicting_gene_expression/```.

### Step 2: Obtain additional resources

Install Synapse using pip:

```bash
pip install synapseclient
```

1. Download the training data:

```bash
synapse get -r syn28469146 --downloadLocation resources/data_synapse/
```

### Step 3: Configure workflow

Configure the workflow according to your needs via editing `config/config.yaml`. For reproducing the training of our final model, only the `synapse_path` variable needs to change to the location of the synapse data (default: `resources/synapse_data`).

### Step 4: Install Snakemake

Install Snakemake using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

    conda create -c bioconda -c conda-forge -n snakemake snakemake

For installation details, see the [instructions in the Snakemake documentation](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

### Step 5: Execute workflow

Activate the conda environment:

    conda activate snakemake

Test your configuration by performing a dry-run via

    snakemake --use-conda --configfile config/config.yaml -n

Execute the workflow locally via

    snakemake --use-conda --cores $N

using `$N` cores or run it in a cluster environment via

    snakemake --use-conda --cluster qsub --jobs 10

or

    snakemake --use-conda --drmaa --jobs 10

See the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for further details on cluster execution.

**Note:** The training procedure expects one or more GPUs.

### Step 5: Investigate results

After successful execution, you find the all intermediate und final files in the `results/` directory.

## Manual usage

### Create environment

Before running the scripts, install the required packages via conda/mamba:

```bash
conda env create -n DREAM2022 -f environment.yaml
```

Activate the conda environment:

```bash
conda activate DREAM2022
```

### Script usage

To use our pre-trained model for prediction with a sequence file (any file with only one column containing sequences), use the following command and substitute `<sequence_to_test>` (e.g. resources/data_synapse/test_sequences.txt) and `<sequence_output.tsv.gz>` (e.g. test_sequences_predictions.tsv.gz):

```bash
python src/predict.py \
--test <sequence_to_test> \
--SavedModel final_model/gc_frac08_standard_classification \
--no-adapter-trimming \
--output <sequence_output.tsv.gz>
```

or using the separate model files:

```bash
python src/predict.py \
--test <sequence_to_test> \
--model final_model/gc_frac08.standard_classification.json \
--weights final_model/gc_frac08.standard_classification.h5 \
--no-adapter-trimming \
--output <sequence_output.tsv.gz> 
```

For more help, please check: ```python src/predict.py --help```

To compute performace (pearson and spearman correlation), use the following command and substitute `<original_expression.data>` with the data you want to correlate with (e.g. train/validate/test data created in our workflow):

```bash
python src/correlation.py \
--model-mode standard \
--original <original_expression.data> \
--method weightedmean \
--output <correlations.tsv>
```

For more help, please check: ```python src/correlation.py --help```

To create a submission file in *.json* format (for leaderboard submission), use this command and substitue <predicted_expression_file> (e.g. predictions on resources/data_synapse/test_sequences.txt) and <original_expression_file> (e.g. resources/data_synapse/test_sequences.txt) :

```bash
python src/create_submission.py \
--model-mode standard \
--predicted <predicted_expression_file.tsv.gz> \
--original <original_expression_file> \
--method weightedmean \
--output <final_predictions_file.tsv.gz> \
--sample-submission resources/sample_submission.json <leaderboard_submission.json>
```

For more help, please check ```python src/create_submission.py --help```

## Final model

To use our final (best) model, all neccesary files are located in ```final_model/``` folder. Please check [```final_model/README.md```](https://github.com/kircherlab/DREAM_predicting_gene_expression/tree/final_submission/final_model#final-model).

## Download prediction files

Predictions made with our final model can be downloaded from [here](https://muntakimrafi:uqFwTp5K5ovYtYvn@kircherlab.bihealth.org/download/DREAM2022/). The URL already contains the access credentials.

## Contact

If questions arise, feel free to contact the authors on Synapse, GitHub, Gitter or via email.
