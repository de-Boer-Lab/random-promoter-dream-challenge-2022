################################
#### Global functions       ####
################################
from snakemake.workflow import srcdir

INPUT_DATA_SCRIPTS_DIR = srcdir("../scripts")
TRAINING_SCRIPTS_DIR = srcdir("../../src")

DATA_DIR = srcdir("../../" + config["synapse_path"])

def getInputDataScript(name):
    return "%s/%s" % (INPUT_DATA_SCRIPTS_DIR, name)


def getTrainingScript(name):
    return "%s/%s" % (TRAINING_SCRIPTS_DIR, name)

from snakemake.utils import validate
import pandas as pd


# this container defines the underlying OS for each job when using the workflow
# with --use-conda --use-singularity
container: "docker://continuumio/miniconda3"


##### load config and sample sheets #####

# preferrred to use --configfile instead of hard-coded config file
# configfile: "config/config.yaml"

validate(config, schema="../schemas/config.schema.yaml")


def getRawTrainDatafile():
    return "%s/%s" % (DATA_DIR,config["train_sequences"])

def getRawTestDatafile():
    return "%s/%s" % (DATA_DIR,config["test_sequences"])

def getSeed():
    return "--seed " + str(config["seed"])

def getFitSequence(training):
    if "fit_sequence" in config["trainings"][training]:
        return "--fit_sequence"
    else:
        return ""
        
def useGCCorrection(training):
    return config["trainings"][training]["gc_correction"]


def useReplacement(training):
    return config["trainings"][training]["replacement"]


def getBucketFraction(training):
    if "bucket_fraction" in config["trainings"][training]:
        return config["trainings"][training]["bucket_fraction"]


def getBucketSize(training):
    if "bucket_size" in config["trainings"][training]:
        return config["trainings"][training]["bucket_size"]


def getModelType(model):
    return config["models"][model]["model_type"]

def getModelMode(model):
    if "model_mode" in config["models"][model]:
        return "--model-mode " + config["models"][model]["model_mode"]
    else:
        return ""

def getModelLoss(model):
    if "loss" in config["models"][model]:
        return config["models"][model]["loss"]
    else:
        return "MSE"

def getTensorBoard(wc):
    model=wc.model
    training_type=wc.training_type
    training=wc.training
    option_string = ""
    if "flags" in config["models"][model]:
        if "tensorboard" in config["models"][model]["flags"]:
            option_string=f"--tensorboard-folder results/tensorboard/{training_type}/{training}_{model}"
    return option_string

def getSavedModel(wc):
    model=wc.model
    training_type=wc.training_type
    training=wc.training
    option_string = ""
    if "flags" in config["models"][model]:
        if "SavedModel" in config["models"][model]["flags"]:
            option_string=f"--SavedModel results/{training_type}/savedmodel/{training}_{model}"
    return option_string

def getLabelColumns(model):
    if "label_columns_start" in config["models"][model] and "label_columns_stop" in config["models"][model]:
        start = config["models"][model]["label_columns_start"]
        stop = config["models"][model]["label_columns_stop"]
        return f"--label-columns {start} {stop}"
    else:    
        return ""
