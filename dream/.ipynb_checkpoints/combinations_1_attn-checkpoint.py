import torch
from torchinfo import summary
import itertools
import pandas as pd
import json
from prixfixe.autosome import AutosomeDataProcessor
from prixfixe.bhi import BHIDataProcessor
from prixfixe.unlockdna import UnlockDNADataProcessor

from prixfixe.unlockdna import (
                      UnlockDNAFirstLayersBlock,
                      UnlockDNAFinalLayersBlock)
from prixfixe.unlockdna_pos import UnlockDNACoreBlock
from prixfixe.bhi import (BHICoreBlock,
                      BHIFirstLayersBlock,
                      BHIFinalLayersBlock)
from prixfixe.autosome import (AutosomeCoreBlock,
                      AutosomeFirstLayersBlock,
                      AutosomeFinalLayersBlock)

from prixfixe.autosome import AutosomeTrainer
from prixfixe.bhi import BHITrainer
from prixfixe.unlockdna import UnlockDNATrainer

from prixfixe.prixfixe import PrixFixeNet
from prixfixe.prixfixe import CoreBlock
from typing import List
import sys

# TRAIN_DATA_PATH = "/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/data/demo_train.txt"
# VALID_DATA_PATH = "/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/data/demo_val.txt"

TRAIN_DATA_PATH = "/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/data/train_0.txt"
VALID_DATA_PATH = "/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/data/val_0.txt"

TRAIN_BATCH_SIZES = [512, 512, 256]
N_PROCS = 8
VALID_BATCH_SIZES = [512, 512, 256]
PLASMID_PATH = "/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/plasmid.json"
SEQ_SIZES = [150, 110, 200]
HEAD_LEN = 17
TAIL_LEN = 13
N_PROCS = 8

with open(f'{TRAIN_DATA_PATH}', 'r') as f:
    lines = f.readlines()
BATCH_PER_EPOCHS = [len(lines) // TRAIN_BATCH_SIZES[0], len(lines) // TRAIN_BATCH_SIZES[1], len(lines) // TRAIN_BATCH_SIZES[2]]
with open(f'{VALID_DATA_PATH}', 'r') as f:
    lines = f.readlines()
BATCH_PER_VALIDATIONS = [len(lines) // VALID_BATCH_SIZES[0], len(lines) // VALID_BATCH_SIZES[1], len(lines) // VALID_BATCH_SIZES[2]]

device = torch.device(f"cuda:0")
embedding_dim = 256
n_blocks = 4
kmer = 10
input_dim = 6
strides = 1
ratio = 0.05
num_heads = 8
rate = 0.1
num_projectors = 32
NUM_EPOCHS = [80, 15, 20]
lr = 0.001
CUDA_DEVICE_ID = 0

generator = torch.Generator()
generator.manual_seed(2147483647)

dataprocessors = [
    lambda: AutosomeDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE, 
    batch_per_epoch=BATCH_PER_EPOCH,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
    ), 
    lambda: BHIDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE, 
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
    ), 
    lambda: UnlockDNADataProcessor(
    path_to_training_data = TRAIN_DATA_PATH,
    path_to_validation_data= VALID_DATA_PATH,
    generator=generator,
    head_len = HEAD_LEN,
    tail_len = TAIL_LEN,
    max_width = SEQ_SIZE//2,
    train_batch_size = TRAIN_BATCH_SIZE,
    train_workers =N_PROCS,
    shuffle_train =True,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_val=False)]

firsts = [
    lambda: AutosomeFirstLayersBlock(in_channels=dataprocessor.data_channels(),
                                   out_channels=256, 
                                   seqsize=dataprocessor.data_seqsize()),
    
    lambda: BHIFirstLayersBlock(
    in_channels = dataprocessor.data_channels(),
    out_channels = 320,
    seqsize = dataprocessor.data_seqsize(),
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout = 0.2
    ),
    
    lambda: UnlockDNAFirstLayersBlock(in_channels = dataprocessor.data_channels(),
        out_channels = 512,
        seqsize = dataprocessor.data_seqsize(),
        kmer = kmer,
        strides = strides,
        num_projectors = num_projectors)]

cores = [
    lambda: AutosomeCoreBlock(in_channels=first.out_channels,
                         out_channels =64,
                         seqsize=first.infer_outseqsize()),
    
    lambda: BHICoreBlock(
    in_channels = first.out_channels,
    out_channels = 320,
    seqsize = first.infer_outseqsize(),
    lstm_hidden_channels = 320,
    kernel_sizes = [9, 15],
    pool_size = 1,
    dropout1 = 0.2,
    dropout2 = 0.5
    ),
    
    lambda: UnlockDNACoreBlock(
    in_channels = first.out_channels, out_channels= first.out_channels, seqsize = dataprocessor.data_seqsize(), n_blocks = n_blocks,
                                     kernel_size = 15, rate = rate, num_heads = num_heads)]

finals = [
    lambda: AutosomeFinalLayersBlock(in_channels=core.out_channels,
                                 seqsize=core.infer_outseqsize()),
    
    lambda: BHIFinalLayersBlock(
    in_channels = core.out_channels,
    seqsize = dataprocessor.data_seqsize(),
    hidden_dim = 64,
    ),
    
    lambda: UnlockDNAFinalLayersBlock(
    in_channels = core.out_channels,
    seqsize = dataprocessor.data_seqsize(),
    num_projectors = num_projectors,
    input_dim = input_dim,
    rate = rate)]

trainers = [
    lambda: AutosomeTrainer(
    model,    
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"), 
    model_dir=MODEL_LOG_DIR,
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCH,
    lr = lr),
    
    lambda: BHITrainer(
    model,    
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"), 
    model_dir=MODEL_LOG_DIR,
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCH
    ),
    
    lambda: UnlockDNATrainer(
    model,
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"),
    model_dir=MODEL_LOG_DIR,
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCH,
    initial_lr = 1e-14,
    embedding_dim = 256,
    warmup_steps = 12500,
    beta_1 =  0.9,
    beta_2 = 0.98,
    eps = 1e-9,
    clip_norm = 1.,
    clip_value = 0.5,
    n_positions = SEQ_SIZE,
    N = 4,
    M = 5)
    ]

job_id = int(sys.argv[1])

for dataprocessor_id in range(3):
    SEQ_SIZE = SEQ_SIZES[dataprocessor_id]
    TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZES[dataprocessor_id]
    VALID_BATCH_SIZE = VALID_BATCH_SIZES[dataprocessor_id]
    BATCH_PER_EPOCH = BATCH_PER_EPOCHS[dataprocessor_id]
    BATCH_PER_VALIDATION = BATCH_PER_VALIDATIONS[dataprocessor_id]
    lr = lr
    NUM_EPOCH = NUM_EPOCHS[dataprocessor_id]
    dataprocessor = dataprocessors[dataprocessor_id]()

    for id, (first, core, final) in enumerate(itertools.product(firsts, cores, finals)):

        check = dataprocessor_id * 27 + id
        if check != job_id:
            continue
        id_0, id_1, id_2 = firsts.index(first), cores.index(core), finals.index(final)

        first = first()
        core = core()
        final = final()

        model = PrixFixeNet(
        first=first,
        core=core,
        final=final,
        generator=generator
        )
        # model.check()
        from torchinfo import summary
        print(summary(model, (1, 4, 110)))
        print(f"{dataprocessor_id}_{id_0}_{id_1}_{id_2}")
        MODEL_LOG_DIR = f"/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/paper_runs_2_pos/model_weight/{dataprocessor_id}_{id_0}_{id_1}_{id_2}"
        trainer = trainers[dataprocessor_id]()
        trainer.fit()