# Required imports
import pandas as pd
import torch
import sys
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
import sys
import ast
import json

filenames = ['/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/clean_data/HepG2_clean.tsv',
             '/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/clean_data/K562_clean.tsv',
             '/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/clean_data/WTC11_clean.tsv']

fold_nums = range(10)
combinations = []

for filename in filenames:
    for fold_num1 in fold_nums:
        for fold_num2 in fold_nums:
            if fold_num1 != fold_num2:
                combinations.append((filename, fold_num1, fold_num2))
                
CUDA_DEVICE_ID = 0
id = int(sys.argv[1])

filename, test_fold, val_fold = combinations[id]
print(filename, test_fold, val_fold)

TRAIN_DATA_PATH = f"/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/splits/{os.path.splitext(os.path.basename(filename))[0]}/test_fold_{test_fold}/val_fold_{val_fold}/train_data.tsv"
VALID_DATA_PATH = f"/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/splits/{os.path.splitext(os.path.basename(filename))[0]}/test_fold_{test_fold}/val_fold_{val_fold}/val_data.tsv"
TEST_DATA_PATH = f"/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/splits/{os.path.splitext(os.path.basename(filename))[0]}/test_fold_{test_fold}/test_data.tsv"
MODEL_LOG_DIR = f"/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/model_weight/{os.path.splitext(os.path.basename(filename))[0]}/autosome/test_fold_{test_fold}/val_fold_{val_fold}"
TRAIN_BATCH_SIZE = 32
N_PROCS = 4
VALID_BATCH_SIZE = 32
lr = 0.005
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH))//TRAIN_BATCH_SIZE
SEQ_SIZE = 230
NUM_EPOCHS = 80

generator = torch.Generator()
generator.manual_seed(42)
device = torch.device(f"cuda:{CUDA_DEVICE_ID}")

from prixfixe.autosome import (AutosomeCoreBlock,
                      AutosomeFirstLayersBlock,
                      AutosomeFinalLayersBlock)

from prixfixe.prixfixe import PrixFixeNet
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.bhi import BHIFirstLayersBlock

first = BHIFirstLayersBlock(
            in_channels = 5,
            out_channels = 320,
            seqsize = 230,
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout = 0.2
        )

core = AutosomeCoreBlock(in_channels=first.out_channels,
                        out_channels =64,
                        seqsize=first.infer_outseqsize())

final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

model = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)
# from torchinfo import summary
# print(summary(model, (1, 5, 230)))

from prixfixe.autosome import AutosomeDataProcessor

dataprocessor = AutosomeDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE, 
    batch_per_epoch=BATCH_PER_EPOCH,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    seqsize=SEQ_SIZE,
    generator=generator
)

from prixfixe.autosome import AutosomeTrainer
trainer = AutosomeTrainer(
    model,
    device=torch.device(f"cuda:{CUDA_DEVICE_ID}"), 
    model_dir=MODEL_LOG_DIR,
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCHS,
    lr = lr)

trainer.fit()


from prixfixe.autosome import AutosomePredictor
model.load_state_dict(torch.load(f"{MODEL_LOG_DIR}/model_best.pth"))
model.eval()
import pandas as pd
test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
test_df['rev'] = test_df['seq_id'].str.contains('_Reversed:').astype(int)

    # Function to one-hot encode a DNA sequence
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]}
    return [mapping[base] for base in seq]

# One-hot encode sequences and concatenate 'rev' column
encoded_seqs = []
for i, row in tqdm(test_df.iterrows()):
    encoded_seq = one_hot_encode(row['seq'])
    rev_value = [row['rev']] * len(encoded_seq)
    encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)]
    encoded_seqs.append(encoded_seq_with_rev)

from tqdm import tqdm
pred_expr = []
for seq in tqdm(encoded_seqs):
    pred = model(torch.tensor(np.array(seq).reshape(1,230,5).transpose(0,2,1), device = device, dtype = torch.float32))
    
    pred_expr.append(pred.detach().cpu().flatten().tolist())

directory_path = f"/home/rafi11/projects/def-cdeboer/rafi11/DREAMNet/Vikram_torch/model_prediction/{os.path.splitext(os.path.basename(filename))[0]}/autosome/test_fold_{test_fold}"
if not os.path.exists(directory_path):
# Create the directory
    os.makedirs(directory_path)

with open(f"{directory_path}/val_fold_{val_fold}.json", 'w') as f:
    json.dump(pred_expr, f)