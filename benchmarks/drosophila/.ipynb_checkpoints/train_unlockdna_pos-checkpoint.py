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
from tqdm import tqdm
import sys

test_fold_nums = range(5)
val_fold_nums = range(10)
combinations = []

for fold_num1 in test_fold_nums:
    for fold_num2 in val_fold_nums:
            combinations.append((fold_num1, fold_num2))

id = int(sys.argv[1])
test_fold, val_fold = combinations[id]
print(test_fold, val_fold)
print(len(combinations))

TRAIN_DATA_PATH = f"k_fold_splits/test_fold_{test_fold}/val_fold_{val_fold}/train_data.tsv"
VALID_DATA_PATH = f"k_fold_splits/test_fold_{test_fold}/val_fold_{val_fold}/val_data.tsv"
TEST_DATA_PATH = f"k_fold_splits/test_fold_{test_fold}/test_data.tsv"
MODEL_LOG_DIR = f"model_weight/unlockdna_pos/test_fold_{test_fold}/val_fold_{val_fold}"

TRAIN_BATCH_SIZE = 32
N_PROCS = 8
VALID_BATCH_SIZE = 32
lr = 0.001
BATCH_PER_EPOCH = len(pd.read_csv(TRAIN_DATA_PATH))//TRAIN_BATCH_SIZE
BATCH_PER_VALIDATION = len(pd.read_csv(VALID_DATA_PATH))//TRAIN_BATCH_SIZE
SEQ_SIZE = 249
NUM_EPOCHS = 80
CUDA_DEVICE_ID = 0

generator = torch.Generator()
generator.manual_seed(42)
device = torch.device(f"cuda:{CUDA_DEVICE_ID}")

from prixfixe.autosome import (AutosomeCoreBlock,
                      AutosomeFirstLayersBlock,
                      AutosomeFinalLayersBlock)

from prixfixe.prixfixe import PrixFixeNet
from prixfixe.unlockdna_pos import UnlockDNACoreBlock
from prixfixe.bhi import BHIFirstLayersBlock

first = AutosomeFirstLayersBlock(in_channels=5,
                                out_channels=256, 
                                seqsize=249)
core = UnlockDNACoreBlock(
    in_channels = first.out_channels, out_channels= first.out_channels, seqsize = 249, 
    n_blocks = 4,kernel_size = 15, rate = 0.1, num_heads = 8)

final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

model = PrixFixeNet(
    first=first,
    core=core,
    final=final,
    generator=generator
)

from torchinfo import summary
print(summary(model, (1, 5, 249)))


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

test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
test_df['rev'] = test_df['ID'].str.contains('\-_').astype(int)

model.load_state_dict(torch.load(f"{MODEL_LOG_DIR}/model_best_MSE.pth"))
model.eval()

# Function to one-hot encode a DNA sequence
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}
    return [mapping[base] for base in seq]

# One-hot encode sequences and concatenate 'rev' column
encoded_seqs = []
Y_test_dev = []
Y_test_hk = []

for i, row in tqdm(test_df.iterrows()):
    encoded_seq = one_hot_encode(row['Sequence'])
    rev_value = [row['rev']] * len(encoded_seq)
    encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)]
    encoded_seqs.append(encoded_seq_with_rev)
    Y_test_dev.append(row['Dev_log2_enrichment'])
    Y_test_hk.append(row['Hk_log2_enrichment'])

# encoded_seqs = encoded_seqs[:100]
pred_expr_dev = []
pred_expr_hk = []

for seq in tqdm(encoded_seqs):
    pred = model(torch.tensor(np.array(seq).reshape(1,249,5).transpose(0,2,1), device = device, dtype = torch.float32))
    pred_expr_dev.append(pred[0].detach().cpu().flatten().tolist())
    pred_expr_hk.append(pred[1].detach().cpu().flatten().tolist())

import numpy as np
pred_path = f"model_prediction_MSE/unlockdna_pos/test_fold_{test_fold}/val_fold_{val_fold}/"

if not os.path.exists(pred_path):
    os.makedirs(pred_path, exist_ok=True)
# Save the arrays to .npy files
np.save(f"{pred_path}" + "dev_GT.npy", Y_test_dev)
np.save(f"{pred_path}" + "hk_GT.npy", Y_test_hk)
np.save(f"{pred_path}" + "dev_pred.npy", pred_expr_dev)
np.save(f"{pred_path}" + "hk_pred.npy", pred_expr_hk)

test_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
test_df['rev'] = test_df['ID'].str.contains('\-_').astype(int)

model.load_state_dict(torch.load(f"{MODEL_LOG_DIR}/model_best_pearsonr.pth"))
model.eval()

# Function to one-hot encode a DNA sequence
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}
    return [mapping[base] for base in seq]

# One-hot encode sequences and concatenate 'rev' column
encoded_seqs = []
Y_test_dev = []
Y_test_hk = []

for i, row in tqdm(test_df.iterrows()):
    encoded_seq = one_hot_encode(row['Sequence'])
    rev_value = [row['rev']] * len(encoded_seq)
    encoded_seq_with_rev = [list(encoded_base) + [rev] for encoded_base, rev in zip(encoded_seq, rev_value)]
    encoded_seqs.append(encoded_seq_with_rev)
    Y_test_dev.append(row['Dev_log2_enrichment'])
    Y_test_hk.append(row['Hk_log2_enrichment'])

# encoded_seqs = encoded_seqs[:100]
pred_expr_dev = []
pred_expr_hk = []

for seq in tqdm(encoded_seqs):
    pred = model(torch.tensor(np.array(seq).reshape(1,249,5).transpose(0,2,1), device = device, dtype = torch.float32))
    pred_expr_dev.append(pred[0].detach().cpu().flatten().tolist())
    pred_expr_hk.append(pred[1].detach().cpu().flatten().tolist())

import numpy as np
pred_path = f"model_prediction_pearsonr/unlockdna_pos/test_fold_{test_fold}/val_fold_{val_fold}/"

if not os.path.exists(pred_path):
    os.makedirs(pred_path, exist_ok=True)
# Save the arrays to .npy files
np.save(f"{pred_path}" + "dev_GT.npy", Y_test_dev)
np.save(f"{pred_path}" + "hk_GT.npy", Y_test_hk)
np.save(f"{pred_path}" + "dev_pred.npy", pred_expr_dev)
np.save(f"{pred_path}" + "hk_pred.npy", pred_expr_hk)