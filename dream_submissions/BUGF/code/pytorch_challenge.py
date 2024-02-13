#!/home/jones/miniconda3/bin/python

import sys
import os
# Uncomment if running on TPU
os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

# Uncomment if running on TPU
import torch_xla.core.xla_model as xm

from nndef_exp_alibi import TransformerNet

BATCH_SIZE = 512

RESTART_FLAG = False

# ################## Download and prepare the dataset ##################

def load_dataset():

    eval_list = []
    tnum = 0

    nt_trans = str.maketrans('ATGCN', 'ABCDE')

    #with open('test_sequences.txt', 'r') as testfile:
    with open(sys.argv[1], 'r') as testfile:
        for line in testfile:
            seq, expval = line.rstrip().split()
            byteseq = np.frombuffer(seq[17:-13].translate(nt_trans).encode('latin-1'), dtype=np.uint8) - ord('A')
            eval_list.append((byteseq, seq))

    return eval_list


class ExpDataset(Dataset):

    def __init__(self, sample_list):
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, tn):

        sample_list = self.sample_list
        ntseq = torch.from_numpy(sample_list[tn][0]).long()
        ntseq = F.pad(ntseq, (0,112-ntseq.size(0)), value=5)
        fullseq = sample_list[tn][1]

        sample = (ntseq, fullseq)

        return sample


# Trivial collate function
def my_collate(batch):

    inputs = torch.stack(list(sample[0] for sample in batch))
    fullseqs = list(sample[1] for sample in batch)

    return inputs, fullseqs


# ############################## Main program ################################

def main(num_epochs=2000):

    # Use this device if running on TPU
    device = xm.xla_device()
    #device = torch.device("cuda")

    # Create neural network model
    network = TransformerNet(512, 16, 16).to(device)

    # Load the dataset
    eval_list = load_dataset()

    ntest = len(eval_list)

    eval_data = ExpDataset(eval_list)

    eval_data_loader = DataLoader(dataset=eval_data, 
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             drop_last=False,
                             num_workers=2,
                             pin_memory=True,
                             collate_fn=my_collate)

    # Load current model snapshot
    network.load_state_dict(torch.load('FINAL_exp_model.pt', map_location=lambda storage, loc: storage), strict=True)

    network.eval()
    # And a full pass over the validation data:
    binvals = torch.arange(18, device=device).unsqueeze(0)
    with torch.no_grad():
        for sample_batch in eval_data_loader:
            inputs = sample_batch[0].to(device)
            fullseqs = sample_batch[1]
            predclasses = network(inputs)[0]
            outvals = (predclasses.softmax(-1) * binvals).sum(-1).cpu()
            outputs = zip(fullseqs, outvals)
            if len(sys.argv) > 2 and sys.argv[2] == '-q':
                # Just output values
                for v in outputs:
                    print("{:3.3f}".format(v[1].item()));
            else:
                for v in outputs:
                    print("{}\t{:3.3f}".format(v[0], v[1].item()));

if __name__=="__main__":
    main()
