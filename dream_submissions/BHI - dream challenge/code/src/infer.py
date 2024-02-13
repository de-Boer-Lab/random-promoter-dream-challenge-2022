import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import wandb
import tqdm

import numpy as np
import pandas as pd

import data
import util
import net as networks

from torch.utils.data import DataLoader

import json

# write down test data path
data_path = 'data/test_sequences.txt'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input checkpoint.')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-t', '--tta', type=int, default=1)
    parser.add_argument('-o', '--output', required=True, help='Output json file')

    return parser.parse_args()

def predict(model, loader):
    model.eval()

    predictions = []
    with torch.no_grad():
        bar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for batch, data in bar:            
            outs = []
            for seq in data['seqs']:
                outs.append(model(seq.cuda()))
            
            # Average output from forward and reverse strand.
            out = torch.stack(outs).mean(axis=0)

            pred = out.cpu().detach().numpy()
            predictions.append(pred)
    
    val_predictions = np.concatenate(predictions)

    return val_predictions

if __name__ == '__main__':
    args = parse_arguments()
    util.seed_everything(args.seed)

    df = pd.read_csv(data_path, sep='\t', names=['sequence', 'measured_expression'])
    print('Loaded data with shape:', df.shape)
    test_set = data.DDPDatasetShift(df, infer=True, tta=args.tta)
    test_loader = DataLoader(test_set, batch_size=1024, drop_last=False, pin_memory=True, num_workers=4)

     # Define model.
    if args.model == 'DeepGXP':
        model = networks.DeepGXP()
    else:
        raise ValueError(f'Unknown model name: {args.model}')

    model.load_state_dict(torch.load(args.input, map_location='cpu'))
    model.cuda() # Assume that GPU is available.

    test_predictions = predict(model, test_loader)
    
    df['measured_expression'] = test_predictions

    prediction_dict = {str(i): float(df.measured_expression.values[i]) for i in range(len(test_predictions))}
    with open(args.output, 'w') as f:
        json.dump(prediction_dict, f)
        
