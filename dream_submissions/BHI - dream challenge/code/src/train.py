import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
#import wandb
import tqdm

import numpy as np
import pandas as pd

import data
import util
import net as networks

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

"""
Optimized Hyperparameter
conv_out_dim=512
lstm_hidden_size=320
kernel_sizes=[9, 15]
batch_size=512
lr=0.0015

<Additional Settings>
optimizer=AdamW
seed=40
num-epochs=15
loss=huber
"""

METRICS = ['pearson', 'spearman']

# write down train data path
data_path = 'data/train_sequences.txt'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-e', '--exp-name')
    parser.add_argument('-b', '--bsz', default=512, type=int)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--init-lr', default=0.0015, type=float)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--loss', default='Huber') # required=True)
    parser.add_argument('--num-epochs', default=15, type=int)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('-o', '--output', required=True)

    return parser.parse_args()

def get_optimizer(name, model, lr, wd=0.01):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Optimizer {name} is not supported yet.')

def get_lr(optimizer):
    for pg in list(optimizer.param_groups)[::-1]:
        return pg['lr']

def train(model, loader, criterion, optimizer, scheduler, epoch, step_scheduler_each_step=True, show_every=20):
    model.train()
    optimizer.zero_grad()

    running_loss, total_loss = 0., 0.
    metric = util.Metric(metrics=METRICS)

    bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch, data in bar:
        # Move loaded data to GPU.
        for k, v in data.items():
            data[k] = v.cuda()
        
        out = model(data['seq'])
        out_rc = model(data['seq_rc'])

        loss = criterion(out.view(-1, 1), data['target'].view(-1, 1))
        loss_rc = criterion(out_rc.view(-1, 1), data['target'].view(-1, 1))

        loss = (loss + loss_rc) / 2
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Average output from forward and reverse strand.
        out = (out + out_rc) / 2

        metric.update(data['target'].view_as(out).cpu().float(), out.detach().cpu())
        running_loss += loss.item()
        total_loss += loss.item()

        if scheduler is not None and step_scheduler_each_step:
            scheduler.step()

        if batch % show_every == 0:
            m = metric.compute()
            bar.set_postfix(loss=running_loss / show_every, lr=get_lr(optimizer), **m)

            # Codes for logging, but not used for now. 
#             log_dict = {'train/loss': running_loss / show_every, 'train/lr': get_lr(optimizer), 'epoch': epoch, 'step': batch}
#             log_dict.update({'train/' + k:v for k, v in metric.compute().items()})
#             wandb.log(log_dict)

            running_loss = 0.
            metric.clear()
    
    return total_loss / len(loader) # Return average loss per batch


def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0.
    predictions, targets = [], []

    metric = util.Metric(metrics=METRICS)

    with torch.no_grad():
        bar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for batch, data in bar:
            for k, v in data.items():
                data[k] = v.cuda()
            
            out = model(data['seq'])
            out_rc = model(data['seq_rc'])
            
            loss = criterion(out, data['target'].view_as(out).float())
            loss_rc = criterion(out, data['target'].view_as(out_rc).float())

            loss = (loss + loss_rc) / 2
            total_loss += loss.item()

            # Average output from forward and reverse strand.
            out = (out + out_rc) / 2

            pred = out.cpu().detach().numpy()
            predictions.append(pred)
            targets.append(data['target'].cpu().detach().numpy())
    
    val_predictions = np.concatenate(predictions)
    val_targets = np.concatenate(targets)

    metric.update(val_targets, val_predictions)
    metrics = metric.compute()

    return total_loss / len(loader), metrics, val_predictions, val_targets

if __name__ == '__main__':
    args = parse_arguments()
#     wandb.init(project='dream-challenge', entity='bhi-dream-challenge', group=args.exp_name, name=f'{args.exp_name}_fold{args.fold}')
#     wandb.config.update(args)
    util.seed_everything(args.seed)

    df = pd.read_csv(data_path, sep='\t', names=['sequence', 'measured_expression'])
    print('Loaded data with shape:', df.shape)


    # Determine train/val indices.
    kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
    for i, (train_idx, val_idx) in enumerate(kfold.split(df)):
        if i == args.fold:
            break
    
    # Dataset and dataloader.
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_set = data.DDPDataset(train_df)
    val_set = data.DDPDataset(val_df)
   
    print('Training set shape:', train_df.shape)
    print('Validation set shape:', val_df.shape)

    train_loader = DataLoader(train_set, batch_size=args.bsz, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.bsz, drop_last=False, pin_memory=True, num_workers=4)
    
    # Define model.
    if args.model == 'DeepGXP':
        model = networks.DeepGXP()
    else:
        raise ValueError(f'Unknown model name: {args.model}')

    model.cuda() # Assume that GPU is available.

    # Optimization settings.
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'MAE':
        criterion = nn.L1Loss()
    elif args.loss == 'Huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f'Unknown model name: {args.model}')
    optimizer = get_optimizer(args.optimizer, model, args.init_lr, args.weight_decay)

    # Let's use CosineAnnealing learning rate scheduler here!
    num_total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_total_steps)
    step_scheduler_each_step = True

    # Training loop!
    best_val_pearson = -10000
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, epoch, step_scheduler_each_step=True, show_every=20)
        val_loss, val_metrics, val_predictions, val_targets = validate(model, val_loader, criterion, epoch)

        # Log validation loss/metrics.
#         log_dict = {'val/loss': val_loss, 'epoch': epoch}
#         log_dict.update({'val/' + k: v for k, v in val_metrics.items()})
#         wandb.log(log_dict)

        # if not step_scheduler_each_step:
            # scheduler.step()

        if val_metrics['pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['pearson']
      
            torch.save(model.state_dict(), args.output)
            print(f'Best validation r={best_val_pearson}. Saved model checkpoint.')
    
        print(f'Train loss = {train_loss}')
        print(f'Validation loss = {val_loss}')
     
        for k, v in val_metrics.items():
            print(f'Validation {k}: {v}')

#     wandb.summary.update({
#         'best_val_pearson': best_val_pearson,
#         'last_val_pearson': val_metrics['pearson'],
#     })
