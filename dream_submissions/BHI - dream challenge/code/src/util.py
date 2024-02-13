import os
import torch
import random
import tqdm
import numpy as np
import scipy.stats as stats

import sklearn.metrics as met

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Metric(object):
    def __init__(self, metrics=[]):
        if len(metrics) == 0:
            raise ValueError('You must specify at least one metrics!')
        
        self.metrics = {}
        for metric in metrics:
            if metric == 'mae':
                self.metrics[metric] = met.mean_absolute_error
            elif metric == 'pearson':
                self.metrics[metric] = lambda x, y: stats.pearsonr(x, y)[0]
            elif metric == 'spearman':
                self.metrics[metric] = lambda x, y: stats.spearmanr(x, y)[0]
            elif metric == 'r2':
                self.metrics[metric] = met.r2_score
            elif metric == 'auc':
                self.metrics[metric] = met.roc_auc_score
            elif metric == 'auprc':
                self.metrics[metric] = met.average_precision_score
            elif metric == 'f1':
                self.metrics[metric] = met.f1_score
            else:
                raise ValueError(f'Unsupported metric: {metric}')

        self.y_true, self.y_pred = [], []

    def update(self, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            self.y_true.append(y_true.flatten())
            self.y_pred.append(y_pred.flatten())
        elif isinstance(y_true, torch.Tensor):
            self.y_true.append(y_true.numpy().flatten())
            self.y_pred.append(y_pred.numpy().flatten())

    def clear(self):
        self.y_true, self.y_pred = [], []

    def compute(self):
        computed = {}
        for metric, f in self.metrics.items():
            try:
                computed[metric] = f(np.concatenate(self.y_true), np.concatenate(self.y_pred))
            except:
                computed[metric] = np.nan

        return computed
