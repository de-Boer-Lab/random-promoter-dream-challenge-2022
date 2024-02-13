"""
Final submission from Camformers.

Usage:
python3 camformers_submission.py
"""


__date__ = "26/07/2022"


### imports ###
import warnings
import math
import os
import sys
import time
from optparse import OptionParser
import collections
from collections import OrderedDict
from itertools import repeat
import json
import random

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Use TPU device?
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU = True
    os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
except:
    TPU = False


### config ###
rdm_seed = 42 # Answer to the Ultimate Question of Life, the Universe, and Everything
torch.manual_seed(rdm_seed) # set seed for torch RNG
np.random.seed(rdm_seed)
random.seed(rdm_seed)

# File options
input_file = "Data/train_sequences.txt" # File with training data
predict_file = "Data/test_sequences.txt" # Datafile to predict
best_model_path = "Models/trained_model.pt" # Path to save model file
sub_path = "Output/submission_file.csv" # Path to save a file for submission
validation_size = 0.1

# One hot encode options
target_len = 110 # Truncate to this length (if drop=True)
margin = 3 # Keep sequences with target_len +- margin
drop = True # Drop instances that are not = target_len+- margin
N_tolerance = 3 # Number of Ns that are acceptable in the train sequence

# Training options
batch_size = 256
epochs = 50 # Max number of epochs
learn_rate = 1e-3
weight_decay = 1e-3
#threshold = 1e-3 # Difference in the evaluation metric to be considered an improvement.
patience = 10 # number of epochs with no improvement before stop.

# Model options
out_channels=[512, 512, 512, 512, 512, 512]
kernels=[(10, 1), (10, 1), (10, 1), (10, 1),(10, 1), (10, 1)]
pool_kernels = [(1, 1), (1, 1), (1, 1), (1, 1), (10, 1), (1, 1)]
paddings="same" # Give a list of paddings ([(0,0), (0,0), (0,0)]) or "same"
strides=[(1,1), (1,1), (1,1), (1,1),(1,1), (1,1), (1,1), (1,1),(1,1), (1,1), (1,1), (1,1)]
pool_strides = [(1, 1), (1, 1), (1, 1), (1, 1),(4, 1), (1, 1)]
dropouts=[0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
linear_dropouts = [0.0, 0.0]
linear_output=[256, 256] # A final output size of 1 will always be appended as the last layer.

# Command line option
parser = OptionParser()
parser.add_option("--cpu", action="store_true", dest="cpu")
(options, args) = parser.parse_args()


# Get cpu/tpu/gpu device for training.
# If torch_xla is not installed, the train function needs to be modified
# to remove xm.mark_step()
if options.cpu:
    device = "cpu" # force cpu for debug
elif TPU:
    device = xm.xla_device()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Check that the required folders exists, else create them.
if not os.path.isdir('Models'):
    os.mkdir('Models')
if not os.path.isdir('Output'):
    os.mkdir('Output')


### Functions ####     
def my_spearmanr(set1, set2):
    """
    Calculate Spearman correlation. If return nan instead return 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rho = stats.spearmanr(set1, set2)[0]
            if np.isnan(rho):
                rho = 0
        except:
            rho=0
    return rho


def my_pearsonr(set1, set2):
    """
    Calculate Pearson correlation. If return nan instead return 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = stats.pearsonr(set1, set2)[0]
            if np.isnan(r):
                r = 0
        except:
            r = 0
    return r


def onehote(seq):
    """
    One hot encode DNA sequence.
    If not ACGT, then fill with 0.
    """
    seq2=[] # empty list to store the endoded seq.
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T":[0., 0., 0., 1.]}
    for i in seq:
        seq2.append(mapping[i] if i in mapping.keys() else [0., 0., 0., 0.]) # If not in the above map, use all 0.
    return seq2


def onehote_reverse(seq):
    """
    Get back sequence from ohe.
    """
    seq2=[] # empty list to store the endoded seq.
    mapping = {"[1.0, 0.0, 0.0, 0.0]":"A", "[0.0, 1.0, 0.0, 0.0]":"C", "[0.0, 0.0, 1.0, 0.0]":"G", "[0.0, 0.0, 0.0, 1.0]":"T"}
    for i in seq:
        i = str(i)
        seq2.append(mapping[i] if i in mapping.keys() else "N") # If not in the above map, use N
    return seq2


def convolution_output_size(input_size, kernel, padding, stride):
    """
    Calculate the output size of a convolutional layer.
    """
    out_height = int(((input_size[0] + 2*padding[0] - (kernel[0] - 1) - 1)/stride[0])+1)
    out_width = int(((input_size[1] + 2*padding[1] - (kernel[1] - 1) - 1)/stride[1])+1)
    
    return (out_height, out_width)


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def conv_block(in_f, out_f, kernel, pool_kernel, stride, pool_stride, padding, dilation=1, dropout=0.1):
    """
    Return a 2D convolutional block.
    """
    modules = []
    if padding == "same":
        modules.append(Conv2dSame(in_channels=in_f, out_channels=out_f, kernel_size=kernel, stride=stride, dilation=dilation))
    else:
        modules.append(nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation))
    modules.append(nn.BatchNorm2d(out_f))
    modules.append(nn.ReLU())
    if pool_kernel != (1,1):
        modules.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
    modules.append(nn.Dropout(dropout))

    layer = nn.Sequential(*modules)
    return layer


def lin_block(in_f, out_f, dropout=0.0):
    """
    Return a linear block.
    """
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

# From https://github.com/pytorch/pytorch/issues/3867#issuecomment-974159134
class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0, 
    so we need to implement the internal torch logic manually. 

    Currently the ``RuntimeError`` is
    
    "PyTorch convert function for op '_convolution_mode' not implemented"
    """

    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)
        _pair = _ntuple(2)
        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_, 
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class CNN(nn.Module):   
    
    """
    Define the neural network structure.
    """
    
    def __init__(self, feature_height, feature_width, batch_size, print_size=False,
            out_channels=[256, 256, 256], kernels=[(12,1),(4,1),(4,1)], pool_kernels=[(1,1),(1,1),(1,1)], paddings=[(0,0), (0,0), (0,0)], 
            strides=[(1,1), (1,1), (1,1)], pool_strides=[(1,1), (1,1), (1,1)], dropouts=[0.2, 0.2, 0.2], linear_output=[64], linear_dropouts=[0.0]):
        
        super(CNN, self).__init__()
        
        if print_size:
            print(locals())
        
        input_channels = 4 # Number of channels in the first layer
        out_size = (feature_height, feature_width) # Set a start size
        
        
        if print_size:
            print("Sizes: ", end="")
        # Calculate layer output sizes, output if print_size = True
        if paddings == "same":
            paddings = ["same" for x in range(len(out_channels))]
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")
        else:
            for n, channels in enumerate(out_channels):
                out_size = convolution_output_size(out_size, kernels[n], paddings[n], strides[n])
                print(out_size[0],"x",out_size[1],end="\t")
                out_size = convolution_output_size(out_size, pool_kernels[n], (0,0), pool_strides[n])
                if print_size:
                    print(out_size[0],"x",out_size[1],end="\t")
        
        # Calculate the linear input size.
        output_size = int(out_channels[-1] * out_size[0] * out_size[1]) # Input to the linear layers
        if print_size:
            print("Linear input size: %s" % output_size)
       
        # Define the convolutional layers
        self.CNN_layers = nn.ModuleList()
        conv_sizes = [x for x in zip([input_channels] + out_channels, out_channels)]
        for n, sizes in enumerate(conv_sizes):
            conv_blocks = conv_block(sizes[0], sizes[1], kernel=kernels[n], pool_kernel=pool_kernels[n], stride=strides[n], pool_stride=pool_strides[n], padding="same", dilation=1, dropout=dropouts[n])
            self.CNN_layers.append(nn.Sequential(*conv_blocks))

        # Define the fully connected layers
        if len(linear_output) == 0:
            linear_blocks = nn.Sequential(nn.Linear(output_size, 1))
        else:
            linear_sizes = [x for x in zip([output_size] + linear_output, linear_output + [1])]
            linear_blocks = [lin_block(in_f, out_f, dropout=linear_dropouts[n]) for n, (in_f, out_f) in enumerate(linear_sizes[:-1])]
            last_layer = nn.Sequential(nn.Linear(linear_sizes[-1][0], linear_sizes[-1][1]))
            linear_blocks.append(last_layer)
        self.linear_layers = nn.Sequential(*linear_blocks)

    def forward(self, x):
        
        for i in range(0, len(self.CNN_layers), 2):
            x = self.CNN_layers[i](x)
            ShortCut = x
            x = self.CNN_layers[i+1](x)
            x = x + ShortCut
        
        # FC layers
        x = x.view(x.size(0), -1) # flatten before FC layers
        x = self.linear_layers(x)
        x = torch.flatten(x)
        
        return x


def train(dataloader, model, loss_fn, optimizer, scheduler, valid_loader, epochs=epochs):
    
    # Store some key variables
    #train_losses = [] # store history
    #val_losses = [] # store history
    best_score = -1 # value for early stopping
    run_val_rho = 0
    epochs_no_improve = 0
    best_epoch = 0
    
    # print output header
    print("\t\tT loss\tV loss\tT r\tV r\tT rho\tV rho")
    
    for epoch in range(epochs): # Train for the stipulated number of epochs (or untill early stopping)
            
        model.train() # Set model to training mode
        
        train_loss = 0
        y_pred = []
        y_true = []
        for (X, y) in dataloader: # Returns X and y in batch-sized chunks. 
            X, y = X.to(device), y.to(device)
    
            # Compute loss
            pred = model(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            optimizer.zero_grad() # clears old gradients from the last step 
            loss.backward() # computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
            xm.mark_step() # xm-code
        
            # Save results
            train_loss += loss.item()
            y_pred.append(pred.cpu().detach())
            y_true.append(y.cpu().detach())
        
        train_loss /= len(dataloader)*1.0 # Divide the loss by the number of batches
        #train_losses.append(train_loss)
        
        # After completing training, evaluate model on the validation set.
        
        # Store some key variables for validation.
        valid_loss = 0
        y_pred_val = []
        y_true_val = []

        model.eval()
        with torch.no_grad(): # Turn off gradient calculation for inference.
            for (X, y) in valid_loader:
                X, y = X.to(device), y.to(device)
    
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                xm.mark_step() # xm-code
                
                # Save results
                valid_loss += loss.item()
                y_pred_val.append(pred.cpu().detach())
                y_true_val.append(y.cpu().detach())
        
        valid_loss /= len(valid_loader)*1.0
        #val_losses.append(valid_loss)
        scheduler.step(valid_loss)
        
        # Check for improvement in validation scores.
        #if min_val_loss - valid_loss >= threshold:
        run_val_rho = my_spearmanr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        run_val_r = my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val))
        val_score = run_val_rho + run_val_r
        if val_score > best_score:
            torch.save(model.state_dict(), best_model_path) # Save the model
            epochs_no_improve=0
            #min_val_loss = valid_loss
            best_epoch = epoch
            #best_r = stats.pearsonr(np.concatenate(y_pred_val), np.concatenate(y_hat_val))[0]
            best_score = val_score
        else:
            epochs_no_improve += 1
                
        # Output the results of the epoch
        output = [] # List of output
        output.append("Epoch %s:" % (epoch+1))
        # Losses
        output.append(round(train_loss, 4))
        output.append(round(valid_loss, 4))
        # Pearson r
        output.append(round(my_pearsonr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(my_pearsonr(np.concatenate(y_pred_val), np.concatenate(y_true_val)), 4) if len(y_pred_val) > 0 else "-")
        # Spearman rho
        output.append(round(my_spearmanr(np.concatenate(y_pred), np.concatenate(y_true)), 4))
        output.append(round(run_val_rho, 4) if len(y_pred_val) > 0 else "-")
        if valid_loader and epochs_no_improve == 0: # mark the best epochs
            output.append("*")
        print("\t".join(str(x) for x in output))
        
        # Check for early stopping
        if epochs_no_improve == patience:
            print('Early stop, best epoch %s' % (best_epoch+1))
            break

    return model

        
def Model_Pred(model, test_load):
    """
    Function to make predictions using the supplied model.
    
    Returns two np arrays, predicted values from the model, and true 
    values from the data loader.
    """
    
    model.eval()
    PredList=[] # Save predictions
    label_list = [] # Save true labels
    
    with torch.no_grad():
        for (data, label) in test_load:
            data = data.to(device=device)
            pred = model(data)
            true_label = label.numpy()
            
            PredList.append(pred.cpu().detach().numpy())
            label_list.append(true_label)

    return np.concatenate(PredList), np.concatenate(label_list) # use concatenate to join the results lists from the different batches.


def OHE(input_file):
    """
    One Hot Encode input and return X, y
    """
    # Read the input file and save as X and y values
    X = []
    y = []
    target_len_data = 0 # Store the max length of input sequence.
    path, ext = os.path.splitext(input_file)
    n_drop = 0 # Number of seq dropped because of N count
    line_count = 0

    with open(input_file) as f:
        for line in f:
            line_count += 1
            line = line.strip()
            line_list = line.split()
            seq = line_list[0]
            y_val = float(line_list[1])
            if seq.count("N") <= N_tolerance:
                X.append(line_list[0])
                if len(line_list[0]) > target_len_data: # Store if longer than seen before
                    target_len_data = len(seq)
                y.append(float(y_val))
            else:
                n_drop+=1
    
    print("Total lines in infile: ", line_count)
    print("Dropped for N count: ", n_drop)
    
    data = []
    y_out = []
    not_on_target = 0 # number of instances outside the target length
    dropped_seq = 0
    for n, item in enumerate(X):
        # Add padding/truncate to make every instance the same length.
        if len(item) != target_len:
            # If drop is set, the instance is discarded, else it is trucated/padded to the set length
            if abs(len(item)-target_len) > margin:
                not_on_target+=1
                if drop:
                    dropped_seq+=1
                    continue # move to next item
            while len(item) < target_len_data and len(item) < target_len:
                item+="N"
            if len(item) > target_len:
                item = item[:target_len]
        
        # OHE and save
        ohe_seq = onehote(item)
        data.append(ohe_seq)
        y_out.append(y[n])
    
    print("Outside length spec: ", dropped_seq)
    return data, y_out


### Main ###

if __name__ == "__main__":
    
    start = time.time()
    
    print("### Reading and Encoding ###")
    
    X, y = OHE(input_file)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, shuffle=True, random_state=rdm_seed)
    
    # Create tensors and DataLoader
    TrainX_Tensor=torch.tensor(X_train)
    TrainX_Tensor = TrainX_Tensor.unsqueeze(1)
    TrainX_Tensor = torch.transpose(TrainX_Tensor,1,3)
    TrainY_Tensor=Tensor(y_train)
    g = torch.Generator()
    g.manual_seed(rdm_seed)
    TrainLoader=DataLoader(dataset=TensorDataset(TrainX_Tensor, TrainY_Tensor), num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=np.random.seed(rdm_seed), generator=g)
    
    ValX_Tensor=torch.tensor(X_val)
    ValX_Tensor = ValX_Tensor.unsqueeze(1)
    ValX_Tensor = torch.transpose(ValX_Tensor,1,3)
    ValY_Tensor=Tensor(y_val)
    ValLoader=DataLoader(dataset=TensorDataset(ValX_Tensor, ValY_Tensor), num_workers=0, batch_size=batch_size, shuffle=True, drop_last=True)

    
    # Output shapes of input.
    for X, y in TrainLoader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        feature_height = X.shape[2] # Get the height of the input
        feature_width = X.shape[3] # Get the width of the input
        print(f"Shape of y: {y.shape} {y.dtype}")
        print("Total training instances: %s" % TrainX_Tensor.size(dim=0))
        break
    
    # Define model and training paramters
    model_args = {
        "feature_height":feature_height,
        "feature_width":feature_width,
        "batch_size":batch_size,
        "print_size":True,
        "out_channels":out_channels,
        "kernels":kernels,
        "pool_kernels":pool_kernels,
        "paddings":paddings,
        "strides":strides,
        "pool_strides":pool_strides,
        "dropouts":dropouts,
        "linear_output":linear_output,
        "linear_dropouts":linear_dropouts
    }
    model = CNN(**model_args)
    model.to(device) # Move model to device
    print(model) # This gives a view of all the layers and sizes.
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # Number of trainable parameters in the model
    
    #loss_fn = nn.MSELoss().to(device) # Mean squared error loss
    loss_fn = nn.L1Loss().to(device) # Absolute error loss
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate) #, weight_decay=weight_decay) # Adaptive extention to SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print("### Model Training ###")
    
    # Train model
    train(TrainLoader, model, loss_fn, optimizer, scheduler, ValLoader)
    
    # Predict the test set
    print("Predicting test set...")
    X, y = OHE(predict_file)
    
    X_Tensor=torch.tensor(X)
    X_Tensor = X_Tensor.unsqueeze(1)
    X_Tensor = torch.transpose(X_Tensor,1,3)
    Y_Tensor=Tensor(y)
    
    TestLoader = DataLoader(dataset=TensorDataset(X_Tensor, Y_Tensor), batch_size=batch_size, shuffle=False, drop_last=False)
    
    Model_best=CNN(**model_args)
    Model_best.load_state_dict(torch.load(best_model_path))
    Model_best.to(device)
    
    y_pred, y_true = Model_Pred(Model_best, TestLoader)
    
    print("### Prediction Summary ###")
    df_describe = pd.DataFrame(y_pred)
    print(df_describe.describe())
    print("< 0:\t",len([x for x in y_pred if x < 0]))
    
    # Create the submission file
    with open('script/sample_submission.json', 'r') as f:
        ground = json.load(f)

    indices = np.array([int(indice) for indice in list(ground.keys())])
    PRED_DATA = OrderedDict()
    
    for i in indices:
    #Y_pred is an numpy array of dimension (71103,) that contains your
    #predictions on the test sequences
        PRED_DATA[str(i)] = float(y_pred[i])
    
    def dump_predictions(prediction_dict, prediction_file):
        with open(prediction_file, 'w') as f:
            json.dump(prediction_dict, f)
    
    timestr = time.strftime('%Y%m%d-%H%M%S')
    dump_predictions(PRED_DATA, 'submission'+timestr+'.json')

    print('Submission file "'+'submission'+timestr+'.json'+'" has been prepared.')
    
    print( "Run time: ", time.time() - start)
