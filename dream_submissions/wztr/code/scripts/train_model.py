from cgi import test
from random import sample
from unicodedata import name
from black import out
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from seq2exp import tools, trainner, data
from seq2exp import models
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import torch
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    # Parse arguments add all hyperparameters to the parser
    parser = argparse.ArgumentParser(description="Model Hyperparameters")
    parser.add_argument(
        "--model_type", type=str, default="Hybrid_CNN", help="Model type"
    )
    parser.add_argument("--in_folder", default="./data/", help="the input folder")
    parser.add_argument("--out_folder", default="./output/hybridCNN/", help="the output folder")
    parser.add_argument("--batch_size", help="the batch size", type=int, default=512)
    parser.add_argument("--epochs", help="the number of epochs", default=100)

    parser.add_argument("--pooling_type", help="the pooling type", default="avg")
    parser.add_argument(
        "--conv_layers",
        help="the number of the large convolutional layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--conv_repeat",
        help="the number of the convolutional conv block",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--kernel_number", help="the number of the kernels", default=1024
    )
    parser.add_argument("--kernel_size", help="the size of the kernels", default=3)
    parser.add_argument("--sample_rate", help="the sample rate", default=1)
    parser.add_argument("--kernel_length", help="the length of the kernels", default=5)
    parser.add_argument("--pooling_size", help="the size of the pooling", default=2)
    parser.add_argument("--split_ratio", help="the split ratio", default=0.99)
    parser.add_argument("--mixup", help="the mixup augmentation", default="False")
    parser.add_argument("--loss", help="the loss function", default="mse")

    parser.add_argument(
        "--first_conv_activation",
        help="the activation function of first conv layer",
        default="relu",
    )
    parser.add_argument(
        "--dilated_conv_layers", help="the number of dilation layer", default=2
    )
    parser.add_argument("--dilation", help="spacing between kernel elements", default=2)
    parser.add_argument("--float_weight", help="weight on float data", default=1)
    parser.add_argument("--optimizer", help="the optimizer", default="adam")
    parser.add_argument(
        "--lr_scheduler",
        help="the learning rate scheduler",
        default="ReduceLROnPlateau",
    )

    parser.add_argument("--dilation_list", help="the list of dilation values", default="1 2 4 6")
    args = parser.parse_args()

    model_type = args.model_type
    # Set the hyperparameters
    in_folder = args.in_folder
    out_folder = args.out_folder
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    pooling_type = args.pooling_type
    conv_layers = int(args.conv_layers)
    conv_repeat = int(args.conv_repeat)
    kernel_number = int(args.kernel_number)
    sample_rate = float(args.sample_rate)
    kernel_length = int(args.kernel_length)
    kernel_size = int(args.kernel_size)

    batch_size = int(args.batch_size)
    pooling_size = int(args.pooling_size)
    split_ratio = float(args.split_ratio)
    mixup = eval(args.mixup)
    loss = args.loss

    first_conv_activation = args.first_conv_activation
    dilated_conv_layers = int(args.dilated_conv_layers)
    dilation = int(args.dilation)
    dilation_list = [int(dilation) for dilation in args.dilation_list.split(' ')]

    float_weight = float(args.float_weight)
    optimizer = args.optimizer
    lr_scheduler = args.lr_scheduler

    shuffle = True
    num_workers = 5
    persistent_workers = True
    padding = "same"

    dropout = 0.2
    patience = 5

    train_input_path = in_folder + "train_sequences.txt"
    test_input_path = in_folder + "test_sequences.txt"
    output_path = (
        out_folder
        + str(kernel_number)
        + "_"
        + str(conv_layers)
        + "_"
        + str(pooling_size)
        + "_"
        + str(kernel_length)
        + "_"
        + str(conv_repeat)
        + "_"
        + str(pooling_type)
        + "_"
        + str(dropout)
        + "_"
        + str(batch_size)
        + "_"
        + str(sample_rate)
        + "_"
        + str(kernel_size)
        + "_"
        + str(mixup)
        + "_"
        + str(split_ratio)
        + "_"
        + str(loss)
        + "_"
        + str(first_conv_activation)
        + "_"
        + str(dilation_list)
        + "_"
        + str(float_weight)
        + "_"
        + str(optimizer)
        + "_"
        + str(lr_scheduler)
    )
    checkpoint_path = output_path + "/checkpoint"
    prediction_path = output_path + "/prediction"
    log_path = out_folder + "/log"
    trainloader, validloader = data.load_data(
        train_input_path,
        batch_size=batch_size,
        shuffle=shuffle,
        split_ratio=split_ratio,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        sample_rate=sample_rate,
        weight_mode=True,
        float_weight=float_weight,
    )
    if model_type == "Residual_CNN":
        model = models.Residual_CNN(
            kernel_number=kernel_number,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            padding=padding,
            kernel_length=kernel_length,
            conv_repeat=conv_repeat,
            pooling_type=pooling_type,
            kernel_size=kernel_size,
            dropout=dropout,
            mixup=mixup,
            loss=loss,
            first_conv_activation=first_conv_activation,
            dilation=dilation,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    elif model_type == "Dilated_CNN":
        model = models.Dilated_CNN(
            kernel_number=kernel_number,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            padding=padding,
            kernel_length=kernel_length,
            conv_repeat=conv_repeat,
            pooling_type=pooling_type,
            kernel_size=kernel_size,
            dropout=dropout,
            mixup=mixup,
            loss=loss,
            first_conv_activation=first_conv_activation,
            dilated_conv_layers=dilated_conv_layers,
            dilation=dilation,
        )
    elif model_type == "Dilated_CNN_2":
        model = models.Dilated_CNN_2(
            kernel_number=kernel_number,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            padding=padding,
            kernel_length=kernel_length,
            conv_repeat=conv_repeat,
            pooling_type=pooling_type,
            kernel_size=kernel_size,
            dropout=dropout,
            mixup=mixup,
            loss=loss,
            first_conv_activation=first_conv_activation,
            dilation=dilation,
        )
    elif model_type == "Dilated_CNN_3":
        model = models.Dilated_CNN_3(
            kernel_number=kernel_number,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            padding=padding,
            kernel_length=kernel_length,
            conv_repeat=conv_repeat,
            pooling_type=pooling_type,
            kernel_size=kernel_size,
            dropout=dropout,
            mixup=mixup,
            loss=loss,
            first_conv_activation=first_conv_activation,
            dilation=dilation,
        )
    elif model_type == "Hybrid_CNN":
        model = models.Hybrid_CNN(
            kernel_number=kernel_number,
            conv_layers=conv_layers,
            pooling_size=pooling_size,
            padding=padding,
            kernel_length=kernel_length,
            conv_repeat=conv_repeat,
            pooling_type=pooling_type,
            kernel_size=kernel_size,
            dropout=dropout,
            mixup=mixup,
            loss=loss,
            first_conv_activation=first_conv_activation,
            dilation_list=dilation_list,
        )
    else:
        raise ValueError("model type not supported")

    es = EarlyStopping(monitor="val_loss", patience=patience)
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor="val_corr", mode="max", save_top_k=2
    )
    lr_monitor = LearningRateMonitor()
    logger = TensorBoardLogger(log_path + "/tf_logs", name="hybrid_cnn")

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=[es, checkpoint_callback, lr_monitor],
        benchmark=False,
        profiler="simple",
        logger=logger,
    )

    trainer.fit(model, trainloader, validloader)

    testloader = data.load_data(
        test_input_path,
        batch_size=batch_size,
        sample_rate=sample_rate,
        num_workers=num_workers,
        test_mode=True,
    )
    prediction = trainer.predict(ckpt_path="best", dataloaders=testloader)
    prediction = torch.cat(prediction, dim=0).cpu().numpy()
    # np.save(prediction_path + "/test_prediction", prediction)
    
    tools.generate_submission_txt(prediction, test_input_path, "./final_prediction_2.txt")
