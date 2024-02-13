from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from seq2exp.layers import (
    ConvBlock,
    GELU,
    Residual,
    AttentionPool,
    RELU0_17,
    Attention,
    SoftmaxPool,
    mixup_augmentation,
    negative_pearson_loss,
    weighted_mse_loss,
    HybridConvBlockClass,
)


class Base(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def trainning_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        return NotImplementedError

    def validation_step(self, batch, batch_idx):

        return NotImplementedError

    def test_step(self, batch, batch_idx):
        return NotImplementedError


class Residual_CNN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Residual_CNN")
        parser.add_argument("--pooling_type", help="the pooling type", default="avg")
        parser.add_argument(
            "--conv_layers",
            help="the number of the large convolutional layers",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--conv_repeat",
            help="the number of the convolutional conv block",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--kernel_number", help="the number of the kernels", type=int,default=2048
        )
        parser.add_argument("--kernel_size", help="the size of the kernels", type=int,default=3)
        parser.add_argument("--pooling_size", help="the size of the pooling",type=int, default=2)
        parser.add_argument(
            "--kernel_length", help="the length of the kernels", type=int,default=10
        )
        parser.add_argument("--loss", help="the loss function", default="mse")
        parser.add_argument(
            "--first_conv_activation",
            help="the activation function of first conv layer",
            default="relu",
        )
        parser.add_argument(
            "--dilation", help="spacing between kernel elements", type=int,default=2
        )
        parser.add_argument(
            "--dilated_conv_layers", help="the number of dilation layer", type=int,default=2
        )
        parser.add_argument("--optimizer", help="the optimizer", default="adam")
        parser.add_argument(
            "--lr_scheduler",
            help="the learning rate scheduler",
            default="ReduceLROnPlateau",
        )
        parser.add_argument("--mixup", help="the mixup augmentation", default=False)
        return parent_parser

    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        first_conv_activation="relu",
        dilation=1,
        optimizer="adam",
        lr_scheduler="ReduceLROnPlateau",
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            activation=first_conv_activation,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))
        # x1 = self.conv0(x)

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        # print(x1.shape)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            raise ValueError("Unknown optimizer, please choose from adam and sgd")
        if self.hparams.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    # TODO: add an argument to control the patience
                    optimizer,
                    patience=1,
                ),
                "monitor": "val_loss",
            }
        elif self.hparams.lr_scheduler == "ExponentialLR":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.5
                ),
                "monitor": "val_loss",
            }
        else:
            raise ValueError(
                "Unknown learning rate scheduler, please choose from ReduceLROnPlateau and ExponentialLR"
            )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


class CNN(nn.Module):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=15,
        hidden_size=256,
        dropout=0.2,
    ) -> None:
        super(CNN, self).__init__()
        self.kernel_number = kernel_number
        self.kernel_length = kernel_length
        self.hidden_size = hidden_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, int(kernel_number / 2), kernel_length, padding="same"),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(kernel_number, kernel_number, kernel_length, padding="same"),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(kernel_number, kernel_number, kernel_length, padding="same"),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(kernel_number, kernel_number, kernel_length, padding="same"),
            nn.ReLU(),
        )
        # TODO don't hardcode the number of features
        self.fc1 = nn.Sequential(
            nn.Linear(142 * kernel_number, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        # print(x.shape)

        x_rc = torch.flip(x, [1])

        x1 = self.conv1(torch.concat([x, x_rc]))

        x_all = self.conv2(x1, dim=1)

        x_all = self.conv3(x_all)
        x_all = self.conv4(x_all)
        x_all = torch.flatten(x_all, 1)
        x_all = self.fc1(x_all)
        x_all = self.fc2(x_all)
        x_all = self.out(x_all)
        return x_all.flatten()


# Transformer (borrow code from enformer-torch, mimic architecture in Vaishnav et. al, 2022)
class Transformer(pl.LightningModule):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="max",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        attention_layers=2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # This conv layer is appied on both forward and RC strands
        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )

        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.attentionlayers = nn.ModuleList()

        for layer in range(attention_layers):
            self.attentionlayers.append(
                nn.Sequential(
                    Residual(
                        Attention(
                            dim=filter_number,  # dimension of the last out channel
                        ),
                    ),
                    nn.LayerNorm(filter_number),
                    Residual(
                        nn.Sequential(
                            nn.Linear(filter_number, filter_number * 2),
                            nn.Dropout(dropout),
                            nn.ReLU(),
                            nn.Linear(filter_number * 2, filter_number),
                            nn.Dropout(dropout),
                        )
                    ),
                    nn.LayerNorm(filter_number),
                )
            )

        # LSTM, output is (batch_size, 113, 2*8)
        # self.lstm = nn.LSTM(64, 8, bidirectional=True, batch_first=True, dropout=0.05)

        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x):  # X is (batch_size, 142, 4)
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = torch.permute(
            x1, (0, 2, 1)
        )  # (batch_size, 113, 64) = (batch_size, seq_len, dim)
        # Attention layer
        for layer in self.attentionlayers:
            x1 = layer(x1)

        # flatten
        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=1,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


# Dialted CNN ---------------------------------------------------------


class Dilated_CNN(pl.LightningModule):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="max",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        first_conv_activation="relu",
        dilated_conv_layers=2,
        dilation=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            activation=first_conv_activation,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )
            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.dilated_convlayers = nn.ModuleList()
        for layer in range(dilated_conv_layers):
            self.dilated_convlayers.append(
                Residual(
                    nn.Sequential(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        ConvBlock(
                            filter_number, filter_number, 1, padding=padding, dilation=1
                        ),
                        nn.Dropout(dropout),
                    )
                )
            )

        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))

        for layer in self.convlayers:
            x1 = layer(x1)

        for layer in self.dilated_convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)

        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=1,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])




# Dilated_CNN version 2---------------------------------------------


class Dilated_CNN_2(pl.LightningModule):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="max",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        first_conv_activation="relu",
        dilation=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """
        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding, activation=first_conv_activation)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )
            self.convlayers.append(
                Residual(
                    nn.Sequential(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation
                        ),
                        ConvBlock(
                            filter_number,
                            filter_number,
                            1,
                            padding=padding,
                            dilation=1
                        ),
                        nn.Dropout(dropout)
                    )
                )
            )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )
            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)


        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))
    
        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=1,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])




# Dilated_CNN version 3---------------------------------------------


class Dilated_CNN_3(pl.LightningModule):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="max",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        first_conv_activation="relu",
        dilation=1,
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """
        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(4, kernel_number, kernel_length, padding=padding, activation=first_conv_activation)

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                        )
                    )
                )

            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

            self.convlayers.append(
                Residual(
                    nn.Sequential(
                        ConvBlock(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation=dilation
                        ),
                        ConvBlock(
                            filter_number,
                            filter_number,
                            1,
                            padding=padding,
                            dilation=1
                        ),
                        nn.Dropout(dropout)
                    )
                )
            )
        

        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))
    
        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=1,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            # loss = weighted_mse_loss(y_hat, y, w)
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])




# Hybrid_CNN ------------------------------------------------------------

class Hybrid_CNN(pl.LightningModule):
    def __init__(
        self,
        kernel_number=512,
        kernel_length=19,
        kernel_size=3,
        pooling_size=2,
        conv_layers=4,
        hidden_size=256,
        dropout=0.2,
        padding="same",
        h_layers=2,
        input_length=142 * 2,
        filter_number=256,
        pooling_type="avg",
        learning_rate=1e-3,
        conv_repeat=1,
        mixup=False,
        loss="mse",
        first_conv_activation="relu",
        dilation_list=None,
        optimizer="adam",
    ) -> None:
        """
        Args:
            kernel_number: number of kernels in the convolutional layer
            kernel_length: length of the kernel
            kernel_size: size of the kernel
            pooling_size: size of the pooling
            conv_layers: number of convolutional layers
            hidden_size: number of hidden units in the fully connected layer
            dropout: dropout rate
            padding: padding type
            h_layers: number of hidden layers
            input_length: length of the input sequence
            filter_number: number of filters in the later convolutional layer
            pooling_type: type of pooling
            learning_rate: learning rate
            conv_repeat: number of repeats of the convolutional layer
        """

        super().__init__()
        self.save_hyperparameters()

        self.conv0 = ConvBlock(
            4,
            kernel_number,
            kernel_length,
            padding=padding,
            activation=first_conv_activation,
        )

        self.convlayers = nn.ModuleList()
        self.convlayers.append(
            ConvBlock(
                kernel_number,
                filter_number,
                kernel_size,
                padding=padding,
            )
        )
        fc_dim = input_length
        for layer in range(conv_layers):
            for repeat in range(conv_repeat):
                self.convlayers.append(
                    Residual(
                        HybridConvBlockClass(
                            filter_number,
                            filter_number,
                            kernel_size,
                            padding=padding,
                            dilation_list=dilation_list,
                        )
                    )
                )
            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "attention":
                self.convlayers.append(AttentionPool(kernel_number, pooling_size))
            elif pooling_type == "softmax":
                self.convlayers.append(SoftmaxPool(kernel_number, pooling_size))
            else:
                raise ValueError(
                    "Unknown pooling type, please choose from max, avg, attention, softmax"
                )

            # Calculate the fc_dim
            fc_dim = ceil(fc_dim / pooling_size)

        self.fc0 = nn.Sequential(nn.Linear(fc_dim * filter_number, hidden_size), GELU())

        self.fclayers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), GELU(), nn.Dropout(dropout)
            )
            for layer in range(h_layers)
        )
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x_rc = torch.flip(x, [1, 2])
        x1 = self.conv0(torch.concat([x, x_rc], dim=-1))
        # x1 = self.conv0(x)

        for layer in self.convlayers:
            x1 = layer(x1)

        x1 = x1.flatten(1)
        x1 = self.fc0(x1)
        # print(x1.shape)
        for layer in self.fclayers:
            x1 = layer(x1)

        x1 = self.out(x1)
        x1 = x1.flatten()
        return x1

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            raise ValueError("Unknown optimizer, please choose from adam and sgd")
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                # TODO: add an argument to control the patience
                optimizer,
                patience=1,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # return the loss
        x, y = batch
        if self.hparams.mixup:
            x, y = mixup_augmentation(x, y)
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == "mse":
            loss = F.mse_loss(y_hat, y)
        elif self.hparams.loss == "pearson":
            loss = negative_pearson_loss(y_hat, y)
        self.log("val_loss", loss)

        return torch.stack([y_hat.cpu().detach(), y.cpu().detach()])

    def validation_epoch_end(self, validation_step_outputs):

        all_y = torch.cat(validation_step_outputs, dim=1)
        valid_corr = torch.corrcoef(all_y)[0, 1]

        self.log("val_corr", valid_corr)
        self.log("hp_metric", valid_corr)
        return all_y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])