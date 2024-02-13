import numpy as np
import pandas as pd
from seq2exp import tools
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


# TODO add  10 fold cross validation
def load_data(
    file_path,
    split_ratio=0.8,
    batch_size=256,
    shuffle=True,
    num_workers=12,
    sample_rate=1,
    length=142,
    persistent_workers=False,
    test_mode=False,
    weight_mode=False,
    float_weight=1,
):
    """Load data from file."""
    if test_mode:
        test_data = pd.read_csv(file_path, header=None, sep="\t")
        test_data.columns = ["sequence", "exp"]
        test_seq = tools.DF2array(test_data, "sequence", length=length)
        x_all = torch.from_numpy(test_seq).float()
        test_dataset = torch.utils.data.TensorDataset(x_all)
        testloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        return testloader
    else:
        train_data = pd.read_csv(file_path, header=None, sep="\t")
        train_data.columns = ["sequence", "exp"]
        train_seq = tools.DF2array(train_data, "sequence", length=length)

        x_all = torch.from_numpy(train_seq).float()
        y_all = torch.from_numpy(train_data["exp"].values).float()

        weight_mode = weight_mode and (float_weight != 1)
        if weight_mode:
            weight_all = np.zeros(len(train_data["exp"].values))
            weight_all[
                train_data["exp"].values == train_data["exp"].values.astype(int)
            ] = 1
            weight_all[
                train_data["exp"].values != train_data["exp"].values.astype(int)
            ] = float_weight
            weight_all = torch.from_numpy(weight_all).float()

        # random subsample index x_all and y_all
        if sample_rate < 1:
            sample_size = int(sample_rate * len(x_all))
            idx = np.random.choice(len(x_all), sample_size, replace=False)
            x_all = x_all[idx]
            y_all = y_all[idx]
            if weight_mode:
                weight_all = weight_all[idx]

        if weight_mode:
            Sequence_data = torch.utils.data.TensorDataset(x_all, y_all, weight_all)
        else:
            Sequence_data = torch.utils.data.TensorDataset(x_all, y_all)

        n_sample = x_all.shape[0]
        n_train_sample = int(n_sample * split_ratio)
        train_dataset, val_dataset = torch.utils.data.random_split(
            Sequence_data,
            [n_train_sample, n_sample - n_train_sample],
            generator=torch.Generator().manual_seed(42),
        )

        trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        validloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        return trainloader, validloader


def load_data_RC(
    file_path,
    split_ratio=0.8,
    batch_size=256,
    shuffle=True,
    num_workers=32,
    use_weight=False,
):
    """Load data from file."""
    train_data = pd.read_csv(file_path, header=None, sep="\t")
    train_data.columns = ["sequence", "exp"]
    train_seq = tools.DF2array_RC(train_data, "sequence", length=142)

    x_all = torch.from_numpy(train_seq)
    y_all = torch.from_numpy(train_data["exp"].values)

    Sequence_data = torch.utils.data.TensorDataset(x_all, y_all)

    n_sample = x_all.shape[0]
    n_train_sample = int(n_sample * split_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        Sequence_data,
        [n_train_sample, n_sample - n_train_sample],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    validloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, validloader


def load_test_data(file_path, batch_size=256, num_workers=32):
    """Load data from file."""
    test_data = pd.read_csv(file_path, header=None, sep="\t")
    test_data.columns = ["sequence", "exp"]
    train_seq = tools.DF2array_RC(test_data, "sequence", length=142)

    x_all = torch.from_numpy(train_seq)

    test_dataset = torch.utils.data.TensorDataset(x_all)
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return testloader
