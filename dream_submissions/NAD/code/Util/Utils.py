from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.cuda.amp.grad_scaler import OptState
from Util.sam import enable_running_stats, disable_running_stats

def transform_data_embed(seq, embedding_dict):
    tmp = [np.array(list(seq[(17 + i):(-13 + i)])).reshape(-1, 1) for i in range(-2, 3)]
    tmp = np.concatenate(tmp, axis=1)
    tmp = [''.join(x) for x in tmp]
    tmp = np.array([embedding_dict[x] for x in tmp])
    return tmp

class DataGen(Dataset):

    def __init__(self, mode='train',
                 em_file='./data/vectors.csv',
                 seed=1,
                 sd=0.1
                 ):
        self.mode = mode
        self.sd = sd
        embedding = pd.read_table(em_file, sep=' ', header=None)
        embedding = embedding.iloc[:(len(embedding) - 1)]
        self.embedding_dict = {embedding.iloc[i, 0]: np.array(list(embedding.iloc[i, 1:].values))
                               for i in range(len(embedding))}

        data = pd.read_csv('./data/train_sequences.txt', header=None, sep='\t')
        data = data[data.iloc[:, 0].apply(len) == 110]
        data = data[data.iloc[:, 0].apply(lambda x: "N" not in x)]
        data.reset_index(drop=True, inplace=True)
        data.columns = range(len(data.columns))

        idx = np.where(data[1].values <= 8)[0]
        data.iloc[idx, 1] = data[1].values[idx] / 8 * 3 + 5
        data[1] = (data[1].values - 5) / 12

        train_data, val_data = train_test_split(data, test_size=0.1, random_state=seed)

        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        if self.mode == 'train':
            self.data = train_data
        if self.mode == 'val':
            self.data = val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = transform_data_embed(self.data.iloc[idx, 0], self.embedding_dict)
        lab = self.data.iloc[idx, 1]

        if self.mode == 'train':
            sd = self.sd
            lab = np.clip(lab + np.random.normal(0, sd), 0, 1)

        return torch.as_tensor(dat, dtype=torch.float32), torch.as_tensor(lab, dtype=torch.float32)


def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()


class GradScalerSAM(torch.cuda.amp.GradScaler):
    def _maybe_opt_step(self, optimizer, optimizer_state, first_step=True, *args, **kwargs):
        retval = None
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            if first_step:
                retval = optimizer.first_step(zero_grad=True, *args, **kwargs)
                retval = "Done"
            else:
                retval = optimizer.second_step(zero_grad=True, *args, **kwargs)
        return retval

    def step(self, optimizer, first_step=True, *args, **kwargs):
        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

        retval = self._maybe_opt_step(optimizer, optimizer_state, first_step=first_step, *args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval


def train_sam(train_loader, model, optimizer, epoch, scaler, device, l1_beta=0.1):
    criteria = torch.nn.SmoothL1Loss(beta=l1_beta)
    print("{} epoch: \t start training....".format(epoch))
    total_loss = []
    all_corr = []
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for dat, lab in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()
            dat = dat.to(device)
            lab = lab.to(device).reshape(-1) * 100

            enable_running_stats(model)
            with torch.cuda.amp.autocast():
                output = model(dat, return_all=True) * 100
                loss = criteria(output, torch.repeat_interleave(lab.reshape(-1, 1), output.shape[1], dim=1))
            scaler.scale(loss).backward()
            retval = scaler.step(optimizer, first_step=True)
            scaler.update()

            if retval is None:
                continue

            disable_running_stats(model)
            with torch.cuda.amp.autocast():
                _ = model(dat, return_all=True) * 100
                _ = criteria(_, torch.repeat_interleave(lab.reshape(-1, 1), _.shape[1], dim=1))
            scaler.scale(_).backward()
            scaler.step(optimizer, first_step=False)
            scaler.update()

            total_loss.append(loss.item())
            output = output.mean(-1)
            all_corr.append(corrcoef(lab, output).item())

            tepoch.set_postfix(loss=np.mean(total_loss), corr=np.mean(all_corr))


def validation(vali_loader, model, epoch, device):
    print("{} epoch: \t start validation....".format(epoch))
    all_corr = []
    all_output = []
    model.eval()
    with tqdm(vali_loader, unit="batch") as tepoch:
        for dat, lab in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            dat = dat.to(device)
            lab = lab.to(device).reshape(-1)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = model(dat).reshape(-1)

            all_output.append(output)
            all_corr.append(corrcoef(lab, output).item())

            tepoch.set_postfix(corr=np.mean(all_corr))

    return torch.cat(all_output, 0).cpu().numpy()
