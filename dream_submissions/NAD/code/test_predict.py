import numpy as np
import torch
from Util.SimpleViTModel import Model
from Util.Utils import transform_data_embed
import pandas as pd
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.deterministic = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DataGen(Dataset):
    def __init__(self,):
        embedding = pd.read_table('./data/vectors.txt', sep=' ', header=None)
        embedding = embedding.iloc[:(len(embedding) - 1)]
        self.embedding_dict = {embedding.iloc[i, 0]: np.array(list(embedding.iloc[i, 1:].values))
                               for i in range(len(embedding))}

        data = pd.read_csv('./data/test_sequences.txt', header=None, sep='\t')
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = transform_data_embed(self.data.iloc[idx, 0], self.embedding_dict)
        return torch.as_tensor(dat, dtype=torch.float32)


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    model = Model()
    model.load_state_dict(torch.load('./best.pth', map_location=device))
    model.to(device)

    test_data = DataGen()
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, drop_last=False, num_workers=8)

    all_output = []
    model.eval()

    for idx, dat in enumerate(test_loader):
        dat = dat.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(dat).reshape(-1)
        all_output.append(output)

    all_output = torch.cat(all_output, 0).cpu().numpy()
    all_output = np.argsort(all_output).argsort() / all_output.shape[0]

    data = pd.read_csv('./data/test_sequences.txt', header=None, sep='\t')
    data[1] = all_output
    data.to_csv('./prediction.txt', header=None, sep='\t', index=False)
