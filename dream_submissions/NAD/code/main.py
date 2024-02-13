from Util.SimpleViTModel import Model
from torch import optim
from torch.utils.data import DataLoader
import scipy.stats
from Util.Utils import *
from Util.sam import SAM

torch.backends.cudnn.deterministic = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    em_file = './data/vectors.txt'
    l1_beta = 0.1 * 100
    sd = 0.05
    n_epoch = 50
    lr = 2e-4

    train_data = DataGen('train', em_file=em_file, sd=sd, seed=seed)
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True, drop_last=True,
                              num_workers=8, pin_memory=True)

    vali_data = DataGen('val', em_file=em_file, sd=sd, seed=seed)
    vali_loader = DataLoader(vali_data, batch_size=4096, shuffle=False, drop_last=False,
                             num_workers=8, pin_memory=True)

    model = Model()
    model.to(device)
    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, alpha=0.05, rho=0.1,
                    lr=lr, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.LinearLR(optimizer.base_optimizer, start_factor=1, end_factor=0.01,
                                            total_iters=n_epoch)
    scaler = GradScalerSAM()

    best = -1
    best_weights = None
    for epoch in range(n_epoch):
        train_sam(train_loader, model, optimizer, epoch, scaler, device, l1_beta=l1_beta)
        scheduler.step()

        all_out = validation(vali_loader, model, epoch, device).reshape(-1)
        label = vali_data.data.iloc[:, 1].values.reshape(-1)

        print('    {:15s}: {}'.format('Pearson Correlation', scipy.stats.pearsonr(label, all_out)[0]))
        print('    {:15s}: {}'.format('Spearman Correlation', scipy.stats.spearmanr(label, all_out)[0]))

        if scipy.stats.pearsonr(label, all_out)[0] > best:
            best = scipy.stats.pearsonr(label, all_out)[0]
            best_weights = model.state_dict().copy()
            torch.save(best_weights, './best.pth')
