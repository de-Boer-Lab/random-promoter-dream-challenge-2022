#!/home/dingml/anaconda3/envs/expressBert39/bin/python3
import ast
from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from biodml import select_device, set_rand_seed
from biodml import split_train_val_test_by_group, split_train_valid_test
from train_utils import model_summary, make_logger, make_directory, get_run_info
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from scipy.stats import pearsonr, spearmanr

from transformers import BertTokenizer, BertTokenizerFast, AutoConfig, AutoModelForMaskedLM, \
    AutoModelForSequenceClassification
from transformers import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from dataloaders_expr import ExpressionTrainData
from biodml import copen
from math import ceil
from train_utils import cal_sampler_prob


@torch.no_grad()
def test_model(model, loader):
    model.eval()
    pred, true = list(), list()
    for it, (seq, label) in enumerate(tqdm(loader, desc="predicting", total=len(loader))):
        bs = seq.size(0)
        seq = seq.long().to(device)
        seq = seq.reshape(bs * 3, -1)
        with autocast():
            out = model.forward(seq).logits.reshape(bs, 3, -1).mean(dim=1)
            del seq
        out = out.cpu().numpy()
        label = label.numpy()
        pred.append(out.astype(np.float16))
        true.append(label.astype(np.float16))
    pred = np.concatenate(pred).reshape((-1,))
    true = np.concatenate(true).reshape((-1,))
    pcc = pearsonr(x=true.T, y=pred.T)[0]
    scc = spearmanr(a=true.T, b=pred.T)[0]
    return pcc, scc, pred, true


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-f", "--fasta", required=True)
    p.add_argument("-o", required=True)
    p.add_argument("-d", '--device')
    p.add_argument("-lr", type=float, default=1e-4)
    p.add_argument('-b', "--batch_size", default=32, type=int)
    p.add_argument('--sample', type=float, default=1.0)
    p.add_argument("--flank", type=int, default=0, help="flank length of the seq")
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--model_path', type=str, help="path of the pretrain model", required=True)
    p.add_argument('--grouped', action="store_true")
    p.add_argument('--weight_sampler', action="store_true", help="whether weight sample")
    return p


if __name__ == "__main__":
    args = get_args().parse_args()
    outdir = make_directory(args.o)
    logger = make_logger(filename="{}/train.log".format(outdir))
    logger.info(get_run_info(argv=sys.argv, args=args))

    # setup GPU
    n_gpu = 1
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        if ',' in os.environ["CUDA_VISIBLE_DEVICES"]:
            n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].strip(',').split(','))
    logger.info("- device: {}".format(device))

    # tokenizer = BertTokenizer(vocab_file="../../vocab/dna_vocab_3bp.txt", do_lower_case=False)
    model_path = args.model_path
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")), strict=False)
    model = model.to(device)

    batch_size = args.batch_size

    dataset = ExpressionTrainData(
        fasta=args.fasta,
        flank=150,
        grouped=args.grouped,
    )
    if args.grouped:
        if os.path.exists(os.path.basename(f"{outdir}/split.pt")):
            train_inds, val_inds, test_inds = torch.load(f"{outdir}/split.pt")
            assert len(train_inds) + len(val_inds) + len(test_inds) == len(dataset)
        else:
            set_rand_seed(args.seed)
            train_inds, val_inds, test_inds = split_train_val_test_by_group(dataset.groups, n_splits=10, val_folds=8, test_folds=9)
            torch.save((train_inds, val_inds, test_inds), "{}/split.pt".format(outdir))
    else:
        if os.path.exists(os.path.basename(f"{outdir}/split.pt")):
            train_inds, val_inds, test_inds = torch.load(f"{outdir}/split.pt")
            assert len(train_inds) + len(val_inds) + len(test_inds) == len(dataset)
        else:
            set_rand_seed(args.seed)
            train_inds, val_inds, test_inds = split_train_valid_test(int(len(dataset) * args.sample), val_ratio=0.1,
                                                                     test_ratio=0.05)
            torch.save((train_inds, val_inds, test_inds), "{}/split.pt".format(outdir))
    logger.info("- train_inds: {}".format(len(train_inds)))
    logger.info("- val_inds: {}".format(len(val_inds)))
    logger.info("- test_inds: {}".format(len(test_inds)))

    if args.weight_sampler:
        if os.path.exists(os.path.join(outdir, 'sampler.pt')):
            sampler_prob = torch.load(os.path.join(outdir, 'sampler.pt'))
        else:
            sampler_prob = np.array(cal_sampler_prob(args.fasta))
            torch.save(sampler_prob, os.path.join(outdir, 'sampler.pt'))
        train_loader = DataLoader(Subset(dataset, indices=train_inds), batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=dataset.collate_fn,
                                  sampler=WeightedRandomSampler(sampler_prob[train_inds], num_samples=len(train_inds)), num_workers=min(8, batch_size))
        val_loader = DataLoader(Subset(dataset, indices=val_inds), batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn,
                                  sampler=WeightedRandomSampler(sampler_prob[val_inds], num_samples=len(val_inds)),num_workers=min(8, batch_size))
        test_loader = DataLoader(Subset(dataset, indices=test_inds), batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn,
                                  sampler=WeightedRandomSampler(sampler_prob[test_inds], num_samples=len(test_inds)), num_workers=min(8, batch_size))
    else:
        train_loader = DataLoader(Subset(dataset, indices=train_inds), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn,
                                  num_workers=min(8, batch_size))
        val_loader = DataLoader(Subset(dataset, indices=val_inds), batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dataset.collate_fn,
                                  num_workers=min(8, batch_size))
        test_loader = DataLoader(Subset(dataset, indices=test_inds), batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn,
                                  num_workers=min(8, batch_size))
    logger.info("model {}\n{}".format(model, model_summary(model)))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    patience = 10
    scaler = GradScaler()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0
    )
    logger.info("optimizer {}".format(optimizer))
    # scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=100,
    #     num_training_steps=len(train_loader)* 100,
    #     lr_end=1e-7,
    #     power=3
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        threshold=1e-4,
        min_lr=1e-7,
        patience=2
    )

    if os.path.exists(f"{outdir}/checkpoint.pt"):
        if n_gpu >= 2:
            logger.info(f"load state from {outdir}/checkpoint.pt")
            d = torch.load(f"{outdir}/checkpoint.pt", map_location=device)
        else:
            logger.info(f"load state from {outdir}/checkpoint.pt")
            d = torch.load(f"{outdir}/checkpoint.pt", map_location=device)
        model.load_state_dict(d["model_state_dict"])
        scheduler.load_state_dict(d["scheduler_state_dict"])
        scaler.load_state_dict(d["scaler_state_dict"])
        optimizer.load_state_dict(d["optimizer_state_dict"])
        start = d["epoch"] + 1
        wait = d["wait"]
        best_pcc = d["best_pcc"]
    else:
        wait = 0
        best_pcc = -1
        start = 0

    fraction_per_epoch = 1
    batches_per_epoch = int(len(train_loader) * fraction_per_epoch)
    train_iters = iter(train_loader)
    cnt = 0
    for epoch in range(start, 300):
        model.train()
        pbar = tqdm(range(batches_per_epoch), total=batches_per_epoch, desc=f"Epoch{epoch}")
        epoch_loss = 0
        for it in pbar:
            if cnt == len(train_loader):
                cnt = 0
                train_iters = iter(train_loader)
            seq, label = next(train_iters)
            cnt += 1
            seq = seq.long().to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            bs = seq.size(0)
            seq = seq.reshape(bs * 3, -1)  # (B, 3, S) -> (B * 3, S)
            with autocast():
                # with torch.no_grad():
                out = model.forward(seq).logits.reshape(bs, 3, -1).mean(dim=1)
                loss = F.smooth_l1_loss(out.squeeze(dim=1).float(), label.float())
                # loss = F.mse_loss(out.squeeze(dim=1).float(), label.float())
                del seq
                epoch_loss += loss.item()

            scaler.scale(loss.double()).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix_str(
                "loss/lr={:.4f}/{:.2e}".format(epoch_loss / (it + 1), optimizer.param_groups[-1]["lr"]))

        val_pcc, val_scc, _, _ = test_model(model, val_loader)
        logger.info(f"validation({epoch})PCC/SCC={np.nanmean(val_pcc):.4f}/{val_scc:.4f}")

        scheduler.step(np.mean(val_pcc))

        if n_gpu >= 2:
            torch.save({
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
                "epoch": epoch,
                "best_pcc": best_pcc,
                "wait": wait
            }, f"{outdir}/checkpoint.pt")
        else:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
                "epoch": epoch,
                "best_pcc": best_pcc,
                "wait": wait
            }, f"{outdir}/checkpoint.pt")
        if np.nanmean(val_pcc) > best_pcc:
            wait = 0
            best_pcc = np.nanmean(val_pcc)
            if n_gpu >= 2:
                model.module.save_pretrained(outdir)
            else:
                model.save_pretrained(outdir)
            logger.info("- model saved\n")
        else:
            wait += 1
            logger.info(f"- wait {wait}\n")
            if wait > patience:
                break

    if n_gpu >= 2:
        model.module.load_state_dict(torch.load(f"{outdir}/pytorch_model.bin"))
    else:
        model.load_state_dict(torch.load(f"{outdir}/pytorch_model.bin"))
    model.eval()
    test_pcc, test_scc, test_pred, test_true = test_model(
        model, test_loader
    )
    logger.info("test PCC/SCC: {:.4f}\t{:.4f}".format(np.nanmean(test_pcc), test_scc))

    torch.save((test_pcc, test_pred, test_true), f"{outdir}/test.pt")
