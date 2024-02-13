#!/home/dingml/anaconda3/envs/expressBert39/bin/python3
import glob
import gzip
import pickle
import argparse
import os
import sys
import numpy as np
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import torch
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from dataloaders_expr import ExpressionPretrainData
from transformers import (
    AutoConfig, AutoModelForMaskedLM,
    BertConfig, BertForMaskedLM,
    DistilBertConfig, DistilBertForMaskedLM,
    AlbertConfig, AlbertForMaskedLM,
    BigBirdConfig, BigBirdForMaskedLM
)
from train_utils import (make_logger, make_directory, get_run_info)
from train_utils import model_summary, get_device, set_seed


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def train(
        model: nn.Module,
        dataset: ExpressionPretrainData,
        outdir: str,
        fraction_per_epoch: float = 0.1,
        lr=5e-5,
        batch_size=128,
        max_epoch=1000,
        patience=30,
        resume: bool = True,
        num_workers: int = 8,
        debug: bool = False,
        # contrast: bool=False,
        is_data_parallel=False,
        seed=2020,
        **kwargs
):
    device = next(model.parameters()).device
    set_seed(seed)

    ## split dataset
    train_inds, val_inds = train_test_split(range(len(dataset)), test_size=min(100000, int(len(dataset) / 20)))
    logger.info("train/validation size: {}/{}".format(len(train_inds), len(val_inds)))

    torch.save({
        "total": len(dataset),
        "val": val_inds,
    }, os.path.join(outdir, "split.pt"))
    logger.info(f"train/val: {len(train_inds)}\t{len(val_inds)}")

    train_loader = DataLoader(
        Subset(dataset, indices=train_inds),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        Subset(dataset, indices=val_inds),
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )

    ## optimizer & lr scheduler & amp scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    logger.info("{}".format(optimizer))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=0.0001,
        min_lr=1e-6
    )

    scaler = GradScaler()

    ## prepare training
    ckpt = os.path.join(outdir, "checkpoint.pt")
    best_model = os.path.join(outdir, "best_model.pt")

    ## whether resume training
    if os.path.exists(ckpt) and resume:
        logger.info("- load checkpoint from {}".format(ckpt))
        d = torch.load(ckpt, map_location=device)
        if is_data_parallel:
            model.module.load_state_dict(d["model_state_dict"])
        else:
            model.load_state_dict(d["model_state_dict"])
        optimizer.load_state_dict(d["optimizer_state_dict"])
        scheduler.load_state_dict(d["scheduler_state_dict"])
        scaler.load_state_dict(d["scaler_state_dict"])
        best_acc = d["best_acc"]
        val_acc = d["val_acc"]
        wait = d["wait"]
        epoch = d["epoch"]
    else:
        best_acc = -1
        wait = 0
        epoch = -1
        if os.path.exists(ckpt):
            os.remove(ckpt)

    start = epoch + 1
    batches_per_epoch = round(fraction_per_epoch * len(train_loader))
    train_iters = iter(cyclic_iter(train_loader))
    for epoch in range(start, max_epoch):
        model.train()
        pbar = tqdm(range(batches_per_epoch), desc="Epoch{}".format(epoch), total=batches_per_epoch)

        if epoch > start:
            torch.save({
                "model_state_dict": model.module.state_dict() if is_data_parallel else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "wait": wait,
            }, ckpt)
            scheduler.step(metrics=val_acc)

        epoch_loss = 0  # total, MLM, contrast loss
        ## train the model
        for it in pbar:
            input_ids, labels, masks = next(train_iters)

            if epoch == 0 and it == 0 and debug:
                print("data size: {}".format((input_ids.size(), labels.size()), flush=True, file=sys.stderr))
                torch.save((input_ids, labels), "{}/sample_data.pt".format(outdir))

            input_ids, labels, masks = input_ids.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model.forward(input_ids=input_ids, attention_mask=masks, labels=labels)
                loss = outputs.loss.mean()

            epoch_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)  # .step()
            scaler.update()

            pbar.set_postfix_str("loss/lr={:.4f}/{:.3g}".format(epoch_loss / (it + 1), optimizer.param_groups[0]["lr"]),
                                 refresh=False)

        ## validation
        val_acc, val_logits, val_targets = test(model, val_loader)
        val_acc = np.mean(val_acc)
        logger.info("validation({}) MLM-ACC: {:.4f}".format(epoch, val_acc))
        if val_acc > best_acc * 1.0001:
            wait = 0
            best_acc = val_acc
            if hasattr(model, "module"):
                model.module.save_pretrained(outdir)
            else:
                model.save_pretrained(outdir)
            logger.info("- model saved {}\n".format(best_model))
        else:
            wait += 1
            logger.warning("- wait{}\n".format(wait))
            if wait >= patience:
                logger.warning("- early stopped{}\n".format(wait))
                break


@torch.no_grad()
def test(model: nn.Module, dataloader: DataLoader) -> list:
    model.eval()
    device = get_device(model)
    test_acc = list()
    test_acc_all = list()
    pbar = tqdm(dataloader, desc="predicting")
    predicts = list()
    targets = list()
    accum_acc = 0
    accum_acc_all = 0
    for it, (input_ids, labels, masks) in enumerate(pbar):
        input_ids, labels, masks = input_ids.to(device), labels.to(device), masks.to(device)
        # input_ids, labels = input_ids.to(device), labels.to(device)
        with autocast():
            outputs = model.forward(input_ids=input_ids, attention_mask=masks, labels=None)
            outputs = torch.argmax(outputs.logits, dim=2).detach().cpu().numpy().reshape(-1)
            labels = labels.detach().cpu().numpy().reshape(-1)
            test_acc_all.append(accuracy_score(
                np.maximum(input_ids.detach().cpu().numpy().reshape(-1), labels),
                outputs
            ))
            accum_acc_all += test_acc_all[-1]
            keep = np.where(labels != -100)[0]
            labels, outputs = labels[keep], outputs[keep]
            predicts.append(outputs.astype(np.int8))
            targets.append(labels.astype(np.int8))
            test_acc.append(accuracy_score(labels, outputs))
            accum_acc += test_acc[-1]
        pbar.set_postfix_str("ACC(mlm/all)={:.4f}/{:.4f}".format(
            accum_acc / (it + 1), accum_acc_all / (it + 1)), refresh=False)

    predicts = np.concatenate(predicts)
    targets = np.concatenate(targets)
    return test_acc, predicts, targets


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-fa', "--fasta", required=True)

    # data
    p.add_argument("--size", type=int, default=3000, help="sequence length")
    p.add_argument("--disable-special-tokens", action="store_true", help="not add [CLS] and [SEP] tokens")

    ## model config:
    p.add_argument('-n', "--model-type", choices=("bert", "distilbert", "albert", "big_bird"), required=False,
                   default="bert", help="model type")
    p.add_argument("--hidden-size", type=int, default=512, help="dimension")
    p.add_argument("--num-hidden-layers", type=int, default=3, help="layers")
    p.add_argument("--num-attention-heads", type=int, default=16, help="head")

    ## mlm config:
    p.add_argument('-k', '--token-size', type=int, default=3, help="token size")
    p.add_argument("--mlm-rate", type=float, default=0.15, help="mlm rate")
    p.add_argument("--rand-rate", type=float, default=0.1, help="default 10%%")
    p.add_argument("--unchange-rate", type=float, default=0.1, help="default 10%%")

    ## train config:
    p.add_argument("--resume", action="store_true", help="resume training using checkpoint")
    # p.add_argument("--n-gpu", type=int, default=1)
    p.add_argument("--device", type=str, required=True,
                   help="device, 'cpu' or integer, e.g.: --device cpu/--device 0/--device 0,1")
    # p.add_argument("--contrast", action="store_true")
    p.add_argument("--fraction-per-epoch", default=1, type=float, help="fration of data used per epoch")
    p.add_argument("--max-epoch", type=int, default=1000, help="max epoch number")
    p.add_argument("--batch-size", type=int, default=8, help="batch size")
    p.add_argument("-lr", default=1e-4, type=float, help="learning rate")
    p.add_argument("--patience", type=int, default=20, help="early stop patience")
    p.add_argument("--num-workers", type=int, default=8, help="num workers in DataLoader")
    p.add_argument("--flank", type=int, default=0, help="flank length of the seq")

    ## output & cache
    p.add_argument('-o', "--outdir", type=str, required=True, help="output directory")
    # p.add_argument("--cache-dir", required=True, help="cache")

    ## debug
    p.add_argument("--debug", action="store_true")
    p.add_argument("--random-genome", action="store_true")
    p.add_argument("--seed", type=int, default=2022)

    # top
    p.add_argument("--top", type=float, default=0.1)
    return p


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    set_seed(args.seed)

    outdir = make_directory(args.outdir)
    logger = make_logger(filename="{}/train.log".format(outdir))

    logger.info(get_run_info(argv=sys.argv, args=args))

    ## data
    dataset = ExpressionPretrainData(
        fasta=args.fasta,
        flank=args.flank,
        expr_percent=args.top,
    )
    logger.info("- sample size: {}".format(len(dataset)))
    tokenizer_config = dataset.tokenizer.save_pretrained(outdir)
    logger.info("tokenizer config saved to {}: {}".format(outdir, tokenizer_config))

    ## setup GPU
    n_gpu = 1
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if ',' in args.device:
            n_gpu = len(args.device.strip(',').split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda")
    logger.info("- device: {}({})".format(device, args.device))

    ## setup model
    config = AutoConfig.for_model(
        args.model_type,
        vocab_size=dataset.tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        output_hidden_states=True,
        intermediate_size=4 * args.hidden_size
    )
    logger.info("- config: {}".format(config))
    model = AutoModelForMaskedLM.from_config(config)
    if hasattr(model, "module"):
        model.module.save_pretrained(outdir)
    else:
        model.save_pretrained(outdir)

    model = model.to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model)
        is_data_parallel = True
    else:
        is_data_parallel = False

    logger.info("- model: {}\n{}\n".format(model, model_summary(model)))

    ## pretrain
    train(
        model,
        dataset,
        outdir=outdir,
        batch_size=args.batch_size,
        lr=args.lr,
        fraction_per_epoch=args.fraction_per_epoch,
        seed=args.seed,
        resume=args.resume,
        max_epoch=args.max_epoch,
        patience=args.patience,
        is_data_parallel=is_data_parallel,
        n_gpu=n_gpu,
        num_workers=args.num_workers,
        # contrast=args.contrast
    )
