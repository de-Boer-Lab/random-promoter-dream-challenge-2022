#!/home/dingml/anaconda3/envs/expressBert39/bin/python3

from tqdm import tqdm
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from biodml import set_rand_seed
from train_utils import model_summary, make_logger, make_directory, get_run_info
from torch.cuda.amp.autocast_mode import autocast
from transformers import \
    BertConfig, \
    BertTokenizer, \
    BertTokenizerFast, \
    AutoConfig, \
    AutoModelForMaskedLM, \
    AutoModelForSequenceClassification
from transformers import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup)
from dataloaders_expr import ExpressionTestData

@torch.no_grad()
def test_model(model, loader):
    model.eval()
    pred= list()
    for it, seq in enumerate(tqdm(loader, desc="predicting", total=len(loader))):
        bs = seq.size(0)
        seq = seq.long().to(device)
        seq = seq.reshape(bs * 3, -1)
        with autocast():
            out = model.forward(seq).logits.reshape(bs, 3, -1).mean(dim=1)
            del seq
        out = out.cpu().numpy()
        pred.append(out.astype(np.float16))
    print(len(pred))
    pred = np.concatenate(pred).squeeze()
    return pred


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-f", "--fasta", required=True)
    p.add_argument("-o", required=True)
    p.add_argument('-b', "--batch_size", default=32, type=int)
    p.add_argument('-d', "--device")
    p.add_argument('--sample', type=float, default=1.0)
    p.add_argument("--flank", type=int, default=0, help="flank length of the seq")
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--model_path', type=str, required=True)
    return p

if __name__ == "__main__":
    args = get_args().parse_args()
    set_rand_seed(args.seed)
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

    batch_size = args.batch_size
    dataset = ExpressionTestData(
        fasta=args.fasta,
        flank=args.flank,
    )

    test_inds = [i for i in range(len(dataset))]
    test_loader = DataLoader(dataset=Subset(dataset=dataset, indices=test_inds), batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    model_path = args.model_path

    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")), strict=False)
    model = model.to(device)
    logger.info("model {}\n{}".format(model, model_summary(model)))
    model.eval()
    test_pred = test_model(model, test_loader)
    #logger.info("test PCC/SCC: {:.4f}\t{:.4f}".format(np.nanmean(test_pcc), test_scc))

    torch.save((test_pred), f"{outdir}/test.pt")