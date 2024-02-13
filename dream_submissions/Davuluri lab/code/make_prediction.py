import os
import json
import csv
import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

import transformers
from transformers import (
    DNAKmerConfig,
    DNASubConfig,
    LongformerForSequenceClassification,
    LongformerModel,
    BertForSequenceClassification,
    DNAKmerTokenizerFast,
    DNASubTokenizerFast,)


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string)-kmer:
        sentence += original_string[i:i+kmer] + " "
        i += stride
    
    return sentence[:-1].strip("\"")


def run(args):
    # get dream predictions
    model_name_or_path = args.model_dir
    tokenizer_name = args.model_dir
    model_revision = "main"
    use_auth_token = False
    task_name = None

    text_dir = args.test_file
    with open(text_dir, "r") as f:
        data = f.readlines()
        texts = [get_kmer_sentence(d.strip("\n"), kmer=6) for d in data]
        num_labels = 1

    config = DNAKmerConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
        output_attentions=True,
    )
    tokenizer = DNAKmerTokenizerFast.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )
    model = BertForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )



    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encoding):
            self.encoding = encoding

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
            return item

        def __len__(self):
            return len(self.encoding['input_ids'])


    device = "cuda"
    batch_size = 128

    n_gpu = torch.cuda.device_count()

    model.to(device)

    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    all_predicts = torch.zeros([1, 1], dtype=torch.float16)

    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    emb_dataset = TextDataset(encoding)
    emb_sampler = SequentialSampler(emb_dataset)
    emb_dataloader = DataLoader(emb_dataset, batch_size=batch_size * n_gpu, sampler=emb_sampler)
    emb_iterator = tqdm(emb_dataloader, desc="Iteration")

    with torch.no_grad():
        model.eval()
        for inputs in emb_iterator:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            pred = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True).logits
            pred = pred.type_as(all_predicts)
            all_predicts = torch.cat([all_predicts, pred])


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, "pred.npy"), all_predicts[1:])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_dir",
        default="output",
        type=str,
        help="The path to the trained model",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=True,
        help="The path to the test file",
    )
    parser.add_argument(
        "--output_dir",
        default="results/",
        type=str,
        help="The path to the predicted results",
    )
    args = parser.parse_args()

    run(args)

