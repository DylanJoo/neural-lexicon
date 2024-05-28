# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import argparse
import torch
import json
from tqdm import tqdm

from transformers import AutoTokenizer

# def save(tensor, split_path):
#     if not os.path.exists(os.path.dirname(split_path)):
#         os.makedirs(os.path.dirname(split_path))
#     with open(split_path, 'wb') as fout:
#         torch.save(tensor, fout)

# revise the process with 'document-wise', 
# each list refers to a single document. 
# the length of token lists eqaul to number of documents in corpus
def apply_tokenizer_then_save(path, output_path, tokenizer, field='text'):

    # output file
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    fout = open(output_path, 'w')
    lines = []
    with open(path, "r", encoding="utf-8") as fin:
        docidx = 0
        for k, line in tqdm(enumerate(fin)):
            line = json.loads(line)[field]
            lines.append(line)

            # tokenize and save with batch
            if len(lines) >= 100:
                tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
                for token in tokens:
                    fout.write(json.dumps({'docidx': docidx, 'input_ids': token})+'\n')
                    docidx += 1

                lines = []

        # tokenize and save the last batch 
        if len(lines) > 0:
            tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
            for token in tokens:
                fout.write(json.dumps({'docidx': docidx, 'input_ids': token})+'\n')
                docidx += 1

    print(f'{k - docidx} documents have been discarded')
    fout.close()

def tokenize_file(args):
    filename = os.path.basename(args.datapath)
    savepath = os.path.join(args.outdir, filename.replace('.jsonl', "_tokenized.jsonl"))
    if os.path.exists(savepath):
        if args.overwrite:
            print(f"File {savepath} already exists, overwriting")
        else:
            print(f"File {savepath} already exists, exiting")
            return
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=False)
    print(f"Encoding {args.datapath}... and save tokenized ids in jsonl")

    tokens = apply_tokenizer_then_save(
            args.datapath, 
            savepath,
            tokenizer, 
            field=args.field
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--field", type=str, default='text')
    args = parser.parse_args()

    tokenize_file(args)
