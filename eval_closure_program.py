# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import argparse
import json
import logging
import os
import random
from io import open
import math
import sys

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertConfig

from models.module_s import TransformerModuleNet
from datasets.clevr_dataset_program_s import CLEVRDataset

from cfgs.path_cfgs import PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_pretrained", default='', type=str, help="model path.",
    )
    parser.add_argument(
        "--num_module_layers", type=int, default=1, help="Number of module layers.",
    )
    parser.add_argument(
        "--arch", type=str, default='s', help="Network architecture (s, t)", choices=['s', 't'],
    )
    parser.add_argument(
        "--dump", action="store_true" , help="whether dump the results."
    )
    args = parser.parse_args()
    print(args)

    print("import path cfgs")
    path_cfgs = PATH()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    config = BertConfig.from_pretrained(path_cfgs.root_path + 'config/bert_base_6layer_6conect.json')        
    config.num_module_layer = args.num_module_layers
    config.use_location_embed = True
    print(config)
    print("num_module_layer:", config.num_module_layer)

    if args.arch == 's':
        from models.module_s import TransformerModuleNet
        from datasets.clevr_dataset_program_s import CLEVRDataset
        config.num_region = 37
        print("select stack arch")
    elif args.arch == 't':
        from models.module_t import TransformerModuleNet
        from datasets.clevr_dataset_program_v2 import CLEVRDataset
        config.num_region = 74
        print("select tree arch")
    else:
        print("arch should be [s, t]")
        exit()        

    model_path = args.from_pretrained if args.from_pretrained else path_cfgs.from_pretrained

    # set up model
    model = TransformerModuleNet(config)

    model.load_state_dict(torch.load(model_path))
    print(f'loaded from {model_path}')

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    base_path = path_cfgs.closure_path
    closure_vals = [
        'and_mat_spa', 'compare_mat', 'compare_mat_spa', 'embed_mat_spa', 'embed_spa_mat', 'or_mat', 'or_mat_spa'
    ]
    corpus_path_val = path_cfgs.path_dict_corpus_val['clevr']

    datasets = [
        CLEVRDataset(
            corpus_path_val, 
            base_path + name + '_test.json',
            path_cfgs.vocab_path,
            path_cfgs.func_vocab_path,
            path_cfgs.args_vocab_path,
            seq_len=36,
        ) for name in closure_vals
        ]

    batch_size = 100
    gradient_accumulation_steps = 1

    for di, ds in enumerate(datasets):
        print(f'{di} | Test data : {closure_vals[di]}')

        data_loader = DataLoader(ds, batch_size=batch_size, num_workers=2)

        num_dataset = ds.num_dataset
        numBatches = math.ceil(num_dataset / batch_size / gradient_accumulation_steps)

        # Do the evaluation 
        torch.set_grad_enabled(False)
        start_t = timer()
        eval_total_loss = 0
        eval_total_matches = 0

        results_dict = {}

        model.eval()
        for step, batch in enumerate(data_loader):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            features, spatials, image_mask, arguments, answer_id, question_id = (
                batch
            )

            outputs, pred =  model(features, spatials, image_mask, arguments)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            # collect results for each questions
            matches = (logits - answer_id) == 0
            for idx, qid in enumerate(question_id):
                results_dict[str(qid.item())] = [matches[idx].item(), logits[idx].item(), answer_id[idx].item()]

            eval_total_matches += count_matches.item()

            if n_gpu > 1:
                loss = loss.mean()
            
            eval_total_loss += loss.item()

            end_t = timer()
            delta_t = " Time: %5.2fs" % (end_t - start_t)
            start_t = end_t
            progressString = "\r Evaluating split '%s' [%d/%d]\t" + delta_t
            sys.stdout.write(progressString % ('val', step + 1, numBatches))
            sys.stdout.flush()

        eval_total_loss = eval_total_loss / float(num_dataset)
        eval_score = eval_total_matches / float(num_dataset)

        printFormat = "Evaluation: [Loss: %.5g][Score: %.5g]"
        printInfo = [eval_total_loss, eval_score]

        print(printFormat % tuple(printInfo))
        print(f'{eval_total_matches} / {num_dataset}')

        if args.dump:
            output_results_json = './results.json'
            with open(output_results_json, 'w') as f:
                json.dump(results_dict, f)

            print("dump the results", output_results_json)

if __name__ == "__main__":

    main()