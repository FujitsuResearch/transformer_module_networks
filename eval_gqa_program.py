# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import argparse
import glob
import json
import logging
import os
import shutil
import random
import math

import numpy as np
import torch
from torch import nn
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer)
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import time
logger = logging.getLogger(__name__)

from models.module_gqa_t import TransformerModuleNet
from datasets.gqa_dataset_program_t import GQADataset

from cfgs.path_cfgs import PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=128, type=float, help="training batch size",
    )
    parser.add_argument(
        "--num_module_layers", type=int, default=2, help="Number of module layers.",
    )
    parser.add_argument(
        "--from_pretrained", default='', type=str, help="model path.",
    )
    parser.add_argument(
        "--arch", type=str, default='s', help="Network architecture (s, t)", choices=['s', 't'],
    )
    parser.add_argument(
        "--test", type=str, default="ind", help="[ind or ood]", choices=['ind', 'ood'],
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
    print(config)

    print("num_module_layer:", config.num_module_layer)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if args.arch == 's':
        from models.module_gqa_s import TransformerModuleNet
        from datasets.gqa_dataset_program import GQADataset
        print("select stack arch")
    elif args.arch == 't':
        from models.module_gqa_t import TransformerModuleNet
        from datasets.gqa_dataset_program_t import GQADataset
        print("select tree arch")
    else:
        print("arch should be [s, t]")
        exit()

    features_path_val = path_cfgs.path_dict_corpus_val['gqa']
    courpus_path_val = []
    if args.test == "ind":
        test_tags = ["testdev"]
        courpus_path_val.append(path_cfgs.path_dict_annotation_val['gqa_prog'])
    elif args.test == "ood":
        test_tags = ["verify_attr", "exist_and", "query_relate", "choose_attr"]
        dir_path = os.path.dirname(os.path.abspath(__file__))
        for tag in test_tags:
            path = os.path.join(dir_path, "datasets", f'test_{tag}_inputs.json')
            courpus_path_val.append(path)

    validation_datasets = []
    for path in courpus_path_val:
        dataset = GQADataset(
            features_path_val, path,
        )
        validation_datasets.append(dataset)

    if len(validation_datasets) == 0:
        print("No datasets found")
        exit()

    num_labels = validation_datasets[0].num_labels

    # set up model
    model = TransformerModuleNet(config, num_modules=35, max_prog_seq=9, num_progs=35, num_args=2374, num_labels=num_labels)

    if args.from_pretrained:
        model.load_state_dict(torch.load(args.from_pretrained))
        print(f'loaded trained model from : {args.from_pretrained}')

    model.to(device)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'use: {n_gpu} GPUs')

    # Do the evaluation 
    torch.set_grad_enabled(False)
    model.eval()
    
    eval_all = 0
    eval_all_matches = 0
    logger.info("***** Running evaluation *****")
    for dataset, tag in zip(validation_datasets, test_tags):
        validation_data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        num_samples = dataset.num_dataset

        result_dict = {}
        eval_total_loss = 0
        eval_total_matches = 0
        step_tmp_val = 0
    
        print(f'--- {tag} | Num examples: {num_samples} ---', flush=True)
        for step, batch in enumerate(validation_data_loader):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            features, spatials, image_mask, operations, answer_id, question_id = (
                batch
            )

            outputs, pred =  model(features, spatials, image_mask, operations)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean()

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            bs = operations.size(0)
            correct = ((logits - answer_id) == 0)
            prog_len = ((operations[:, :, 0] - 35) < 0).sum(dim=1)
            for b in range(bs):
                result_dict[question_id[b].item()] = [correct[b].item(), prog_len[b].item()]

            eval_total_matches += count_matches.item()   
            eval_total_loss += loss.item()
            step_tmp_val += features.size(0)

        eval_score = eval_total_matches / float(num_samples)
        eval_loss = eval_total_loss / float(num_samples)

        eval_all = eval_all + num_samples
        eval_all_matches = eval_all_matches + eval_total_matches

        print('Evaluation')
        print(f'Score:{eval_score} ({eval_total_matches / float(step_tmp_val)}), loss:{eval_loss}', flush=True)
        
        if args.dump:
            model_tag = os.path.split(os.path.dirname(args.from_pretrained))[-1]
            base = os.path.splitext(os.path.basename(args.from_pretrained))[0]
            output_results_json = f'./results_dict_{model_tag}_{base}_{tag}.json'
            with open(output_results_json, 'w') as f:
                json.dump(result_dict, f)
            print("dump the results", output_results_json)
    
    print(f'Overall | {float(eval_all_matches) / eval_all} ({float(eval_all_matches)} / {eval_all})')


if __name__ == "__main__":
    main()
