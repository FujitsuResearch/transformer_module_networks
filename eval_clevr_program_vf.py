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

from models.module_s import TransformerModuleNet
from models.module_vf import TransformerModuleNetWithExtractor
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
        "--vf", type=str, default='region', help="use othre visual features", choices=['region', 'vt'],
    )
    parser.add_argument(
        "--dump", action="store_true" , help="whether dump the results."
    )
    parser.add_argument(
        "--test", type=str, default='valB', help="target dataset", choices=['valA', 'valB', 'val']
    )
    args = parser.parse_args()
    print(args)
    
    print("import path cfgs")
    path_cfgs = PATH()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    config = BertConfig.from_pretrained(path_cfgs.root_path + 'config/bert_base_6layer_6conect.json')
    config.num_module_layer = args.num_module_layers
    config.arch = args.arch
    config.vf = args.vf
    config.use_location_embed = True
    print(config)
    print("num_module_layer:", config.num_module_layer)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

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

    # set up model
    extractor = None
    if not args.vf:
        print("use pre-trained object detector")
    else:
        if args.vf == 'region':
            from models.extractor import FeatureExtractor
            extractor = FeatureExtractor(config)
            print("select regional features without Visual Genome")
        elif args.vf == 'vt':
            from models.visual_tokenizer import VisualTokenizer
            extractor = VisualTokenizer(config)
            print("select grid features as tokens")

            if args.arch == 's':
                config.num_region = 151
            elif args.arch == 't':
                config.num_region = 302

            config.use_location_embed = False         

        if args.arch == 't':
            from datasets.clevr_dataset_program_v2_raw import CLEVRDataset
        else:
            from datasets.clevr_dataset_program_raw import CLEVRDataset

    transformer = TransformerModuleNet(config)
    model = TransformerModuleNetWithExtractor(config, transformer, extractor)

    if args.from_pretrained:
        model.load_state_dict(torch.load(args.from_pretrained))
        print(f'loaded from {args.from_pretrained}')

    model.to(device)

    features_path_val = path_cfgs.path_dict_corpus_val[args.test]
    annotation_path_val = path_cfgs.path_dict_annotation_val[args.test]
    proposal_path_val = path_cfgs.path_dict_proposal_val[args.test]
    print(f'test : {args.test}')
    
    validation_dataset = CLEVRDataset(
        features_path_val,
        proposal_path_val,
        annotation_path_val,
        path_cfgs.vocab_path,
        path_cfgs.func_vocab_path,
        path_cfgs.args_vocab_path,
        seq_len=36,
    )

    val_batch_size = 128
    validation_data_loader = DataLoader(validation_dataset, batch_size=val_batch_size, num_workers=4)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'use: {n_gpu} GPUs')

    print(f'Num examples = {len(validation_dataset)}')

    # Do the evaluation 
    torch.set_grad_enabled(False)
    model.eval()

    eval_total_matches = 0
    eval_total_loss = 0
    step_tmp_val = 0

    results_dict = {}

    num_samples = validation_dataset.num_dataset

    for step, batch in enumerate(validation_data_loader):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

        img, regions, img_info, spatials, image_mask, arguments, answer_id, question_id = (
            batch
        )

        outputs, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, answer_id)

        if n_gpu > 1:
            loss = loss.mean()

        logits = torch.max(pred, 1)[1].data  # argmax
        count_matches = ((logits - answer_id) == 0).sum().float()

        # collect results for each questions
        matches = (logits - answer_id) == 0
        for idx, qid in enumerate(question_id):
            results_dict[str(qid.item())] = [matches[idx].item(), logits[idx].item(), answer_id[idx].item()]

        eval_total_matches += count_matches.item()
        eval_total_loss += loss.item()
        step_tmp_val += img.size(0)

        # print(f'Step:{step} {count_matches} / {num_samples}', flush=True)

    eval_score = eval_total_matches / float(num_samples)
    eval_loss = eval_total_loss / float(num_samples)

    print('Evaluation')
    print(f'Score:{eval_score} ({eval_total_matches / float(step_tmp_val)}), loss:{eval_loss}', flush=True)

    if args.dump:
        tag = os.path.split(os.path.dirname(args.from_pretrained))[-1]
        base = os.path.splitext(os.path.basename(args.from_pretrained))[0]
        output_results_json = f'./results_{tag}_{base}.json'
        with open(output_results_json, 'w') as f:
            json.dump(results_dict, f)
        
        print("dump the results", output_results_json)


if __name__ == "__main__":
    main()
