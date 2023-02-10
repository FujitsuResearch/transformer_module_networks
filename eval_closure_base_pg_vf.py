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
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertConfig, BertTokenizer

from datasets.clevr_dataset_pg_raw import CLEVRDataset
from models.base_model_pg_vf import BaseTransformerWithExtractor
import torch.distributed as dist

import pdb

from cfgs.path_cfgs import PATH

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name",
        default='pretrained',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--distributed", action="store_true" , help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of hidden layers.",
    )
    args = parser.parse_args()
    print(args)

    print("import path cfgs")
    path_cfgs = PATH()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = BertConfig.from_json_file(path_cfgs.root_path + args.config_file)
    config.v_hidden_size = 768
    config.bi_hidden_size = 768
    config.num_hidden_layers = args.num_layers
    config.vocab_size = 49
    config.type_vocab_size = 4

    print("BERT model: [{}] is used".format(path_cfgs.bert_model))
    tokenizer = BertTokenizer.from_pretrained(
        path_cfgs.bert_model, do_lower_case=args.do_lower_case
    )

    base_path = path_cfgs.closure_path
    closure_vals = [
        'and_mat_spa', 'compare_mat', 'compare_mat_spa', 'embed_mat_spa', 'embed_spa_mat', 'or_mat', 'or_mat_spa'
    ]
    corpus_path_val = path_cfgs.path_dict_corpus_val['clevr_raw']

    datasets = [
        CLEVRDataset(
            corpus_path_val, 
            None,
            base_path + name + '_test.json',
            path_cfgs.vocab_path,
            path_cfgs.func_vocab_path,
            path_cfgs.args_vocab_path,
            seq_len=45,
        ) for name in closure_vals
        ]

    if len(datasets) > 0:
        print("Number of loaded data", len(datasets))
        num_labels = datasets[0].num_labels   # 32
    else:
        print("Failed to load data :", base_path)
        exit()

    config.num_labels = datasets[0].num_labels   # 32
    config.max_region = 150
    config.use_layer_norm_feat = False
    config.use_location_embed = False

    from models.visual_tokenizer import VisualTokenizer
    extractor = VisualTokenizer(config)
    
    model = BaseTransformerWithExtractor(config, extractor)
    model.load_state_dict(torch.load(args.from_pretrained))
    print("loaded pre-trained model", args.from_pretrained)
    
    model.cuda()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    eval_all = 0
    eval_all_matches = 0
    for di, ds in enumerate(datasets):
        print(f'{di} | Test data : {closure_vals[di]}')

        data_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

        num_dataset = ds.num_dataset
        numBatches = math.ceil(num_dataset / args.batch_size)

        start_t = timer()
        eval_total_loss = 0
        eval_total_matches = 0

        for step, batch in enumerate(data_loader):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            img, regions, img_info, spatials, image_mask, question, segment_ids, input_mask, co_attention_mask, answer_id, question_id = (
                batch
            )

            vil_prediction = \
                model(question, img, regions, img_info, spatials, segment_ids, input_mask, image_mask, co_attention_mask)        
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(vil_prediction, answer_id)

            logits = torch.max(vil_prediction, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            eval_total_matches += count_matches.item()

            if n_gpu > 1:
                loss = loss.mean()
            
            eval_total_loss += loss.item()

            end_t = timer()
            delta_t = " Time: %5.2fs" % (end_t - start_t)
            start_t = end_t

            if step > 0 and (step + 1) % 10000 == 0:
                progressString = "\r Evaluating split '%s' [%d/%d]\t" + delta_t
                sys.stdout.write(progressString % ('val', step + 1, numBatches))
                sys.stdout.flush()

        eval_total_loss = eval_total_loss / float(num_dataset)
        eval_score = eval_total_matches / float(num_dataset)

        eval_all = eval_all + num_dataset
        eval_all_matches = eval_all_matches + eval_total_matches

        printFormat = "Evaluation: [Loss: %.5g][Score: %.5g]"
        printInfo = [eval_total_loss, eval_score]

        print(printFormat % tuple(printInfo))
        print(f'{eval_total_matches} / {num_dataset}')
    
    print(f'Overall | {float(eval_all_matches) / eval_all} ({float(eval_all_matches)} / {eval_all})')


if __name__ == "__main__":

    main()