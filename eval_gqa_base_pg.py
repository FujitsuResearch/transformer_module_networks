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

from transformers import BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from datasets.gqa_dataset_pg import GQADataset
from models.base_model_v2 import BaseTransformer
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
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Total batch size.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
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
        "--distributed", action="store_true" , help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of hidden layers.",
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
    
    config = BertConfig.from_json_file(path_cfgs.root_path + args.config_file)
    config.v_hidden_size = 768
    config.bi_hidden_size = 768
    config.num_hidden_layers = args.num_layers
    config.vocab_size = 2409
    config.type_vocab_size = 5
    print(config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
            features_path_val, path, seq_len=36,
        )
        validation_datasets.append(dataset)

    if len(validation_datasets) == 0:
        print("No datasets found")
        exit()

    config.num_labels = validation_datasets[0].num_labels

    model = BaseTransformer(config)
    if args.from_pretrained:
        model = BaseTransformer.from_pretrained(args.from_pretrained, from_tf=False, config=config)
        print(f'loaded trained model from : {args.from_pretrained}')

    model.cuda()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print("Use {} GPU".format(n_gpu))

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
    
        print(f'--- {tag} | Num examples: {num_samples} ---', flush=True)
        for step, batch in enumerate(validation_data_loader):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            features, spatials, image_mask, question, segment_ids, input_mask, co_attention_mask, answer_id, question_id = (
                batch
            )

            vil_prediction = \
                model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(vil_prediction, answer_id)

            logits = torch.max(vil_prediction, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()
            score = count_matches / float(args.batch_size)

            eval_total_matches += count_matches

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            
            eval_total_loss += loss.item()

            bs = question.size(0)
            correct = ((logits - answer_id) == 0)
            for b in range(bs):
                result_dict[question_id[b].item()] = [correct[b].item()]

        eval_total_loss = float(eval_total_loss) / float(num_samples)
        eval_score = float(eval_total_matches) / float(num_samples)

        eval_all = eval_all + num_samples
        eval_all_matches = eval_all_matches + eval_total_matches

        printFormat = "Evaluation: [Loss: %.5g][Score: %.5g]"
        printInfo = [eval_total_loss, eval_score]

        print(printFormat % tuple(printInfo))
        
        if args.dump:
            model_tag = os.path.split(os.path.dirname(args.from_pretrained))[-1]
            base = os.path.splitext(os.path.basename(args.from_pretrained))[0]
            output_results_json = f'./results_dict_gt_{model_tag}_{base}_{tag}.json'
            with open(output_results_json, 'w') as f:
                json.dump(result_dict, f)
            print("dump the results", output_results_json)
    
    print(f'Overall | {float(eval_all_matches) / eval_all} ({float(eval_all_matches)} / {eval_all})')


if __name__ == "__main__":

    main()