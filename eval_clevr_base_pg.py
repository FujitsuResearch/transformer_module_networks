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

from transformers import BertConfig

from datasets.clevr_dataset_pg import CLEVRDataset
from models.base_model_pg import BaseTransformer
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
    parser.add_argument(
        "--from_pretrained",
        default="",
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
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
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
        default='test',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--distributed", action="store_true" , help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of hidden layers.",
    )
    parser.add_argument(
        "--vf", type=str, default='region', help="use othre visual features",
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
    print(config)

    features_path_val = path_cfgs.path_dict_corpus_val[args.test + '_obj']
    annotation_path_val = path_cfgs.path_dict_annotation_val[args.test]     
    print(f'test : {args.test}')

    validation_dataset = CLEVRDataset(
        features_path_val,
        annotation_path_val,
        path_cfgs.vocab_path,
        path_cfgs.func_vocab_path,
        path_cfgs.args_vocab_path,
        seq_len=36,
    )
    
    validation_data_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=2)

    num_labels = validation_dataset.num_labels   # 32
    model_path = args.from_pretrained if args.from_pretrained else path_cfgs.from_pretrained
    print("load model from :", model_path)

    model = BaseTransformer(config, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))

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

    numBatches = int(validation_dataset.num_dataset / args.batch_size)

    logger.info("***** Running validation *****")
    logger.info("  Num examples = %d", validation_dataset.num_dataset)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num Batches = %d", numBatches)

    # Do the evaluation 

    torch.set_grad_enabled(False)
    start_t = timer()
    eval_total_loss = 0
    eval_total_matches = 0

    results_dict = {}

    model.eval()
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

    eval_total_loss = eval_total_loss / float(validation_dataset.num_dataset)
    eval_score = eval_total_matches / float(validation_dataset.num_dataset)

    printFormat = "Evaluation: [Loss: %.5g][Score: %.5g]"
    printInfo = [eval_total_loss, eval_score]

    print(printFormat % tuple(printInfo))
    print(f'{eval_total_matches} / {validation_dataset.num_dataset}')

    if args.dump:
        output_results_json = './results_clevr_base_pg_6l.json'
        with open(output_results_json, 'w') as f:
            json.dump(results_dict, f)
        
        print("dump the results", output_results_json)

if __name__ == "__main__":

    main()