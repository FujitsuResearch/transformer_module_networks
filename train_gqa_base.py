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

from datasets.gqa_dataset import GQADataset
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
        "--train_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
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
    args = parser.parse_args()
    print(args)

    print("import path cfgs")
    path_cfgs = PATH()

    savePath = os.path.join(path_cfgs.save_path, args.save_name)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(path_cfgs.root_path + args.config_file)
    config.v_hidden_size = 768
    config.bi_hidden_size = 768
    config.num_hidden_layers = args.num_layers
    print(config)

    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    bert_weight_file = path_cfgs.root_path + "config/" + args.bert_model + "_weight_name.json"
    bert_weight_name = json.load(open(bert_weight_file, "r"))
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("BERT model: [{}] is used".format(path_cfgs.bert_model))
    tokenizer = BertTokenizer.from_pretrained(
        path_cfgs.bert_model, do_lower_case=args.do_lower_case
    )

    num_train_optimization_steps = None

    features_path_train = path_cfgs.path_dict_corpus_train['gqa']
    annotation_path_train = path_cfgs.path_dict_annotation_train['gqa']
    features_path_val = path_cfgs.path_dict_corpus_val['gqa']
    annotation_path_val = path_cfgs.path_dict_annotation_val['gqa']
    
    train_dataset = GQADataset(
        features_path_train,
        annotation_path_train,
        tokenizer,
        seq_len=36,
    )
    
    validation_dataset = GQADataset(
        features_path_val,
        annotation_path_val,
        tokenizer,
        seq_len=36,
    )
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=8)
    validation_data_loader = DataLoader(validation_dataset, batch_size=args.train_batch_size, num_workers=2)

    num_train_optimization_steps = (
        math.ceil(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )
    if args.local_rank != -1:
        num_train_optimization_steps = (
            num_train_optimization_steps // torch.distributed.get_world_size()
        )

    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    config.num_labels = train_dataset.num_labels

    model = BaseTransformer(config)
    if args.from_pretrained:
        # use 'bert-base-uncased' for BERT
        model = BaseTransformer.from_pretrained(args.from_pretrained, from_tf=False, config=config)
        print("loaded pre-trained model")

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

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = []
    lr = args.learning_rate
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_prediction' in key:
                lr = 1e-4
            else:
                lr = args.learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    warmup_steps = num_train_optimization_steps * 0.1

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    loss_tmp = 0
    score_tmp = 0
    start_t = timer()
    num_steps = int(train_dataset.num_dataset / args.train_batch_size / args.gradient_accumulation_steps)

    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_data_loader):
            iterId = startIterID + step + (epochId * num_steps)
            target_batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            features, spatials, image_mask, question, segment_ids, input_mask, co_attention_mask, answer_id, question_id = (
                target_batch
            )

            vil_prediction = \
                model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(vil_prediction, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            logits = torch.max(vil_prediction, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()
            score = count_matches / float(args.train_batch_size)

            score_tmp += score.item()

            loss.backward()

            if dist.is_available() and args.distributed:
                rank = dist.get_rank()
            else:
                rank = 0

            loss_tmp += loss.item()

            nb_tr_examples += question.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if step % 20 == 0 and step != 0:
                loss_tmp = loss_tmp / 20.0
                score_tmp = score_tmp / 20.0

                end_t = timer()
                timeStamp = strftime("%a %d %b %y %X", gmtime())

                Ep = epochId + nb_tr_steps / float(num_steps)
                printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g][Score: %.5g][LR: %.8g]"

                printInfo = [
                    timeStamp,
                    Ep,
                    nb_tr_steps,
                    end_t - start_t,
                    loss_tmp,
                    score_tmp,
                    scheduler.get_last_lr()[0],
                ]               
                
                start_t = end_t
                print(printFormat % tuple(printInfo), flush=True)

                loss_tmp = 0
                score_tmp = 0

        # Do the evaluation 
        torch.set_grad_enabled(False)
        start_t = timer()
        numBatches = int(validation_dataset.num_dataset / args.train_batch_size / args.gradient_accumulation_steps)
        eval_total_loss = 0
        eval_total_matches = 0

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
            score = count_matches / float(args.train_batch_size)

            if n_gpu > 1:
                loss = loss.mean()
            
            eval_total_loss += loss.item()
            eval_total_matches += count_matches

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
        torch.set_grad_enabled(True)

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )

            torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":

    main()