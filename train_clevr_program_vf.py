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
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)s %(funcName)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from models.module_s import TransformerModuleNet
from models.module_vf import TransformerModuleNetWithExtractor
from datasets.clevr_dataset_program_s import CLEVRDataset

from cfgs.path_cfgs import PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--save_name", default='modular_test', type=str, help="save name for training.",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=float, help="start epoch.",
    )
    parser.add_argument(
        "--num_epochs", default=20.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="training batch size",
    )
    parser.add_argument(
        "--gas", default=1, type=float, help="gradient accumulation steps",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--num_module_layers", type=int, default=1, help="Number of module layers.",
    )
    parser.add_argument(
        "--from_pretrained", default='', type=str, help="model path.",
    )
    parser.add_argument(
        "--arch", type=str, default='s', help="Network architecture (s, t)", choices=['s', 't']
    )
    parser.add_argument(
        "--vf", type=str, default='region', help="use othre visual features", choices=['region', 'vt']
    )
    parser.add_argument(
        "--tgt", type=str, default='clevr', help="target dataset", choices=['clevr', 'cgt']
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

    savePath = os.path.join(path_cfgs.save_path, args.save_name)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
        exit()
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

    corpus_path_train = path_cfgs.path_dict_corpus_train[args.tgt+'_raw']
    corpus_path_val = path_cfgs.path_dict_corpus_val[args.tgt+'_raw']
    annotation_path_train = path_cfgs.path_dict_annotation_train[args.tgt]
    annotation_path_val = path_cfgs.path_dict_annotation_val[args.tgt]
    print(f'target {args.tgt}')
    print(f'corpus_path_train : {corpus_path_train}')
    print(f'corpus_path_val : {corpus_path_val}')
    print(f'annotation_path_train : {annotation_path_train}')
    print(f'annotation_path_val : {annotation_path_val}')

    proposal_path_train = path_cfgs.path_dict_proposal_train['cgt']
    proposal_path_val = path_cfgs.path_dict_proposal_val['valA']

    if args.vf == 'vt':
        proposal_path_train = None
        proposal_path_val = None
    
    train_dataset = CLEVRDataset(
        corpus_path_train,
        proposal_path_train,
        annotation_path_train,
        path_cfgs.vocab_path,
        path_cfgs.func_vocab_path,
        path_cfgs.args_vocab_path,
        seq_len=36,
    )

    validation_dataset = CLEVRDataset(
        corpus_path_val,
        proposal_path_val,
        annotation_path_val,
        path_cfgs.vocab_path,
        path_cfgs.func_vocab_path,
        path_cfgs.args_vocab_path,
        seq_len=36,
    )

    train_batch_size = args.batch_size  # 1
    val_batch_size = args.batch_size 
    num_train_epochs = args.num_epochs
    start_epoch = args.start_epoch
    gradient_accumulation_steps = args.gas # 100
    
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=16, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=val_batch_size, num_workers=4)

    num_train_optimization_steps = (
        math.ceil(train_dataset.num_dataset / train_batch_size / gradient_accumulation_steps)
        * (num_train_epochs - start_epoch)
    )

    # warmup_steps = num_train_optimization_steps * 0.1
    warmup_steps = 0

    learning_rate = args.learning_rate
    adam_epsilon = 1e-8

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": learning_rate, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": learning_rate, "weight_decay": 0.0}
                ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'use: {n_gpu} GPUs')

    # Train
    print(f'Num examples = {len(train_dataset)}')
    print(f'Num Epochs = {num_train_epochs}')
    print(f'Learning rate = {learning_rate}')
    print(f'Total train batch size (w. parallel, distributed & accumulation) = {train_batch_size * gradient_accumulation_steps}')
    print(f'Gradient Accumulation steps = {gradient_accumulation_steps}')
    print(f'Total optimization steps = {num_train_optimization_steps}')

    num_steps = int(train_dataset.num_dataset / train_batch_size / gradient_accumulation_steps)
    model.zero_grad()
    
    global_step = 0
    step_tmp = 0
    startIterID = 0
    matches_tmp = 0
    loss_tmp = 0
    global_loss_tmp = 0
    global_matches_tmp = 0

    logger.info('start training')
    
    for epochId in range(int(start_epoch), int(num_train_epochs)):
        model.train()
        matches_tmp = 0
        loss_tmp = 0
        step_tmp = 0
        global_loss_tmp = 0
        global_matches_tmp = 0

        for step, batch in enumerate(train_data_loader):
            iterId = startIterID + step + (epochId * num_steps)
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            img, regions, img_info, spatials, image_mask, arguments, answer_id, question_id = (
                batch
            )

            outputs, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            matches_tmp += count_matches.item()
            loss_tmp += loss.item()
            step_tmp += 1

            # print("step", step, logits, answer_id, count_matches)

            loss.backward()

            # nb_tr_examples += question.size(0)
            # nb_tr_steps += 1

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == int(train_dataset.num_dataset / train_batch_size):
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                # optimizer.zero_grad()
                scheduler.step()        # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                global_loss_tmp += loss_tmp
                global_matches_tmp += matches_tmp

                # print(epochId, step, global_step, matches_tmp / (step_tmp * train_batch_size), loss_tmp, " | ", scheduler.get_last_lr()[0], flush=True)
                # print(f'Epoch:{epochId}, Step:{step}, g:{global_step}, r:{matches_tmp / (step_tmp * train_batch_size)}, loss:{loss_tmp} | lr:{scheduler.get_last_lr()[0]}', flush=True)

                matches_tmp = 0
                loss_tmp = 0
                step_tmp = 0

                if global_step % 20 == 0 and global_step != 0:
                    global_loss_tmp = global_loss_tmp / 20.0
                    global_matches_tmp = global_matches_tmp / (gradient_accumulation_steps * train_batch_size * 20.0)

                    print(f'# Epoch:{epochId}, Step:{step}, g:{global_step}, gR:{global_matches_tmp}, gL:{global_loss_tmp} | lr:{scheduler.get_last_lr()[0]}', flush=True)
                    
                    global_loss_tmp = 0
                    global_matches_tmp = 0

        
        # Do the evaluation 
        torch.set_grad_enabled(False)
        model.eval()

        eval_total_matches = 0
        eval_total_loss = 0
        step_tmp_val = 0

        for step, batch in enumerate(validation_data_loader):
            iterId = startIterID + step + (epochId * num_steps)
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

            img, regions, img_info, spatials, image_mask, arguments, answer_id, question_id = (
                batch
            )

            outputs, pred =  model(img, spatials, image_mask, arguments, region_props=regions, image_info=img_info)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred, answer_id)

            if n_gpu > 1:
                loss = loss.mean() 
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            logits = torch.max(pred, 1)[1].data  # argmax
            count_matches = ((logits - answer_id) == 0).sum().float()

            eval_total_matches += count_matches.item()   
            eval_total_loss += loss.item()
            step_tmp_val += img.size(0)

        eval_score = eval_total_matches / float(validation_dataset.num_dataset)
        eval_loss = eval_total_loss / float(validation_dataset.num_dataset)

        print('Evaluation')
        print(f'Epoch:{epochId}, Score:{eval_score} ({eval_total_matches / float(step_tmp_val)}), loss:{eval_loss}', flush=True)
        torch.set_grad_enabled(True)

        # Save a trained model
        print("*** Saving fine - tuned model ***")
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model it-self
        output_model_file = os.path.join(
            savePath, "pytorch_model_" + str(epochId) + ".bin"
        )

        torch.save(model_to_save.state_dict(), output_model_file)

    logger.info('finished training')

if __name__ == "__main__":
    main()
