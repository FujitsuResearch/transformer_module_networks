# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import random
import numpy as np
import glob
import copy
import json
import os

import sys
import csv
import base64
import time 

from torch.utils.data import Dataset
import torch

from utils import load_obj_tsv

class GQADataset(Dataset):
    def __init__(self, features_path, annotation_path, seq_len,):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.seq_len = seq_len
        self.region_len = 36
        self.num_labels = 32

        self.feature_dict = self.load_features(self.features_path)
        self.annotations = self.load_annotations(self.annotation_path)

        self.num_images = len(self.feature_dict)
        self.num_dataset = len(self.annotations)

        dir_path = os.path.dirname(os.path.abspath(__file__))

        self.vocab_path = os.path.join(dir_path, 'answer_vocab.json')
        self.answer_vocab = json.load(open(self.vocab_path, 'r'))
        
        self.vocab_path_full = os.path.join(dir_path, 'full_vocab_gqa_balanced.json')
        self.vocab = json.load(open(self.vocab_path_full, 'r'))

        self.vocab_path_func = os.path.join(dir_path, 'func_vocab_gqa.json')
        self.func_vocab = json.load(open(self.vocab_path_func, 'r'))

        self.num_labels = len(self.answer_vocab) + 1 # + 1 = unknown
        self.vocab_size = len(self.vocab)
        self.vocab_size_full = self.vocab_size + len(self.func_vocab)
        self.vocab_size_func = len(self.func_vocab)

        print(f'found {self.num_images} images')
        print(f'found {self.num_dataset} entries')
        print(f'answer vocab size : {len(self.answer_vocab)}')
        print(f'function vocab size : {self.vocab_size_func}')
        print(f'vocab size : {self.vocab_size}')
        print(f'full vocab size : {self.vocab_size_full}')

    def __len__(self):
        return self.num_dataset

    def __getitem__(self, index):
        entry = self.annotations[index]

        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        connection = entry[4]
        question_id = entry[-2]
        answer = entry[-1]

        # make visual inputs
        image_data = self.feature_dict[image_id]

        # for test
        # image_data = {}
        # image_data['num_boxes'] = 36
        # image_data['boxes'] = [[0] * 4] * 36
        # image_data['features'] = [[0] * 2048] * 36
        # image_data['img_h'] = 640
        # image_data['img_w'] = 480

        num_boxes = image_data['num_boxes']
        image_location = image_data['boxes'].copy()
        image_feature = image_data['features'].copy()
        image_h = image_data['img_h']
        image_w = image_data['img_w']

        assert len(image_location) == len(image_feature) == num_boxes

        mix_num_boxes = min(int(num_boxes), self.region_len)
        mix_location_pad = np.zeros((self.region_len, 5))
        mix_features_pad = np.zeros((self.region_len, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.region_len:
            image_mask.append(0)
        
        mix_features_pad[:mix_num_boxes] = image_feature[:mix_num_boxes]
        mix_location_pad[:mix_num_boxes,:4] = image_location[:mix_num_boxes]

        mix_location_pad[:,4] = (mix_location_pad[:,3] - mix_location_pad[:,1]) * (mix_location_pad[:,2] - mix_location_pad[:,0]) / (float(image_w) * float(image_h))
        mix_location_pad[:,0] = mix_location_pad[:,0] / float(image_w)
        mix_location_pad[:,1] = mix_location_pad[:,1] / float(image_h)
        mix_location_pad[:,2] = mix_location_pad[:,2] / float(image_w)
        mix_location_pad[:,3] = mix_location_pad[:,3] / float(image_h)
        
        features = mix_features_pad
        spatials = mix_location_pad

        g_image_feat = np.sum(features, axis=0) / np.sum(image_mask, axis=0, keepdims=True)
        features = np.concatenate([np.expand_dims(g_image_feat, axis=0), features], axis=0)
        features = np.array(features, dtype=np.float32)

        g_image_loc = np.array([[0,0,1,1,1]], dtype=np.float32)
        spatials = np.concatenate([g_image_loc, spatials], axis=0)
        spatials = np.array(spatials, dtype=np.float32)

        g_image_mask = np.array([1])
        image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

        # make program
        operations = self.make_programs(inputs, connection)

        # print(operations)
        
        input_ids, segment_ids, input_mask = self.tokenize(operations, self.seq_len)

        # Prepare answer
        answer_id = self.answer_vocab.get(answer, 2)

        # print(answer_id)
        
        features = torch.tensor(features).float()
        spatials = torch.tensor(spatials).float()
        image_mask = torch.tensor(image_mask).long()

        input_ids = torch.tensor(input_ids).long()
        segment_ids = torch.tensor(segment_ids).long()
        input_mask = torch.tensor(input_mask).long()

        answer_id = torch.tensor(answer_id).long()
        question_id = torch.tensor((int)(question_id)).long()

        co_attention_mask = torch.zeros((self.region_len, self.seq_len))

        return (features, spatials, image_mask, input_ids, segment_ids, input_mask, co_attention_mask, answer_id, question_id,)

    def make_programs(self, inputs, connection):
        # make program
        operations = np.full((9, 6), 35)
        num_operations = 0
        step = 0
        count_ops = 0
        for conn in connection:
            num_step = len(conn)
            num_inputs = 1
            if num_step == 0:
                num_step = 1
                conn = [[0, 0]]

            if num_step > 1 and conn[0][0] == conn[1][0]:
                num_step = 1
                num_inputs = 2

            step = num_operations
            # print(num_step, conn)
            for si, s in enumerate(range(step, step + num_step)):
                pi = conn[si][0]
                pi = pi
                p = inputs[pi]
                # print(si, s, pi, p)
                func = p[0]
                args = ["[PAD]"] * 3
                num_args = 0
                for i in range(1, len(p)):
                    if p[i] is not None:
                        args[num_args] = p[i]
                        num_args = num_args + 1

                # arg_idx = [self.vocab[o] for o in args]
                arg_idx = [self.vocab.get(o, 2) for o in args]
                func_idx = self.func_vocab[func]
                input_idx_0 = conn[si][1] + 1 if conn[si][0] != conn[si][1] else 0
                input_idx_1 = conn[si+1][1] + 1 if num_inputs == 2 else -1

                for di in [input_idx_0, input_idx_1]:
                    if di > 1:
                        di = di - 1
                        dfunc = operations[di][0]
                        if dfunc == 35:
                            # print("find missing dependency @", di)
                            dp = inputs[di]
                            dfunc = dp[0]
                            dfunc_idx = self.func_vocab[dfunc]
                            dargs = ["[PAD]"] * 3
                            dnum_args = 0
                            for i in range(1, len(dp)):
                                if dp[i] is not None:
                                    dargs[dnum_args] = dp[i]
                                    dnum_args = dnum_args + 1
                            
                            darg_idx = [self.vocab.get(o, 2) for o in dargs]
                            operations[di] = [dfunc_idx, 0, -1, darg_idx[0], darg_idx[1], darg_idx[2]]
                            count_ops = count_ops + 1

                operations[pi] = [func_idx, input_idx_0, input_idx_1, arg_idx[0], arg_idx[1], arg_idx[2]]
                num_operations = num_operations + 1
                count_ops = count_ops + 1
        
        return operations
    
    def tokenize(self, progs, max_length=36, padding_index=0):
        prog_tokens = []
        segments = []

        prog_len, arg_num = progs.shape
        # print(prog_len, arg_num)

        inputs_idx = [0] * (prog_len + 1)
        for step in range(prog_len):
            func_idx = progs[step, 0]
            if func_idx < self.vocab_size_func:
                func_idx = func_idx + self.vocab_size
                idx_0 = progs[step, 1]
                idx_1 = progs[step, 2]

                if idx_1 < 0:
                    inputs_idx[step + 1] = inputs_idx[idx_0]
                    inputs_idx[idx_0] = inputs_idx[idx_0] + 1
                else:
                    inputs_idx[step + 1] = 2

                prog_tokens.append(func_idx)
                segments.append(inputs_idx[step + 1] + 1)

                for ai in range(3):
                    arg_idx = progs[step, 3 + ai]
                    if arg_idx > 0:
                        prog_tokens.append(arg_idx)
                        segments.append(inputs_idx[step + 1] + 1)
            
            # print(f'step: {num_progs} | {func_idx}: {self.func[func_idx]}, {arg_idx}: {self.arg[arg_idx]} | {inputs}: {inputs_idx}')
    
        # print(prog_tokens)
        # print(segments)

        token_ids = [1] + prog_tokens + [1]
        segment_ids = [0] + segments + [0]

        token_ids = token_ids[:max_length]
        segment_ids = segment_ids[:max_length]
        input_mask = [1] * len(token_ids)

        if len(token_ids) < max_length:
            # Note here we pad in front of the sentence
            padding = [padding_index] * (max_length - len(token_ids))
            token_ids = token_ids + padding
            input_mask += [0] * len(padding)
            segment_ids += [0] * len(padding)

        return token_ids, segment_ids, input_mask

    def load_features(self, features_path):
        img_data = []
        img_data.extend(load_obj_tsv(features_path))

        id2Img = {}

        for d in img_data:
            id2Img[d['img_id']] = d

        return id2Img

    def load_annotations(self, caption_path):
        annos = json.load(open(caption_path, 'r'))

        return annos