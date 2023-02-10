# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import json
import os
import numpy as np
import pickle
import sys
import csv
import base64
import time 
import torch
from torch.utils.data import Dataset
from torch import nn

from utils import load_obj_tsv

class GQADataset(Dataset):
    def __init__(self, features_path, annotation_path, seq_len=9,):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.seq_len = seq_len
        self.region_len = 36

        self.feature_dict = self.load_features(self.features_path)
        self.annotations = self.load_annotations(self.annotation_path)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.vocab_path = os.path.join(dir_path, 'answer_vocab.json')

        self.vocab_path = os.path.join(dir_path, 'answer_vocab.json')
        self.answer_vocab = json.load(open(self.vocab_path, 'r'))
        
        self.vocab_path_full = os.path.join(dir_path, 'full_vocab_gqa_balanced.json')
        self.vocab = json.load(open(self.vocab_path_full, 'r'))

        self.vocab_path_func = os.path.join(dir_path, 'func_vocab_gqa.json')
        self.func_vocab = json.load(open(self.vocab_path_func, 'r'))

        self.num_images = len(self.feature_dict)
        self.num_dataset = len(self.annotations)

        self.num_labels = len(self.answer_vocab) + 1 # + 1 = unknown
        self.vocab_size = len(self.vocab)

        print(f'found {self.num_images} images')
        print(f'found {self.num_dataset} entries')
        print(f'answer vocab size : {len(self.answer_vocab)}')
        print(f'function vocab size : {len(self.func_vocab)}')
        print(f'vocab size : {self.vocab_size}')

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
        operations = np.full((9, 4), 35)
        num_operations = 0
        for p in inputs:
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

            operations[num_operations] = [func_idx, arg_idx[0], arg_idx[1], arg_idx[2]]
            num_operations = num_operations + 1

        # print(operations)

        # Prepare answer
        # answer_id = self.answer_vocab[answer]
        answer_id = self.answer_vocab.get(answer, 2)

        # print(answer_id)
        
        features = torch.tensor(features).float()
        spatials = torch.tensor(spatials).float()
        image_mask = torch.tensor(image_mask).long()
        
        operations = torch.tensor(operations).long()
        answer_id = torch.tensor(answer_id).long()
        question_id = torch.tensor((int)(question_id)).long()

        # print("features", features.shape)
        # print("spatials", spatials.shape)
        # print("image_mask", image_mask.shape)
        # print("operations", operations.shape)
        # print("answer_id", answer_id.shape, answer_id)
        # print("question_id", question_id.shape, question_id)

        return features, spatials, image_mask, operations, answer_id, question_id

    def load_features(self, features_path):
        img_data = []
        img_data.extend(load_obj_tsv(features_path))

        id2Img = {}

        for d in img_data:
            id2Img[d['img_id']] = d

        return id2Img

    def load_annotations(self, caption_path):
        annos = []
        with open(caption_path, 'r') as f:
            annos = json.load(f)

        return annos

    def __len__(self):
        return len(self.annotations)