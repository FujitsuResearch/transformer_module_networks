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
    def __init__(self, features_path, annotation_path, tokenizer, seq_len,):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.tokenizer = tokenizer
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

        self.num_labels = len(self.answer_vocab) + 1    # + 1 = unknown

        print(f'found {self.num_images} images')
        print(f'found {self.num_dataset} entries')
        print(f'vocab size : {len(self.answer_vocab)}')

    def __len__(self):
        return self.num_dataset

    def __getitem__(self, index):
        features, spatials, image_mask, question, segment_ids, input_mask, answer_id, question_id = self.get_item_set(index)

        g_image_feat = np.sum(features, axis=0) / np.sum(image_mask, axis=0, keepdims=True)
        features = np.concatenate([np.expand_dims(g_image_feat, axis=0), features], axis=0)
        features = np.array(features, dtype=np.float32)

        g_image_loc = np.array([[0,0,1,1,1]], dtype=np.float32)
        spatials = np.concatenate([g_image_loc, spatials], axis=0)
        spatials = np.array(spatials, dtype=np.float32)

        g_image_mask = np.array([1])
        image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

        features = torch.tensor(features).float()
        spatials = torch.tensor(spatials).float()
        image_mask = torch.tensor(image_mask).long()

        question = torch.tensor(question).long()
        segment_ids = torch.tensor(segment_ids).long()
        input_mask = torch.tensor(input_mask).long()

        # answer_id = torch.tensor([answer_id]).long()
        # question_id = torch.tensor([question_id]).long()
        answer_id = torch.tensor(answer_id).long()
        question_id = torch.tensor(question_id).long()
        
        # print("features", features.shape)
        # print("spatials", spatials.shape)
        # print("image_mask", image_mask.shape)
        # print("question", question.shape)
        # print("segment_ids", segment_ids.shape)
        # print("input_mask", input_mask.shape)
        # print("answer_id", answer_id.shape)
        # print("question_id", question_id.shape)
        # print("answer_id", answer_id)
        # print("question_id", question_id)

        co_attention_mask = torch.zeros((self.region_len, self.seq_len))

        return (features, spatials, image_mask, question, segment_ids, input_mask, co_attention_mask, answer_id, question_id,)

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

    def get_item_set(self, index):
        entry = self.annotations[index]

        img_id = entry['img_id']
        question_id = (int)(entry['question_id'])
        question = entry['sent']
        label = list(entry['label'].keys())[0]

        # Get image info
        image_data = self.feature_dict[img_id]
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

        # tokenize
        tokenized_question, segment_ids, input_mask = self.tokenize(question, self.seq_len)
        
        question = tokenized_question
        
        answer_id = self.answer_vocab.get(label, 1845)

        # print(f'find {label} => {answer_id}')

        return features, spatials, image_mask, question, segment_ids, input_mask, answer_id, question_id

    def tokenize(self, question, max_length=36, padding_index=0):
        tokens = self.tokenizer.tokenize(question)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_ids = [
            self.tokenizer.vocab.get(w, self.tokenizer.vocab["[UNK]"])
            for w in tokens
        ]

        token_ids = token_ids[:max_length]
        segment_ids = [0] * len(token_ids)
        input_mask = [1] * len(token_ids)

        if len(token_ids) < max_length:
            # Note here we pad in front of the sentence
            padding = [padding_index] * (max_length - len(token_ids))
            token_ids = token_ids + padding
            input_mask += padding
            segment_ids += padding

        return token_ids, segment_ids, input_mask