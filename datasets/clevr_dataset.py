# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np
import glob
import copy
import json
import os


class CLEVRDataset(Dataset):
    def __init__(self, features_path, annotation_path, vocab_path, tokenizer, seq_len,):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.region_len = 36
        self.num_labels = 32

        self.feature_dict = self.load_features(self.features_path)
        self.annotations = self.load_annotations(self.annotation_path)
        self.vocabs = json.load(open(vocab_path, 'r'))["answer_token_to_idx"]

        self.num_images = len(self.feature_dict)
        self.num_dataset = len(self.annotations)

        print(f'found {self.num_images} images')
        print(f'found {self.num_dataset} entries')

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

    def load_features(self, corpus_path):
        # print(corpus_path)
        feature_path_list = glob.glob(corpus_path + '*.npz')

        feature_dict = {}
        for path in feature_path_list:
            image_id = os.path.splitext(os.path.basename(path))[0]
            feature_dict[image_id] = path

        return feature_dict

    def load_annotations(self, caption_path):
        annos = json.load(open(caption_path, 'r'))
        ques = annos["questions"]

        return ques

    def get_item_set(self, index):
        entry = self.annotations[index]

        # visual feature
        image_filename = entry["image_filename"]
        image_id = os.path.splitext(image_filename)[0]
        image_path = self.feature_dict[image_id]

        image_data = np.load(image_path)
        image_feature = image_data['x'].transpose((1, 0))
        image_location = image_data['bbox']
        image_target_wp = image_data['cls_prob']
        image_h = image_data['image_h']
        image_w = image_data['image_w']
        num_boxes = image_data['num_bbox']

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
        
        # features = torch.tensor(mix_features_pad).float()
        # spatials = torch.tensor(mix_location_pad).float()
        # image_mask = torch.tensor(image_mask).long()
        features = mix_features_pad
        spatials = mix_location_pad

        # print(num_boxes)
        
        # question
        question = entry["question"]
        answer = entry["answer"]

        # print(question)
        # print(answer)

        tokenized_question, segment_ids, input_mask = self.tokenize(question, self.seq_len)

        # print(tokenized_question)
        # print(segment_ids)
        # print(input_mask)

        # question = torch.tensor(tokenized_question).long()
        # segment_ids = torch.tensor(segment_ids).long()
        # input_mask = torch.tensor(input_mask).long()
        
        question = tokenized_question

        # question = entry["q_token"]
        # input_mask = entry["q_input_mask"]
        # segment_ids = entry["q_segment_ids"]

        # co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        # target = torch.zeros(self.num_labels)

        # if "test" not in self.split:
        #     answer = entry["answer"]
        #     labels = answer["labels"]
        #     scores = answer["scores"]
        #     if labels is not None:
        #         target.scatter_(0, labels, scores)

        # answer_id = self.vocabs.values().index(answer)
        # answer_id = self.vocabs.values()
        answer_id = self.vocabs[answer]
        # answer_id = torch.tensor(answer_id).long()

        # question_id = torch.tensor(index).long()
        question_id = index

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
