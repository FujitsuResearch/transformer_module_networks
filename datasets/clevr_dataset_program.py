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
    def __init__(self, corpus_path, caption_path, vocab_path, func_vocab_path, args_vocab_path, seq_len,):
        self.corpus_path = corpus_path
        self.caption_path = caption_path
        self.seq_len = seq_len
        self.region_len = 36
        self.num_labels = 32

        self.feature_dict = self.load_features(self.corpus_path)
        self.annotations = self.load_annotations(self.caption_path)
        self.vocabs = json.load(open(vocab_path, 'r'))["answer_token_to_idx"]
        self.func_vocab = json.load(open(func_vocab_path, 'r'))
        self.args_vocab = json.load(open(args_vocab_path, 'r'))

        self.num_images = len(self.feature_dict)
        self.num_dataset = len(self.annotations)

        print(f'found {self.num_images} images')
        print(f'found {self.num_dataset} entries')
        print(f'function vocabulary:', len(self.func_vocab))
        print(f'argument vocabulary:', len(self.args_vocab))

    def __len__(self):
        return self.num_dataset

    def __getitem__(self, index):
        features, spatials, image_mask, args, answer_id, question_id = self.get_item_set(index)

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

        args = torch.tensor(args).long()

        answer_id = torch.tensor(answer_id).long()
        question_id = torch.tensor(question_id).long()
        
        # print("features", features.shape)
        # print("spatials", spatials.shape)
        # print("image_mask", image_mask.shape)
        # print("answer_id", answer_id.shape)
        # print("question_id", question_id.shape)
        # print("answer_id", answer_id)
        # print("question_id", question_id)
        # print("args", args)
        # print("args", args.shape)

        # co_attention_mask = torch.zeros((self.region_len, self.seq_len))

        # return (features, spatials, image_mask, args, co_attention_mask, answer_id, question_id,)
        return (features, spatials, image_mask, args, answer_id, question_id,)

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
        
        features = mix_features_pad
        spatials = mix_location_pad

        # print(num_boxes)
        
        # question
        progs = entry["program"]
        answer = entry["answer"]
        # question = entry["question"]
        # print("question", question)

        # print(answer)

        args = np.full((25, 2), 25)
        num_progs = 0
        for p in progs:
            func = p['function']
            arg = p['value_inputs']
            arg_idx = 19
            if arg:
                arg_idx = self.args_vocab[arg[0]]

            func_idx = self.func_vocab[func]
            if func_idx > 0:
                args[num_progs] = [func_idx - 1, arg_idx]
                num_progs = num_progs + 1

            # print(func_idx - 1, arg_idx)

        answer_id = self.vocabs[answer]
        question_id = index

        return features, spatials, image_mask, args, answer_id, question_id