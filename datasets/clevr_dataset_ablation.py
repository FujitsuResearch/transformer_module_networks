# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import random
import numpy as np
import glob
import pickle
import copy
import json
import os
import tqdm
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class CLEVRDataset(Dataset):
    def __init__(self, features_path, proposal_path, annotation_path, vocab_path, func_vocab_path, args_vocab_path, seq_len,):
        self.features_path = features_path
        self.annotation_path = annotation_path
        self.seq_len = seq_len
        self.region_len = 36
        self.num_labels = 32

        self.proposal_path = proposal_path

        self.image_dict = self.load_image_paths(self.features_path)
        self.segms, self.boxes, self.nImgs, self.nCats = self.load_proposals(self.proposal_path)

        self.annotations = self.load_annotations(self.annotation_path)
        self.vocabs = json.load(open(vocab_path, 'r'))["answer_token_to_idx"]
        self.func_vocab = json.load(open(func_vocab_path, 'r'))
        self.args_vocab = json.load(open(args_vocab_path, 'r'))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_images = len(self.image_dict )
        self.num_dataset = len(self.annotations)

        print(f'found {self.num_images} images')
        print(f'found {self.nImgs} proposals')
        print(f'found {self.num_dataset} entries')


    def __len__(self):
        return self.num_dataset

    def __getitem__(self, index):
        img, regions, img_info, spatials, image_mask, input_ids, segment_ids, input_mask, answer_id, question_id = self.get_item_set(index)

        g_image_loc = np.array([[0,0,1,1,1]], dtype=np.float32)
        spatials = np.concatenate([g_image_loc, spatials], axis=0)
        spatials = np.array(spatials, dtype=np.float32)

        g_image_mask = np.array([1])
        image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

        spatials = torch.tensor(spatials).float()
        image_mask = torch.tensor(image_mask).long()

        regions = torch.tensor(regions).float()
        img_info = torch.tensor(img_info).long()

        input_ids = torch.tensor(input_ids).long()
        segment_ids = torch.tensor(segment_ids).long()
        input_mask = torch.tensor(input_mask).long()

        answer_id = torch.tensor(answer_id).long()
        question_id = torch.tensor(question_id).long()

        co_attention_mask = torch.zeros((self.region_len, self.seq_len))

        return (img, regions, img_info, spatials, image_mask, input_ids, segment_ids, input_mask, answer_id, question_id,)
        
    def load_image_paths(self, image_path):
        image_path_list = glob.glob(image_path + '*.png')

        image_dict = {}
        for path in image_path_list:
            image_id = os.path.splitext(os.path.basename(path))[0]
            image_dict[image_id] = path

        return image_dict

    def load_annotations(self, caption_path):
        annos = json.load(open(caption_path, 'r'))
        ques = annos["questions"]

        return ques

    def load_proposals(self, proposal_path):
        if proposal_path is None:
            return [], [], 0, 0

        proposals = {}
        nimgs = ncats = 0
        with open(proposal_path, 'rb') as f:
            proposals = pickle.load(f)

            segms = proposals['all_segms']
            boxes = proposals['all_boxes']

            nimgs = len(segms[0])
            ncats = len(segms)

        return segms, boxes, nimgs, ncats

    def get_item_set(self, index):
        entry = self.annotations[index]

        image_filename = entry["image_filename"]
        image_id = os.path.splitext(image_filename)[0]

        # load image
        image_path = self.image_dict[image_id]
        image = Image.open(image_path).convert('RGB')
        img = self.preprocess(image)

        img_size = image.size

        image_idx = entry["image_index"]

        regions = []
        for c in range(1, self.nCats):
            for j, m in enumerate(self.segms[c][image_idx]):
                if self.boxes[c][image_idx][j][4] > 0.9:
                    regions.append(self.boxes[c][image_idx][j][:4])

        # print("num of regions:", len(regions))

        image_location = np.array(regions)
        image_h = img_size[1]
        image_w = img_size[0]
        num_boxes = len(regions)

        img_info = img_size + (num_boxes,)

        mix_num_boxes = min(int(num_boxes), self.region_len)
        mix_location_pad = np.zeros((self.region_len, 5))
        mix_region_pad = np.zeros((self.region_len, 4))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.region_len:
            image_mask.append(0)
        
        if mix_num_boxes > 0:
            mix_location_pad[:mix_num_boxes,:4] = image_location[:mix_num_boxes]
            mix_region_pad[:mix_num_boxes] = regions[:mix_num_boxes]

            mix_location_pad[:,4] = (mix_location_pad[:,3] - mix_location_pad[:,1]) * (mix_location_pad[:,2] - mix_location_pad[:,0]) / (float(image_w) * float(image_h))
            mix_location_pad[:,0] = mix_location_pad[:,0] / float(image_w)
            mix_location_pad[:,1] = mix_location_pad[:,1] / float(image_h)
            mix_location_pad[:,2] = mix_location_pad[:,2] / float(image_w)
            mix_location_pad[:,3] = mix_location_pad[:,3] / float(image_h)
        
        spatials = mix_location_pad
        regions = mix_region_pad
        
        # question
        question = entry["question"]
        progs = entry["program"]
        answer = entry["answer"]
        
        input_ids, segment_ids, input_mask = self.tokenize(progs, self.seq_len)

        answer_id = self.vocabs[answer]
        question_id = index

        return img, regions, img_info, spatials, image_mask, input_ids, segment_ids, input_mask, answer_id, question_id

    
    def tokenize(self, progs, max_length=52, padding_index=26):

        # 0-25 : function
        # 26 : no function (pad)
        # 27-45 : argument (0-18 + 27)
        # no arg : 46 = pad not used
        # cls, sep : 47, 48

        prog_tokens = []
        segments = []

        args = np.full((25, 2), 26)
        inputs_idx = -1
        for step, p in enumerate(progs):
            func = p['function']
            arg = p['value_inputs']
            inputs = p['inputs']

            func_idx = self.func_vocab[func]
            if func_idx == 0:
                inputs_idx = inputs_idx + 1
            elif len(inputs) == 2:
                inputs_idx = 0
            else:
                inputs_idx = args[inputs[0], 1]

            args[step] = [func_idx, inputs_idx]

            prog_tokens.append(func_idx)
            segments.append(inputs_idx + 1)

            arg_idx = self.args_vocab[arg[0]] + 27 if arg else 26
            prog_tokens.append(arg_idx)
            segments.append(inputs_idx + 1)
            
            # print(f'step: {num_progs} | {func_idx}: {self.func[func_idx]}, {arg_idx}: {self.arg[arg_idx]} | {inputs}: {inputs_idx}')
    
        token_ids = [47] + prog_tokens + [48]
        segment_ids = [0] + segments + [0]

        assert len(token_ids) <= 52

        input_mask = [1] * len(token_ids)

        max_length = 52
        if len(token_ids) < max_length:
            # Note here we pad in front of the sentence
            padding = [padding_index] * (max_length - len(token_ids))
            token_ids = token_ids + padding
            input_mask += [0] * len(padding)
            segment_ids += [0] * len(padding)

        return token_ids, segment_ids, input_mask