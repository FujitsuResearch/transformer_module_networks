# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from transformers.modeling_bert import BertLayer

class TransformerModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_module_layer)])

    def forward(
        self,
        hidden_states,
        args,
        attention_mask=None,
    ):
        len_features = hidden_states.size(1)
        hidden_states = torch.cat([hidden_states, args], dim=1)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, None, None, None
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states[:, :len_features],)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class ImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super(ImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.hidden_size)

        self.use_location_embed = config.use_location_embed
        if self.use_location_embed:
            self.image_location_embeddings = nn.Linear(5, config.hidden_size)
        
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):

        img_embeddings = self.image_embeddings(input_ids)

        if self.use_location_embed:
            loc_embeddings = self.image_location_embeddings(input_loc)        
            img_embeddings = img_embeddings + loc_embeddings

        embeddings = self.layerNorm(img_embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class TransformerModuleNet(nn.Module):
    def __init__(self, config, num_modules=12, max_prog_seq=25, num_progs=26, num_args=20, num_labels=32):
        super().__init__()

        self.num_modules = num_modules if config.map else 26 
        self.max_prog_seq = max_prog_seq
        self.num_progs = num_progs
        self.num_args = num_args + num_progs

        config.output_attentions = False
        self.num_region = config.num_region

        self.t_modules = nn.ModuleList([TransformerModule(config) for _ in range(self.num_modules)])

        self.img_embeddings = ImageEmbeddings(config)
        self.arg_embeddings = nn.Embedding(self.num_args, config.hidden_size)
        self.position_embeddings = nn.Embedding(self.num_region, config.hidden_size)     # 37 default, 151 vt

        self.pred_head = SimpleClassifier(config.hidden_size, config.hidden_size*2, num_labels, 0.5)

        self.map = config.map
        if config.map == "func":
            self.func_map = {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 7, 9: 8, 10: 7, 11: 8, 12: 8, 13: 8, 14: 9, 15: 9, 16: 9, 17: 9, 18: 10, 19: 10, 20: 10, 21: 10, 22: 11, 23: 11, 24: 11, 25: 11
                }
            print("use func map")
        elif config.map == "random":
            self.func_map = {
                18: 0, 20: 1, 5: 2, 24: 3, 12: 4, 7: 5, 11: 6, 16: 7, 3: 7, 23: 8, 6: 7, 15: 8, 1: 8, 25: 8, 13: 9, 19: 9, 0: 9, 9: 9, 21: 10, 8: 10, 2: 10, 4: 10, 17: 11, 10: 11, 14: 11, 22: 11
            }
            print("use random map")
        elif config.map == "arg":
            self.func_map = {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 7, 9: 8, 10: 7, 11: 9, 12: 10, 13: 11, 14: 8, 15: 9, 16: 10, 17: 11, 18: 8, 19: 9, 20: 10, 21: 11, 22: 8, 23: 9, 24: 10, 25: 11
            }
            print("use arg map")
        elif config.map == "order":
            self.func_map = {i: i%num_modules for i in range(50)}
            print("use order map")
        else:
            self.func_map = { i: i for i in range(num_progs) }
            print("direct (no mapping)")

    def forward(
        self,
        features, 
        spatials, 
        image_mask, 
        args,
        attention_mask=None,
    ):
        hidden_states = self.img_embeddings(features, spatials)
        bs = hidden_states.size(0)
        l = hidden_states.size(1)

        position_ids = torch.arange(
            l, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).repeat(bs, 1)
        position_embed = self.position_embeddings(position_ids)

        hidden_states = hidden_states + position_embed

        for step in range(self.max_prog_seq):
            for b in range(bs):
                orig_id = args[b, step, 0].detach().item()
                if orig_id < self.num_progs:
                    func_id = self.func_map[step] if self.map == "order" else self.func_map[orig_id]
                    arg_id = args[b:b+1, step, 0:2]
                    arg_embed = self.arg_embeddings(arg_id)
                    
                    module_output = self.t_modules[func_id](hidden_states[b:b+1], arg_embed)

                    hidden_states[b] = module_output[0]
                else:
                    continue

        outputs = (hidden_states,)

        pred = self.pred_head(hidden_states[:, 0])

        return outputs, pred
        