# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

from numpy.core.fromnumeric import shape
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
    def __init__(self, config, num_modules=26, max_prog_seq=25, num_progs=26, num_args=20, num_labels=32):
        super().__init__()

        self.num_modules = num_modules 
        self.max_prog_seq = max_prog_seq
        self.num_progs = num_progs
        self.num_args = num_args

        config.output_attentions = False
        self.num_region = config.num_region

        self.t_modules = nn.ModuleList([TransformerModule(config) for _ in range(self.num_modules)])

        self.img_embeddings = ImageEmbeddings(config)
        self.arg_embeddings = nn.Embedding(self.num_args, config.hidden_size)

        self.position_embeddings = nn.Embedding(self.num_region, config.hidden_size) # 74 default, 302 vt
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.pred_head = SimpleClassifier(config.hidden_size, config.hidden_size*2, num_labels, 0.5)


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
            l*2, dtype=torch.long, device=hidden_states.device
        )
        token_type_ids = torch.zeros(
            l*2, dtype=torch.long, device=hidden_states.device
        )
        token_type_ids[l:] += 1
        position_ids = position_ids.view(2, l).unsqueeze(0).repeat(bs, 1, 1).permute(1, 0, 2)
        token_type_ids = token_type_ids.view(2, l).unsqueeze(0).repeat(bs, 1, 1).permute(1, 0, 2)

        position_embed = self.position_embeddings(position_ids)
        token_type_embed = self.token_type_embeddings(token_type_ids)

        hs = hidden_states.unsqueeze(0).repeat(2, 1, 1, 1)
        hs = hs + position_embed + token_type_embed

        hi = [-1] * bs
        for step in range(self.max_prog_seq):
            for b in range(bs):
                func_id = args[b, step, 0].detach()
                if func_id < self.num_progs:
                    arg_id = args[b:b+1, step, 1:2]
                    arg_embed = self.arg_embeddings(arg_id)
                    hi[b] = args[b, step, 2].detach()
                    
                    if func_id in [3, 5, 7, 8, 9, 10, 11, 12, 13]:
                        # print(f'step: {step} | {func_id.item()}: {self.func[func_id]}, {arg_id.item()}: {self.arg[arg_id]} | {hi[b]} | merge')

                        hc = torch.cat((hs[0, b:b+1], hs[1, b:b+1]), dim=1)
                        module_output = self.t_modules[func_id](hc, arg_embed)
                        hs[0, b] = module_output[0][0:1, 0:l]
                    else:                    
                        # print(f'step: {step} | {func_id.item()}: {self.func[func_id]}, {arg_id.item()}: {self.arg[arg_id]} | {hi[b]} | batch : {b}')

                        module_output = self.t_modules[func_id](hs[hi[b], b:b+1], arg_embed)
                        hs[hi[b], b] = module_output[0]
                else:
                    continue

        outputs = (hs[0],)

        pred = self.pred_head(hs[0, :, 0])

        return outputs, pred
        