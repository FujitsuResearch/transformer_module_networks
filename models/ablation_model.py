# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from transformers.modeling_bert import BertLayer, BertPreTrainedModel

from models.extractor import FeatureExtractor
from models.visual_tokenizer import VisualTokenizer

class BertDynamicEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers - 1
        self.num_head = config.num_head
        
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_hidden_layers)])
        self.head = nn.ModuleList([BertLayer(config) for _ in range(self.num_head)])

        self.num_region = config.num_region

    def forward(
        self,
        hidden_states,
        args,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        num_layers=0,
        head_type=0,
        split_args=False,
    ):
        bs = hidden_states.size(0)
        max_layers = max(num_layers)
        len_features = self.num_region

        if split_args:
            hs = []
            for b in range(bs):
                idx_t = num_layers[b] * 2 + 1
                h = hidden_states[b:b+1]
                ab = args[b:b+1, 1:3]
                hb = torch.cat([h, ab], dim=1)

                hs.append(hb)
            hidden_states = torch.cat(hs, dim=0)

            attention_mask = None

            # print("split_args[ 0 ]", hidden_states.shape)
        else:
            hidden_states = torch.cat([hidden_states, args], dim=1)
            # print("hidden_states", hidden_states.shape)
            
        all_hidden_states = ()
        all_attentions = ()
        # print("num_layers", num_layers)
        for i in range(max_layers):
            if split_args and i > 0:
                idx_a = i * 2 + 1
                hidden_states = torch.cat([hidden_states[:, :len_features], args[:, idx_a:idx_a+2]], dim=1)
                
                # print("split_args[", i, "]", hidden_states.shape, h.shape, ah.shape, ab.shape, at.shape)
                
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i < max_layers - 1:
                layer_module = self.layer[i]
                
                layer_outputs = layer_module(
                    hidden_states, attention_mask, None, None, None
                )

                if len(num_layers) == 1:
                    hidden_states = layer_outputs[0]
                    # print("stack", i)
                else:
                    l = []
                    hs = []
                    for b in range(bs):
                        if i < num_layers[b] - 1:
                            hs.append(layer_outputs[0][b:b+1])
                            l.append(i)
                        else:
                            hs.append(hidden_states[b:b+1])
                            l.append('')
                    # print("stack:", l)
                    hidden_states = torch.cat(hs, dim=0)
            else:
                if len(head_type) == 1:
                    layer_module = self.head[head_type[0]]

                    layer_outputs = layer_module(
                        hidden_states, None, None, None, None
                    )

                    hidden_states = layer_outputs[0]

                    # print("head:", head_type[0])
                else:
                    h = []
                    hs = []
                    for b in range(bs):
                        layer_module = self.head[head_type[b]]
                        
                        layer_outputs = layer_module(
                            hidden_states[b:b+1], None, None, None, None
                        )

                        hs.append(layer_outputs[0])
                        h.append(head_type[b])

                    # print("head:", h)
                    
                    hidden_states = torch.cat(hs, dim=0)

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


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        
        self.dynamic_layers = config.dynamic_layers
        self.dynamic_head = config.dynamic_head
        self.split_args = config.split_args

        self.num_region = config.num_region
        self.num_pos = 2 + 52

        self.word_embeddings = nn.Embedding(
            50, config.hidden_size, padding_idx=26 # 49 + 1 (visual)
        )
        self.position_embeddings = nn.Embedding(
            self.num_pos, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            4, config.hidden_size                   # 3 + 1 (visual)
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_layer_norm_feat = config.use_layer_norm_feat
        self.use_location_embed = config.use_location_embed
        if self.use_layer_norm_feat:
            self.LayerNorm_feat = nn.LayerNorm(config.hidden_size, eps=1e-12) # ext
            
        if self.use_location_embed:
            self.LayerNorm_loc = nn.LayerNorm(config.hidden_size, eps=1e-12)  # ext
            self.image_location_embeddings = nn.Linear(5, config.hidden_size) # ext

        self.LayerNorm_img = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout_img = nn.Dropout(config.hidden_dropout_prob)
        
        self.image_embeddings = nn.Linear(config.v_feature_size, config.hidden_size)

        self.max_region_len = config.max_region      # 36 default, 150 vt
        self.max_seq_len = 52

    def forward(self, img_ids, img_loc, input_ids, token_type_ids=None):
        batch_size = input_ids.size(0)

        input_ids_v = torch.zeros((batch_size, self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        input_ids_v[:, 0] = 47
        input_ids_v[:, 1:] = 49
        input_ids[:, 0] = 48
        input_ids = torch.cat([input_ids_v, input_ids], dim=-1)

        token_type_ids_v = torch.zeros((batch_size, self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        token_type_ids[:] += 1
        token_type_ids = torch.cat([token_type_ids_v, token_type_ids], dim=-1)

        # image regions
        position_ids_v = torch.zeros((self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        position_ids_v[1:] = position_ids_v[1:] + 1
        
        position_ids_t = torch.arange(
            self.max_seq_len, dtype=torch.long, device=input_ids.device
        )
        # add +2 for image tokens
        position_ids_t = position_ids_t + 2
        position_ids = torch.cat([position_ids_v, position_ids_t], dim=0)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if embeddings.size(0) != img_ids.size(0):
            embeddings = embeddings.repeat((int)(img_ids.size(0) / embeddings.size(0)), 1, 1)

        # original image embeddings
        img_embeddings = self.image_embeddings(img_ids[:, 1:])

        if self.use_layer_norm_feat:
            img_embeddings = self.LayerNorm_feat(img_embeddings)  # ext

        if self.use_location_embed:
            loc_embeddings = self.image_location_embeddings(img_loc[:, 1:]) # ext
            loc_embeddings = self.LayerNorm_loc(loc_embeddings)   # ext
            img_embeddings = img_embeddings + loc_embeddings      # ext

        img_embeddings = self.dropout_img(self.LayerNorm_img(img_embeddings))

        embeddings[:, 1:self.max_region_len + 1] += img_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

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


class AblationTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.dynamic_layers = config.dynamic_layers
        self.dynamic_head = config.dynamic_head
        self.split_args = config.split_args

        self.max_prog_seq = 25
        self.num_args = 54
        self.num_func = 26
        self.num_region = config.num_region
        num_pos = self.num_region if self.split_args else self.num_region + self.max_prog_seq * 2

        if self.dynamic_layers:
            config.num_hidden_layers = self.max_prog_seq
        config.num_head = 1 if not self.dynamic_head else 5 # ['compare_int', 'compare_attr', 'query_attr', 'count', 'exist']

        self.num_layers = self.max_prog_seq if self.dynamic_layers else config.num_hidden_layers

        self.encoder = BertDynamicEncoder(config)
        self.embeddings = BertEmbeddings(config)

        self.pred_head = SimpleClassifier(config.hidden_size, config.hidden_size*2, config.num_labels, 0.5)

        self.head_type_dict = { 1: 0, 2: 1, 7: 2, 8: 2, 9: 3, 10: 2, 11: 3, 12: 3, 13: 3, 18: 4, 19: 4, 20: 4, 21: 4 }
        
        self.init_weights()

    def forward(
        self,
        features, 
        spatials, 
        image_mask, 
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        bs = features.size(0)
        lf = features.size(1)

        extended_attention_mask = torch.cat([image_mask, attention_mask], dim=-1)
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = self.embeddings(features, spatials, input_ids, token_type_ids)
        
        args = input_ids[:, 1:-1].view(input_ids.size(0), -1, 2)
        arg_len = args.size(1)

        args_embed = hidden_states[:, lf:]
        hs = hidden_states[:, :lf]

        if self.dynamic_layers or self.dynamic_head or self.split_args:
            num_layers = [self.num_layers]
            head_type = [0]
            if self.dynamic_layers or self.dynamic_head:
                num_layers_dynamic = []
                head_type_dynamic = []
                for b in range(bs):
                    if args[b, -1, 0] < self.num_func:
                        func_id = args[b, -1, 0].item()
                        prog_len = self.max_prog_seq
                    else:
                        for step in range(arg_len):
                            if args[b, step, 0] >= self.num_func:
                                func_id = args[b, step-1, 0].item()
                                prog_len = step
                                break
                    num_layers_dynamic.append(prog_len)
                    head_type_dynamic.append(self.head_type_dict[func_id])
                
                if self.dynamic_layers:
                    num_layers = num_layers_dynamic
                if self.dynamic_head:
                    head_type = head_type_dynamic

                hidden_states = self.encoder(hs, args_embed, attention_mask=extended_attention_mask, num_layers=num_layers, head_type=head_type, split_args=self.split_args)[0]
        else:
            hidden_states = self.encoder(hs, args_embed, attention_mask=extended_attention_mask, num_layers=[self.num_layers], head_type=[0])[0]

        outputs = (hidden_states,)

        pred = self.pred_head(hidden_states[:, 0])

        return outputs, pred

class TransformerModuleNetWithExtractor(nn.Module):
    def __init__(self, config, transformer, extractor=None):
        super().__init__()

        self.config = config

        self.extractor = extractor
        self.transformer = transformer

    def forward(
        self,
        input_imgs, 
        spatials, 
        image_mask, 
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        region_props = None, 
        image_info = None,
    ):
        if self.extractor is not None:
            if isinstance(self.extractor, FeatureExtractor):
                input_imgs = self.extractor(input_imgs, region_props, image_info)
            if isinstance(self.extractor, VisualTokenizer):
                input_imgs, image_mask = self.extractor(input_imgs, region_props, image_info)

        outputs, prediction = self.transformer(input_imgs, spatials, image_mask, input_ids, token_type_ids, attention_mask)

        return outputs, prediction
        