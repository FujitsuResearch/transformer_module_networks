# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from transformers.modeling_bert import BertEncoder, BertPreTrainedModel

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.LayerNorm_feat = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_loc = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_img = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout_img = nn.Dropout(config.hidden_dropout_prob)
        
        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)

        self.max_region_len = 36
        self.max_seq_len = 36

        # self.head_token = config.head_token
        self.head_token = 100 # 100, 3 for pg

    def forward(self, img_ids, img_loc, input_ids, token_type_ids=None):
        seq_length = self.max_seq_len

        batch_size = input_ids.size(0)

        input_ids_v = torch.zeros((batch_size, self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        input_ids_v[:, 0] = self.head_token
        input_ids_v[:] += 1        
        input_ids[:, 0] += 1
        input_ids = torch.cat([input_ids_v, input_ids], dim=-1)

        token_type_ids_v = torch.zeros((batch_size, self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        token_type_ids[:] += 1
        token_type_ids = torch.cat([token_type_ids_v, token_type_ids], dim=-1)

        # image regions
        position_ids_v = torch.zeros((self.max_region_len + 1), dtype=torch.long, device=input_ids.device)
        position_ids_v[1:] = position_ids_v[1:] + 1
        
        position_ids_t = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        # add +2 for image tokens
        position_ids_t = position_ids_t + 2
        position_ids = torch.cat([position_ids_v, position_ids_t], dim=0)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if embeddings.size(0) != img_ids.size(0):
            embeddings = embeddings.repeat((int)(img_ids.size(0) / embeddings.size(0)), 1, 1)

        # original image embeddings
        img_embeddings = self.image_embeddings(img_ids[:, 1:])
        loc_embeddings = self.image_location_embeddings(img_loc[:, 1:])
        img_embeddings = self.LayerNorm_feat(img_embeddings)
        loc_embeddings = self.LayerNorm_loc(loc_embeddings)

        img_embeddings = img_embeddings + loc_embeddings
        img_embeddings = self.dropout_img(self.LayerNorm_img(img_embeddings))

        embeddings[:, 1:self.max_region_len + 1] += img_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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

class BaseTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = Pooler(config)
        
        self.pred_head = SimpleClassifier(config.hidden_size, config.hidden_size*2, config.num_labels, 0.5)
        
        self.init_weights()

    def forward(
        self,
        input_ids,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
    ):
        extended_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=-1)
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_imgs, image_loc, input_ids, token_type_ids)

        head_mask = [None] * 12

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        pooled_output = encoder_outputs[0][:, 0]
        pred = self.pred_head(pooled_output)

        return pred