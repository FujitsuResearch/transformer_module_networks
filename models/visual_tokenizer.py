# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import MultiScaleRoIAlign

from models.position_encoding import PositionEmbeddingSine

class VisualTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.backbone.out_channels = 2048

        # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"} 
        return_layers = {"layer4": "0"}
        self.extractor = IntermediateLayerGetter(self.backbone, return_layers=return_layers)

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        self.proj = nn.Linear(config.v_feature_size, config.v_feature_size)
        self.pos_embed = PositionEmbeddingSine(config.v_feature_size // 2, normalize=True)
        
        self.norm = nn.LayerNorm(config.v_feature_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, img, regions, image_info,
    ):
        region_props = []
        image_shapes = []

        batch_size = img.size(0)

        with torch.no_grad():
            features = self.extractor(img)['0']

        mask = torch.zeros((batch_size, features.size(2), features.size(3)), dtype=torch.bool, device=img.device)
        pos = self.pos_embed(mask)

        features = features.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)

        visual_tokens = self.proj(features)
        visual_tokens = visual_tokens + pos

        head_token = torch.mean(visual_tokens, dim=1, keepdim=True)

        visual_tokens = torch.cat([head_token, visual_tokens], dim=1)

        visual_tokens = self.norm(visual_tokens)
        visual_tokens = self.dropout(visual_tokens)

        image_mask = ~mask
        image_mask = image_mask.flatten(1).long()
        image_mask = torch.cat([torch.ones((batch_size, 1), device=img.device), image_mask], dim=1)     # add mask for head

        return visual_tokens, image_mask