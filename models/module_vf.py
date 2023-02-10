# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from transformers.modeling_bert import BertLayer

from models.extractor import FeatureExtractor
from models.visual_tokenizer import VisualTokenizer

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
        arguments,
        attention_mask=None,
        region_props = None, 
        image_info = None,
    ):
        if self.extractor is not None:
            if isinstance(self.extractor, FeatureExtractor):
                input_imgs = self.extractor(input_imgs, region_props, image_info)
            if isinstance(self.extractor, VisualTokenizer):
                input_imgs, image_mask = self.extractor(input_imgs, region_props, image_info)

        outputs, prediction = self.transformer(input_imgs, spatials, image_mask, arguments)

        return outputs, prediction
        