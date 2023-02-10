# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2022 Fujitsu Limited. All rights reserved.
# Licensed under the BSD 3-Clause Clear License. See LICENSE for details.

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import MultiScaleRoIAlign

class FeatureExtractor(nn.Module):
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

        self.proj = nn.Linear(config.v_feature_size * 7 * 7, config.v_feature_size)

    def forward(
        self, img, regions, image_info,
    ):
        region_props = []
        image_shapes = []

        batch_size = img.size(0)

        # without back propagation
        with torch.no_grad():
            features = self.extractor(img)

            num_regions = 0

            # print(features.keys())
            # for k in features.keys():
            #     print(k, features[k].shape)

            for idx in range(batch_size):
                image_shapes.append((image_info[idx, 1], image_info[idx, 0]))
                region = regions[idx, :image_info[idx, 2]]
                region_props.append(region)

                num_regions += image_info[idx, 2]

            # print("num regions", num_regions)

            # print("region_props", region_props[0].shape)
            # print("image_shapes", image_shapes[0])

            box_features = self.box_roi_pool(features, region_props, image_shapes)

            # print(box_features.shape)
        
            x = box_features.flatten(start_dim=1)

        # with back propagation
        # features = self.extractor(img)

        # for idx in range(batch_size):
        #     image_shapes.append((image_info[idx, 1], image_info[idx, 0]))
        #     region = regions[idx, :image_info[idx, 2]]
        #     region_props.append(region)

        # box_features = self.box_roi_pool(features, region_props, image_shapes)

        # # print(box_features.shape)
    
        # x = box_features.flatten(start_dim=1)

        region_features = torch.zeros((batch_size, 37, x.size(1)), dtype=torch.float, device=img.device)

        obj_idx = 0
        for idx in range(batch_size):
            region_features[idx, 1:image_info[idx, 2]+1] = x[obj_idx:obj_idx+image_info[idx, 2]]
            region_features[idx, 0] = region_features[idx, 1:image_info[idx, 2]+1].mean(dim=0)
            obj_idx += image_info[idx, 2]

        # print(region_features.shape)

        region_features = self.proj(region_features)

        # print(region_features.shape)

        # exit()

        return region_features