# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision Transformer(VIT) model."""

import torch
from megatron import get_args
from megatron.model.utils import get_linear_layer
from megatron.model.vision.vit_backbone import VitBackbone, VitMlpHead
from megatron.model.vision.mit_backbone import mit_b3_avg
from megatron.model.vision.utils import trunc_normal_
from megatron.model.module import MegatronModule

class VitClassificationModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, num_classes, finetune=False,
                 pre_process=True, post_process=True):
        super(VitClassificationModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.finetune = finetune
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
            pre_process=self.pre_process,
            post_process=self.post_process,
            single_token_output=True
        )
        
        if self.post_process:
            if not self.finetune:
                self.head = VitMlpHead(self.hidden_size, self.num_classes)
            else:
                self.head = get_linear_layer(
                    self.hidden_size,
                    self.num_classes,
                    torch.nn.init.zeros_
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process:
            hidden_states = self.head(hidden_states)

        return hidden_states


class MitClassificationModel(MegatronModule):
    """Mix vision Transformer Model."""

    def __init__(self, num_classes
                 pre_process=True, post_process=True):
        super(MitClassificationModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes

        self.backbone = mit_b3_avg()
        self.head = torch.nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        hidden_states = self.backbone(input)
        hidden_states = self.head(hidden_states)

        return hidden_states
