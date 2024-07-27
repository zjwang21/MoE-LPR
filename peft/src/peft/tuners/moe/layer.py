# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import os
import math
import copy
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer

class MoeLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("moe_experts", "moe_router_embedding")
    other_param_names = ("num_experts")

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.num_experts = {}
        self.moe_router_embedding = nn.ModuleDict({})
        self.moe_experts = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        if hasattr(base_layer, "gate_proj"):
            self.in_features = base_layer.gate_proj.in_features
        elif hasattr(base_layer, "fc1"):
            self.in_features = base_layer.fc1.in_features
        else:
            raise NotImplementedError

    def update_layer(self, base_layer, adapter_name, num_experts, init_moe_weights):
        self.num_experts[adapter_name] = num_experts
        self.moe_router_embedding[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)
        self.moe_experts[adapter_name] = nn.ModuleList([copy.deepcopy(base_layer) for _ in range(num_experts - 1)])

        if init_moe_weights:
            self.reset_moe_parameters(adapter_name)

        if hasattr(base_layer, "gate_proj"):
            weight = base_layer.gate_proj.weight
        elif hasattr(base_layer, "fc1"):
            weight = base_layer.fc1.weight
        else:
            raise NotImplementedError
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_moe_parameters(self, adapter_name):
        if adapter_name in self.moe_router_embedding.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.xavier_normal_(self.moe_router_embedding[adapter_name].weight)

class MLP(nn.Module, MoeLayer):
    # Moe implemented in a mlp layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        num_experts: int = 2,
        init_moe_weights: bool = True,
        topk: int = None,
        aux_loss_coef: float = None,
        lpr_loss_coef: float = None,
        **kwargs,
    ) -> None:
        super().__init__()
        MoeLayer.__init__(self, base_layer)

        self.aux_loss_coef = aux_loss_coef
        self.topk = topk
        self.lpr_loss_coef = lpr_loss_coef
        self._active_adapter = adapter_name
        self.update_layer(base_layer, adapter_name, num_experts, init_moe_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        router = self.moe_router_embedding[self.active_adapter[0]]  # b x s x e
        result, router_logits = self.topk_route(x, router, self.active_adapter[0]) 
        result = result.to(previous_dtype)
        return result, router_logits

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moe." + rep

    def topk_route(self, hidden_states, router, adapter=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk, dim=-1)
        if self.topk != 1:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts[adapter]).permute(2, 1, 0)

        experts = [self.base_layer] + [k for k in self.moe_experts[adapter]]
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts[adapter]):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits