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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MoeConfig(PeftConfig):
    init_moe_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Moe layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    num_experts: int = field(
        default=2,
        metadata={
            "help": (
                "Whether to initialize the weights of the Moe layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    topk: int = field(
        default=1,
        metadata={
            "help": (
                "Whether to initialize the weights of the Moe layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    save_all_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the moe router logits for analysis."
            ),
        },
    )
    aux_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "Whether to save the moe router logits for analysis."
            ),
        },
    )
    lpr_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "Whether to save the moe router logits for analysis."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.MOE
        if self.lpr_loss_coef is not None and self.aux_loss_coef is not None:
            assert NotImplementedError