# coding=utf-8
# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
""" Llava model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

TEXT_LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava-hf/llava-v1.5-7b": "https://huggingface.co/llava-hf/llava-v1.5-7b/resolve/main/config.json",
}


class TextLlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaForConditionalGeneration`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`LlavaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llava model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlavaForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava llava-1.5-7b style configuration
    >>> configuration = LlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = LlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "text_llava"
    is_composition = False

    def __init__(
        self,
        left_config=None,
        right_config=None,
        ignore_index=-100,
        left_token_index=32000,
        projector_hidden_act="gelu",
        training_stage=1,
        vocab_size=32000,
        **kwargs,
    ):
        
        
        self.ignore_index = ignore_index
        self.left_token_index = left_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vocab_size = vocab_size
        self.training_stage = training_stage

        self.left_config = left_config
        if isinstance(self.left_config, dict):
            left_config["model_type"] = left_config["model_type"] if "model_type" in left_config else "bert"
            self.left_config = CONFIG_MAPPING[left_config["model_type"]](**left_config)
        elif left_config is None:
            self.left_config = CONFIG_MAPPING["bert"]()        
        
        self.right_config = right_config
        if isinstance(self.right_config, dict):
            right_config["model_type"] = right_config["model_type"] if "model_type" in right_config else "llama"
            self.right_config = CONFIG_MAPPING[right_config["model_type"]](**right_config)
            self.vocab_size = self.right_config.vocab_size
        elif right_config is None:
            self.right_config = CONFIG_MAPPING["llama"]()

        super().__init__(**kwargs)