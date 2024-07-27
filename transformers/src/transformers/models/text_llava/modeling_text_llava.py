# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_text_llava import TextLlavaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TextLlavaConfig"

TEXT_LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/bakLlava-v1-hf",
    # See all Llava models at https://huggingface.co/models?filter=llava
]


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class TextLlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    left_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TextLlavaProjector(nn.Module):
    def __init__(self, config: TextLlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.left_config.hidden_size, config.right_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.right_config.hidden_size, config.right_config.hidden_size, bias=True)

    def forward(self, left_features):
        hidden_states = self.linear_1(left_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TextLlavaPreTrainedModel(PreTrainedModel):
    config_class = TextLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TextLlavaAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.right_config.initializer_range
        )
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


class TextLlavaForConditionalGeneration(TextLlavaPreTrainedModel):
    def __init__(self, config: TextLlavaConfig):
        super().__init__(config)
        self.left_model = AutoModel.from_config(config.left_config)

        self.projector = TextLlavaProjector(config)
        self.vocab_size = config.vocab_size
        self.right_model = AutoModelForCausalLM.from_config(
            config.right_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 2
        self.post_init()

    def set_training_stage(self, stage):
        assert stage in [1, 2], "stage only supported 1 or 2!"
        if stage == 1:
            for n, p in self.left_model.named_parameters():
                p.requires_grad = False
            for n, p in self.right_model.named_parameters():
                p.requires_grad = False
            for n, p in self.projector.named_parameters():
                p.requires_grad = True
            self.config.training_stage = stage
            logger.info("Prepare training params in stage 1, only update params of projector.")
        if stage == 2:
            for n, p in self.left_model.named_parameters():
                p.requires_grad = False
            for n, p in self.right_model.named_parameters():
                p.requires_grad = True
            for n, p in self.projector.named_parameters():
                p.requires_grad = True
            self.config.training_stage = stage
            logger.info("Prepare training params in stage 2, update up and llama.")

    def load_from_pretrained_first(self, path_a, path_b):
        self.left_model = self.left_model.from_pretrained(path_a)
        self.right_model = self.right_model.from_pretrained(path_b)

    def get_input_embeddings(self):
        return self.right_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.right_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.right_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.right_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.right_model.set_decoder(decoder)

    def get_decoder(self):
        return self.right_model.get_decoder()

    def tie_weights(self):
        return self.right_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.right_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.right_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_left_features(self, left_features, inputs_embeds, input_ids, attention_mask, labels, left_inputs):
        bsz, left_seqlen, embed_dim = left_features.shape
        bsz1, right_seqlen = input_ids.shape
        assert bsz == bsz1, "only supported one special tokens now!"

        # cal the true max len for padding after inject bert features.
        left_lens = torch.sum(left_inputs['attention_mask'] == 1, dim=-1)
        right_lens = torch.sum(attention_mask == 1, dim=-1)
        final_max_len = max([right_lens[idx] + left_lens[idx] for idx in range(bsz)]) # 154

        final_embedding = torch.zeros(
            bsz, final_max_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            bsz, final_max_len, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (bsz, final_max_len), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        left_features = list(left_features[left_inputs['attention_mask']].split(left_lens))
        right_features = list(inputs_embeds[attention_mask].split(right_lens))
        true_input_ids = list(input_ids[attention_mask].split(right_lens))
        for idx in bsz:
            pass

    def _merge_input_ids_with_image_features(self, left_features, inputs_embeds, input_ids, attention_mask, labels, left_inputs):
        bsz, left_seqlen, embed_dim = left_features.shape
        bsz1, right_seqlen = input_ids.shape
        assert bsz == bsz1, "only supported one special tokens now!"
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))

        # cal the true max len for padding after inject bert features.
        left_lens = torch.sum(left_inputs['attention_mask'] == 1, dim=-1)
        right_lens = torch.sum(attention_mask == 1, dim=-1)
        final_max_len = max([right_lens[idx] + left_lens[idx] for idx in range(bsz)]) # 154

        final_embedding = torch.zeros(
            bsz, final_max_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            bsz, final_max_len, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (bsz, final_max_len), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        if left_padding == False:
            for idx in range(bsz):
                # fill in bert and embeds features, pos_left, pos_right
                # repres the seqlen of this sample in left and right model.
                pos_left, pos_right = left_lens[idx], right_lens[idx]
                #<s>要放在最开始
                final_embedding[idx, 0, :] = inputs_embeds[idx, 0, :]
                final_embedding[idx, 1:pos_left + 1, :] = left_features[idx, :pos_left, :]
                final_embedding[idx, pos_left + 1:pos_left + pos_right, :] = inputs_embeds[idx, 1:pos_right, :]
                final_attention_mask[idx, :pos_left + pos_right] = 1

                if labels is not None:
                    final_labels[idx, 0] = labels[idx, 0]
                    final_labels[idx, pos_left + 1:pos_left + pos_right] = labels[idx, 1:pos_right]
        else:
            for idx in range(bsz):
                # fill in bert and embeds features, pos_left, pos_right
                # repres the seqlen of this sample in left and right model.
                pos_left, pos_right = left_lens[idx], right_lens[idx]
                final_embedding[idx, -pos_right + 1:, :] = inputs_embeds[idx, -pos_right + 1:, :]
                final_embedding[idx, -(pos_left + pos_right) + 1 : -pos_right + 1, :] = left_features[idx, :pos_left, :]
                final_embedding[idx, -(pos_left + pos_right), :] = inputs_embeds[idx, -pos_right, :]
                final_attention_mask[idx, -(pos_left + pos_right):] = 1

                if labels is not None:
                    final_labels[idx, -(pos_left + pos_right)] = labels[idx, -pos_right]
                    final_labels[idx, -pos_right + 1:] = labels[idx, -pos_right + 1:]

        position_ids = torch.cumsum(final_attention_mask, dim=-1) - 1
        position_ids.masked_fill_(final_attention_mask == 0, 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    @replace_return_docstrings(output_type=TextLlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        left_inputs: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        #vision_feature_layer: Optional[int] = None,
        #vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_left=None,
        input_ids_right=None,
    ) -> Union[Tuple, TextLlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # logger.info("left inputs: ")
        # logger.info(left_inputs)

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            #当input_id中只有一个token的时，是确保使用了past_key_value。
            if left_inputs is not None and input_ids.shape[1] != 1:
                left_outputs = self.left_model(**left_inputs)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                left_features = left_outputs.last_hidden_state

                left_features = self.projector(left_features)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    left_features, inputs_embeds, input_ids, attention_mask, labels, left_inputs
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and left_inputs is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.right_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                #attention_mask [bsz,seq_len]
                #logits[bsz,seq_len,vocab_size]
                shift_attention_mask = attention_mask[..., 1:]
                
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TextLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, left_inputs=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
                            
            #有问题，要改。
            
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            # elif self.config.image_token_index in input_ids:
            #     input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            
            input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "left_inputs": left_inputs,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)