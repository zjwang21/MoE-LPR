# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for Llava.
"""


from typing import List, Optional, Union, Dict

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class TextLlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a Llava image processor and a Llava tokenizer into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["left_tokenizer", "right_tokenizer"]
    left_tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    right_tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, left_tokenizer=None, right_tokenizer=None):
        super().__init__(left_tokenizer, right_tokenizer)

    def __call__(
        self,
        right_text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        left_text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Union[BatchFeature, Dict]:
        if left_text is not None:
            left_inputs = self.left_tokenizer(
                left_text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
            )
        else:
            left_inputs = None

        right_inputs = self.right_tokenizer(
            right_text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        if padding and return_tensors == 'pt':  
            return BatchFeature(data={**right_inputs, "left_inputs": left_inputs})
        else:
            return {**right_inputs, "left_inputs": left_inputs}

    def pad(self, samples, ignore_index=None):
        left_inputs = samples.pop("left_inputs")
        if "labels" in samples.keys():
            labels = samples.pop("labels")
        else:
            labels = None
        #pad会返回attention_mask
        right_inputs = self.right_tokenizer.pad(samples, return_tensors='pt')
        left_inputs = self.left_tokenizer.pad(left_inputs, return_tensors='pt')

        if labels is not None:
            labels_padded = self.right_tokenizer.pad({"input_ids": labels}, return_tensors='pt')
            labels = labels_padded['input_ids'].masked_fill(labels_padded['attention_mask'] == 0, ignore_index)
            assert labels.size() == right_inputs['input_ids'].size()
            return {**right_inputs, "labels": labels, "left_inputs": left_inputs}
        
        return {**right_inputs, "left_inputs": left_inputs}

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.right_tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.right_tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        right_tokenizer_input_names = self.right_tokenizer.model_input_names
        left_tokenizer_input_names = self.left_tokenizer.model_input_names
        return list(dict.fromkeys(left_tokenizer_input_names + right_tokenizer_input_names))