## Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router

This repository contains the code for our paper [Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router](https://arxiv.org/abs/2211.12781)
## Quick Links

- [Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router](#enhancing-multilingual-capability-for-large-language-models-through-mixture-of-experts-with-language-priors-router)
- [Quick Links](#quick-links)
- [Overview](#overview)
  - [Stage 1: Upcycling for Post-pretraining](#stage-1-upcycling-for-post-pretraining)
  - [Stage 2: Language Priors Router](#stage-2-language-priors-router)
- [Main Results](#main-results)
- [Train MoE-LPR](#train-moe-lpr)
  - [Requirements](#requirements)
  - [Data Preprocessing](#data-preprocessing)
  - [Post-pretraining](#post-pretraining)
  - [Language Priors Router Training](#language-priors-router-training)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview
We propose a strategy called MoE-LPR (Mixture-of-Experts with Language Priors Router) to enhance multilingual capability of LLMs. MoE-LPR employs a two-strategy training approach. First, we utilize the Mixture-of-Experts (MoE) architecture to upcycle the model, freezing all the original parameters and adding new experts. We post-pretrain the model without any original language data to improve the abilities of the expanded languages. Then, we incorporate a language priors router using very small amounts of data to recover the abilities of the original languages. We evaluate our model on multiple benchmarks and the results indicate that our method outperforms other post-pretraining methods. Our analysis shows that freezing the original parameters does not limit the model’s learning ability while preserving the knowledge of the original languages. Additionally, the language priors router successfully enables the model to utilize the knowledge of various languages within the parameters. Furthermore, with the MoE architecture, our method can maintain the same inference overhead while continuously increasing the total number of model parameters. Extensive experiments demonstrate that MoE-LPR effectively helps LLMs improve expanded languages and preserves proficiency in the original languages with superior scalability.

### Stage 1: Upcycling for Post-pretraining
As shown in the Figure, we upcycle the dense model to the MoE architecture. To enhance the MoE model’s multilingual capabilities while preserving its performance on the originally supported languages, we freeze the parameters of the original dense model within the MoE during post-pretraining on the expanded languages corpus. This approach retains the model’s existing knowledge and only updates the parameters of the newly added experts and router. These freezing parameters preserve the knowledge of not only the original languages but also the expanded language. The router can freely choose new experts or frozen experts, corresponding to storing new knowledge or reusing old knowledge.
<div align="center">
<img src=figures/model.png width=90% height=90% />
</div>

### Stage 2: Language Priors Router
After post-pretraining on the expanded languages corpus, the router, which has only been trained on the expanded languages data and not on the original languages corpus used for the dense model, may incorrectly assign experts for languages previously supported. This misallocation can lead to severe catastrophic forgetting in the MoE model. To retain the model's original capabilities while not affecting the performance on the extended languages, we attempt to train only the router with language priors. The number of the router parameters accounts for a negligible proportion. This language priors are that all the original languages tokens should be routed to the freezing expert during stage 1 while all the expanded languages tokens should keep their routing unchanged in this stage. We incorporate the language priors through a cross-entropy loss guided by the langauges of the trained tokens. Details are referred in our paper.

## Main Results
We show the main results of MoE-LPR on several LLM tasks. MoE-LPR outperforms other methods, particularly in ARC-Challenge, HellaSwag, and Belebele benchmarks. These results suggest that MoE-LPR has a strong learning capability to enhance the model's multilingual proficiency and excellent catastrophic forgetting prevention effects. Further details on each benchmark are provided in our paper.
<div align="center">
<img src=figures/main.png width=95% height=95% />
</div>


## Train MoE-LPR

In the following section, we provide instructions on training MoE-LPR with our code.

### Requirements

First, create a new conda environment through 'environment.yml'

```
conda env create -f environment.yml
```

Then try runing the following script to install other dependencies.

```bash
cd MoE-LPR/peft
pip install --editable ./

cd MoE-LPR/transformers
pip install --editable ./
```

### Data-Preprocessing
MoE-LPR concentrates on the post-pretraining stage. The data pipeline follows the LLaMA-Factory repo. In detail, prepare you documents to this path: MoE-LPR/LLaMA-Factory/data
The file endswith '.jsonl' or '.json' and the content follows:
```
{"text": your one doc}
```

Then add your file item in MoE-LPR/LLaMA-Factory/data/dataset_info.json.
For example:
```
"hu_1b": {
  "file_name": "hu_part1b_00000.jsonl",
  "file_sha1": "e70375e28eda542a90c68213640cc371898ce184",
  "language": "new",
  "columns": {
    "prompt": "text"
  }
},
```
The 'language' item means that if the language of your data is the original language or the expanded language ('old' or 'new'). This info will be used in the stage 2 LPR training. You can use the hyper-parameter "generate_lang_mask" to control whether use this info.


### Post-pretraining

`MoE-LPR/LLaMA-Factory/scripts/stage1.sh` comes loaded with all relevant details to set hyperparameters and start training 
```
cd MoE-LPR/LLaMA-Factory
bash scripts/stage1.sh
```
Part of the key parameters you can adjust:
* `--moe_num_experts`: how much experts in total. n-1 new experts are added and the rest one is the original FFN. 
* `--topk`: how much experts are selected for each token. 
* `--aux_loss_coef`: the weight of the load balancing loss.

The logs will log load balancing loss and the average scores per expert.
See more details about the hyperparameters in our paper.

### Language Priors Router Training

```
cd MoE-LPR/LLaMA-Factory
bash scripts/stage2.sh
```
Part of the key parameters you can adjust:
* `--lpr_loss_coef`: the weight of the lpr loss.
* `--max_samples`: how much docs are used for each language. 

The logs will log lpr loss and the average selections for the original ffn.
See more details about the hyperparameters in our paper.


## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhijun Wang (wzhijun21@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you use StrokeNet in your work:
```
@inproceedings{wang2022StrokeNet,
 author = {Wang, Zhijun and Liu, Xuebo and Zhang, Min},
 booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
 title = {Breaking the Representation Bottleneck of Chinese Characters: Neural Machine Translation with Stroke Sequence Modeling},
 url = {https://arxiv.org/abs/2211.12781}, 
 year = {2022}
}
```