## Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router

This repository contains the code for our paper [Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router](https://arxiv.org/abs/2211.12781)
## Quick Links

- [Enhancing Multilingual Capability for Large Language Models through Mixture-of-Experts with Language Priors Router](#enhancing-multilingual-capability-for-large-language-models-through-mixture-of-experts-with-language-priors-router)
- [Quick Links](#quick-links)
- [Overview](#overview)
  - [Stage 1: Upcycling for Post-pretraining](#stage-1:-upcycling-for-post-pretraining)
  - [Stage 2: Language Priors Router](#stage-2:-language-priors-router)
- [Main Results](#main-results)
- [Train MoE-LPR](#train-moe-lpr)
  - [Requirements](#requirements)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
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

First, install PyTorch by following the instructions from [the official website](https://pytorch.org/). To faithfully reproduce our results, please use the correct `1.10.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.10.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Then try runing the following script to install other dependencies.

```bash
pip install -r requirements.txt
cd fairseq-cipherdaug
pip install --editable ./
```

### Preprocessing
#### Data
All example scripts are based on the NIST Zh-En.
All the bash scripts are sufficiently annotated for reference.

Prepare the NIST Zh-En train and evaluation(MT02, MT03, MT04, MT08, MT06) data from https://catalog.ldc.upenn.edu/. 

Use MT06 as the valid data. Place train, valid, test data like this:
```
|-- StrokeNet/data/NIST/
    |-- source/
        |-- train.zh-en.zh
        |-- train.zh-en.en
        |-- valid.zh-en.zh
        |-- valid.zh-en.en
    |-- dev_test/
        | --nist02/
            | --nist02.en
            | --nist02.en1
            | --nist02.en2
            | --nist02.en3
            | --nist02.zh
        | --nist03/
        ......
```

Follow the procedure below to prerpocess the data.

#### Convert Chinese to Latinized stroke sequence and cipher with keys.

```bash
bash $LOC/StrokeNet/scripts/preprocess.sh
```
This creates all parallel Latinized stroke data of cipher-1 and cipher-2 in the output dir.
If you need to generate Latinized stroke data of your own, your file names should follow the rules mentioned in [DATA](#data) and take the following commands:
```bash
python $LOC/StrokeNet/fairseq-cipherdaug/strokenet/zh2letter.py \
    -i input_file_path \
    -o output_file_path \
    -v vocab_file_path \
    --workers n
```
This generates stroke sequence corpus in output_file_path of the files in input_file_path.
```bash
python $LOC/StrokeNet/fairseq-cipherdaug/strokenet/cipher.py \
    -i input_file_path \
    -s zh -t en --workers n \
    --keys 1 2
```
This generates ciphertexts with keys (1 and 2) in input_file_path.


#### Conduct BPE algorithm and binarize the data.
We use subword-nmt for BPE oprations.
For learning and applying BPE algorithm on all relevant files at once, use the `bpe.sh`
```
bash /home/StrokeNet/scripts/bpe.sh
```
Number of BPE merge operations can be changed in bash file.
This part could last for minutes, wait patiently for it to finish.

Then use `multi_binarize.sh` to generate joint multilingual dictionary and binary files for fairseq to use.
```
bash /home/StrokeNet/scripts/multi_binarize.sh
```

### Training

`train.sh` comes loaded with all relevant details to set hyperparameters and start training 
```
bash /home/StrokeNet/scripts/train.sh
```
Part of the key parameters:
```
fairseq-train $DATABIN --save-dir ${CKPT} \
    --lang-dict "${LANG_LIST}" --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs ${EVAL_LANG_PAIRS} \
    --task ${TASK} \                                         
    --arch transformer --share-all-embeddings \                 # Weight tying
    --criterion ${LOSS} --label-smoothing 0.1 \            
    --valid-subset valid --ignore-unused-valid-subsets --batch-size-valid 200 \
```
For keys 1 and 2:  

* `--lang-pairs` should be "zh-en,zh1-en,zh2-en". 
* ` --eval-lang-pairs` shoule be "zh-en,". 
* `--lang-dict` should be a file containing "zh, zh1, zh2, en".  
* `--task` should be "translation_multi_simple_epoch_cipher --prime-src zh --prime-tgt en".  
* `--criterion` should be "label_smoothed_cross_entropy_js --js-alpha 5 --js-warmup 500".   
*  `--js-alpha` is the coefficient of the consistent loss, StrokeNet does [consistency learning](#frequency-aware-ciphertext-based-data-augmentation)

See more details about the hyperparameters in our paper.


### Evaluation

```
bash /home/StrokeNet/scripts/eval.sh
```
Evaluation will be conducted on MT02, MT03, MT04, MT08, ALL OF THEM AND MT06(VALID SET). Results will be generated in the output checkpoint dir.

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