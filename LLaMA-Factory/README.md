![# LLaMA Factory for Multilingual Moe](assets/logo.png)

## 本仓库是多语言moe的训练脚本代码，完整code请按如下流程

```bash
conda env create -f environment.yml
git clone https://github.com/zjwang21/moe-trainsformers.git
cd moe-transformers
pip install -e .

git clone https://github.com/zjwang21/moe-peft.git
cd moe-peft 
pip install -e .

LLaMA-Factory/llama-pt/qwen/moe.sh 是示例脚本
用作post-pretrain，其中指定的ar_2b数据集在
LLaMA-Factory/data
将你的单语数据整理成jsonl格式，每行是{"text": your text}
然后在dataset_info.json当中按照ar_2b的格式添加item
```