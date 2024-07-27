from transformers import AutoTokenizer
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default=None)
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("/home/nfs02/model/llama2/hf/Llama-2-7b-hf")
def count(data):
    c = 0
    for line in tqdm(data):
        c += len(line['ids']) - 2
    return c

def tok(example):
    return {"ids": tokenizer(example['text'])['input_ids']}

def collect(data):
    res = []
    c = 0
    for k in tqdm(data):
        res.append(k)
        c += len(tokenizer(k['text'])['input_ids'])
        if c >= 1e9:
            break
    return res

if __name__ == "__main__":
    args = get_args()
    #src = load_dataset("json", data_files=args.i)['train'].shuffle(seed=22)
    src = load_dataset("/home/wangzj/LLaMA-Factory/data/slimpajam")['train'].shuffle(seed=22)
    print(src)
    #data = src.map(tok,
    #                num_proc=20,)
    #print(data)
    #res = count(data)
    #print(res / 1e9)

    final = collect(src)
    with open("/home/wangzj/LLaMA-Factory/data/slimpajam_1b.jsonl", 'w') as f:
        for k in tqdm(final):
            f.write(json.dumps(k) + '\n')