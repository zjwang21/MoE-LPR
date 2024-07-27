from pandas import read_parquet
from tqdm import tqdm
import os
lang = "id"
data = []
path_dir ="/mnt/vol1/huangxin/nanda/dataset/CulturaX"
for k in tqdm(os.listdir("{}/{}".format(path_dir,lang))):
    path = os.path.join("{}/{}".format(path_dir,lang), k)
    print(path)
    data.extend(read_parquet(path)['text'])

import json
with open("/mnt/vol1/huangxin/nanda/dataset/CulturaX/stackdedup_{}.jsonl".format(lang), 'w') as f:
    for k in tqdm(data):
        f.write(json.dumps({"text": k}, ensure_ascii=False) + "\n")
