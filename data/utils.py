import json
import pandas as pd


def load_jsonl_pd(path):
    return pd.read_json(path, lines=True)


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data
