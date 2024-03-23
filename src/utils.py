import json
import pandas as pd


def load_dataset():
    paths = [r"../data/scicite/train.jsonl", 
             r"../data/scicite/dev.jsonl",
             r"../data/scicite/test.jsonl"]

    data = []

    for path in paths:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.json_normalize(data)
    df.drop(columns=["label2", "label2_confidence"], inplace=True)
    
    data_x, data_y = df["string"], df["label"]

    return data_x, data_y


if __name__ == "__main__":
    load_dataset()