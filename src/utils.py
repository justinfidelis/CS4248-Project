import json
import pandas as pd
from sklearn.model_selection import train_test_split

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
    
    data_x, data_y = df["string"], df["label"]
    return data_x, data_y

def load_dataset_split():
    paths = [r"../data/scicite/train.jsonl", 
             r"../data/scicite/dev.jsonl"]

    train = []
    test = []

    for path in paths:
        with open(path, "r") as f:
            for line in f:
                train.append(json.loads(line))

    with open(r"../data/scicite/test.jsonl") as f:
        for line in f:
            test.append(json.loads(line))

    train_df = pd.json_normalize(train)
    test_df = pd.json_normalize(test)
    
    return train_df["string"], train_df["label"], test_df["string"], test_df["label"]

def save_features(features, data_y, filename):
    path = r"../data/features/" + filename

    data_y = data_y.reset_index().drop(columns=["index"])

    features = features.assign(label=data_y)
    features.to_csv(path_or_buf=path)

def load_features(filename):
    path = r"../data/features/" + filename

    df = pd.read_csv(path)

    data_y = df["label"]
    data_x = df.drop(columns=["label"])

    return data_x, data_y


# if __name__ == "__main__":
#     train_x, train_y, test_x, test_y = load_dataset_split()

#     print(len(train_x), len(test_x))