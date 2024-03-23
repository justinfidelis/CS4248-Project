import json
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_extractor import extract_features

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

if __name__ == "__main__":
    data_x, data_y = load_dataset()

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

    feat_sel = ["pos_tag", "word_vector", "word_embedding"]

    train_feats = extract_features(train_x, feat_sel, True)
    test_feats = extract_features(test_x, feat_sel, False)

    save_features(train_feats, train_y, "train.csv")
    save_features(test_feats, test_y, "test.csv")