from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import utils
from feature_extractor import FeatureExtractor
from naive_bayes import NaiveBayes
from log_regression import LogReg
from mlp import MLP

data_x, data_y = utils.load_dataset()

model_type = "MLP"
feature_list = {"glove_embedding", "word_vector", "pos_tag"}  # {"word_embedding", "word_vector", "pos_tag", "glove_embedding"}
vect = "count"
vect_pca = True

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

if model_type in {"MLP", "LR"}:
    feature_ext = FeatureExtractor(feature_list=feature_list, word_vectorizer=vect, vector_pca=vect_pca)

    train_feat = feature_ext.extract_features(train_x, train=True)
    test_feat = feature_ext.extract_features(test_x, train=False)

if model_type == "MLP":
    model = MLP(212)
elif model_type == "LR":
    model = LogReg()
elif model_type == "NB":
    model = NaiveBayes(tokenizer=vect, alpha=1e-10)

if model_type in {"MLP", "LR"}:
    model.train(train_feat, train_y)
    test_pred = model.predict(test_feat)
elif model_type == "NB":
    model.train(train_x, train_y)
    test_pred = model.predict(test_x)


print(f"Model: {model_type}", f", Features: {feature_list}" if model_type in {"MLP", "LogReg"} else "")
if model_type == "NB" or (model_type in {"MLP", "LogReg"} and "word_vector" in feature_list):
    print(f"Vectorizer: {vect}") 
if (model_type in {"MLP", "LogReg"} and "word_vector" in feature_list):
    print(f"Vector PCA: {vect_pca}")
print("Accuracy: ", accuracy_score(test_y, test_pred))
print("F1 Score: ", f1_score(test_y, test_pred, average="macro"))
