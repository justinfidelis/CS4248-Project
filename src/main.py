from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import utils
from naive_bayes import NaiveBayes
from log_regression import LogReg

# data_x, data_y = utils.load_dataset()
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

train_x, train_y = utils.load_features("train.csv")
test_x, test_y = utils.load_features("test.csv")

print(test_x)
model = LogReg()

model.train(train_x, train_y, is_features=True)

test_pred = model.predict(test_x, is_features=True)

print(f1_score(test_y, test_pred, average="macro"))
