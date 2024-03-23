from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import utils
from naive_bayes import NaiveBayes


data_x, data_y = utils.load_dataset()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

model = NaiveBayes()

model.train(train_x, train_y)

test_pred = model.predict(test_x)

print(f1_score(test_y, test_pred, average="macro"))
