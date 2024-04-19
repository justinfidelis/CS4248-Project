from sklearn.linear_model import LogisticRegression

class LogReg():
    def __init__(self):
        self.model = LogisticRegression(max_iter=5000)

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, X):
        return self.model.predict(X)