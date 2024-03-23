from sklearn.linear_model import LogisticRegression
from feature_extractor import extract_features

class LogReg():
    def __init__(self):
        self.model = LogisticRegression(max_iter=6000)
        self.feature_selection = ["pos_tag", "word_vector", "word_embedding"]

    def train(self, train_x, train_y, is_features=False):
        if is_features:
            train_feats = train_x
        else:
            train_feats = extract_features(train_x, self.feature_selection, train=True)

        self.model.fit(train_feats, train_y)


    def predict(self, X, is_features=False):
        if is_features:
            feats = X
        else:
            feats = extract_features(X, self.feature_selection, train=False)
        pred = self.model.predict(feats)

        return pred