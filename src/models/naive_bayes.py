from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class NaiveBayes():
    def __init__(self, alpha=1, tokenizer="count", remove_stop_words=False):
        self.model = MultinomialNB(alpha=alpha)

        if remove_stop_words:
            stop_words = stopwords.words('english') + [",", ".", "'m", "'ve", "'s", "'t", "'ll"]
        else:
            stop_words = []

        if tokenizer == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        else:
            self.vectorizer = CountVectorizer(stop_words=stop_words) 

    def train(self, train_x, train_y):
        train_feats = self.vectorizer.fit_transform(train_x)
        self.model.fit(train_feats, train_y)

    def predict(self, X):
        feats = self.vectorizer.transform(X)
        pred = self.model.predict(feats)
        return(pred)