import pandas as pd
import spacy
import numpy as np
from re import finditer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# So initialise this and use it to extract features from the dataset
# feature_list is a list of the features that you want to extract
# currently supports ["pos_tag", "word_vector", "word_embedding"]
# The function to extract features is just extract_features()
# set train=True for extracting features for the training dataset,
# Vectorizer and SVD for "word_vector" features needs to be fit during training
# train=False for test dataset

class FeatureExtractor():
    def __init__(self, feature_list, tokenizer="regex", word_vectorizer="count"):
        self.feature_list = feature_list
        self.tokenizer = tokenizer

        if "word_vector" in feature_list:
            if word_vectorizer == "tfidf":
                self.vectorizer = TfidfVectorizer()
            self.vectorizer = CountVectorizer()
        
            self.word_vector_svd = TruncatedSVD(n_components=100)

        if "word_embedding" in feature_list:
            self.word_embedder = spacy.load("en_core_web_lg")
        
    def regex_tokenize(self, string):
        tokenizer_pattern = r"\b[A-Za-z]+(-[A-Za-z]*)*\b" + r"|" +\
                    r"(?<=[a-z])['][a-z]+" + r"|" +\
                    r"\$[\d]+(,[\d]+)*(\.[\d]+)?" + r"|" +\
                    r"\b[\d]+(\.[\d]+)?%" + r"|" +\
                    r"\b[\d]+(\.[\d]+)?\b" + r"|" +\
                    r"\.\.\.|[()\\,;:.?!_'\"“”[\]<>\/\-*#–—]"
    
        tokens = [match.group() for match in finditer(tokenizer_pattern, string)]

        return tokens

    def tokenize_sentences(self, sentences):
        tokens_list = []

        for sentence in sentences:
            if self.tokenizer == "nltk":
                tokens_list.append(word_tokenize(sentence))
            else:
                tokens_list.append(self.regex_tokenize(sentence))

        return tokens_list

    def get_pos_tag_features(self, tokens, type="count"):
        tag_types = {'.': 'n_punct', 'ADJ': 'n_adj', 'ADP': 'n_adp', 'ADV': 'n_adv', 'CONJ': 'n_conj', 'DET': 'n_det', 'NOUN': 'n_noun', 
                    'NUM': 'n_num', 'PRON': 'n_pron', 'PRT': 'n_part', 'VERB': 'n_verb', 'X': 'n_other'}

        pos_features = []
        for tokenized in tokens:
            tags = pos_tag(tokenized, tagset="universal")

            counts = {tag: 0 for tag in tag_types.values()}
            for _, tag in tags:
                counts[tag_types[tag]] += 1

            if type == "prop":
                n_tokens = len(tokenized)
                for tag in counts:
                    counts[tag] /= n_tokens

            pos_features.append(counts)

        return pd.DataFrame(pos_features)

    def get_word_vector_features(self, sentences, train=False):
        if train:
            word_vectors = self.vectorizer.fit_transform(sentences)
            wv_features = self.word_vector_svd.fit_transform(word_vectors)
        else:
            word_vectors = self.vectorizer.transform(sentences)
            wv_features = self.word_vector_svd.transform(word_vectors)

        wv_cols = [f"wv{i}" for i in range(wv_features.shape[1])]
        return pd.DataFrame(wv_features, columns=wv_cols)

    def get_word_embedding_features(self, sentences):
        we_features = []

        for sentence in sentences:
            doc = self.word_embedder(sentence)
            we_features.append(doc.vector)

            if len(we_features) % 1000 == 0:
                print(f"Embedding Progress: {len(we_features)} / {len(sentences)}", end="\r")

        we_features = np.array(we_features)

        print("                                            ", end="\r")
        we_cols = [f"we{i}" for i in range(we_features.shape[1])]
        return pd.DataFrame(we_features, columns = we_cols)

    def extract_features(self, sentences, train=False):
        tokens = self.tokenize_sentences(sentences)

        feature_ls = []
        if "pos_tag" in self.feature_list:
            pos_tag_feats = self.get_pos_tag_features(tokens)
            feature_ls.append(pos_tag_feats)
        if "word_vector" in self.feature_list:
            wv_feats = self.get_word_vector_features(sentences, train=train)
            feature_ls.append(wv_feats)
        if "word_embedding" in self.feature_list:
            we_feats = self.get_word_embedding_features(sentences)
            feature_ls.append(we_feats)

        feature_df = feature_ls[0]
        for feature in feature_ls[1:]:
            feature_df = feature_df.join(feature)

        return feature_df

