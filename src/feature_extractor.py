import pandas as pd
import spacy
import numpy as np
from re import finditer, fullmatch
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from glove import Glove

# So initialise this and use it to extract features from the dataset
# feature_list is a list of the features that you want to extract
# currently supports ["basic", "pos_tag", "word_vector", "word_embedding"]
# The function to extract features is just extract_features()
# set train=True for extracting features for the training dataset,
# Vectorizer and SVD for "word_vector" features needs to be fit during training
# train=False for test dataset

def regex_tokenize(string):
    tokenizer_pattern = r"\b[A-Za-z]+(-[A-Za-z]*)*\b" + r"|" +\
                r"(?<=[a-z])['][a-z]+" + r"|" +\
                r"\$[\d]+(,[\d]+)*(\.[\d]+)?" + r"|" +\
                r"\b[\d]+(\.[\d]+)?%" + r"|" +\
                r"\b[\d]+(\.[\d]+)?\b" + r"|" +\
                r"\.\.\.|[()\\,;:.?!_'\"“”[\]<>\/\-*#–—]"

    tokens = [match.group() for match in finditer(tokenizer_pattern, string)]

    return tokens

def tokenize_sentence(sentence, tokenizer="regex"):
    if tokenizer == "nltk":
        return word_tokenize(sentence)
    return regex_tokenize(sentence)

def tokenize_sentences(sentences, tokenizer="regex"):
    return [tokenize_sentence(sentence, tokenizer) for sentence in sentences]

def reconstruct(tokens):
    return " ".join(tokens)

def remove_punctuation_tokens(tokens):
    pattern = r"\.\.\.|[()\\,;:.?!_'\"“”[\]<>\/\-*#–—]"
    
    return [token for token in tokens if not fullmatch(pattern, token)]

def remove_punctuations(sentence):
    tokens = tokenize_sentence(sentence)
    tokens = remove_punctuation_tokens(tokens)
    return reconstruct(tokens)

class FeatureExtractor():
    def __init__(self, feature_list, tokenizer="regex", word_vectorizer="count", vector_pca=False, vector_filter=False, glove_dims=100):
        self.feature_list = set(feature_list)
        self.tokenizer = tokenizer

        if vector_filter:
            stop_words = stopwords.words('english')
            tokenizer_pattern = r"(?u)\b[A-Za-z]+\b"
        else:
            stop_words = None
            tokenizer_pattern = r"(?u)\b\w+\b"

        if "word_vector" in feature_list:
            if word_vectorizer == "tfidf":
                self.vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=tokenizer_pattern)
            else:
                self.vectorizer = CountVectorizer(stop_words=stop_words, token_pattern=tokenizer_pattern)

            if vector_pca:
                self.word_vector_svd = TruncatedSVD(n_components=100)

            self.vector_pca = vector_pca

        if "word_embedding" in feature_list:
            self.langauge_model = spacy.load("en_core_web_lg")

        if "glove_embedding" in feature_list:
            self.glove = Glove(glove_dims)

    def get_basic_features(self, tokens):
        basic_features = []

        stop_words = set(stopwords.words('english'))
        word_pattern = r"\b[A-Za-z]+(-[A-Za-z]*)*(['][a-z]+)?\b"
        number_pattern = r"\$[\d]+(,[\d]+)*(\.[\d]+)?" + r"|" +\
                        r"\b[\d]+(\.[\d]+)?%" + r"|" +\
                        r"\b[\d]+(\.[\d]+)?\b"
        punct_pattern = r"\.\.\.|[()\\,;:.?!_'\"“”[\]<>\/\-*#–—]"
        
        for tokenized in tokens:
            feats = {}
            feats["n_tokens"] = len(tokenized)
            feats["n_stopwords"] = len([t for t in tokenized if t in stop_words])
            feats["n_words"] = len([m.group() for m in (fullmatch(word_pattern, t) for t in tokenized) if m is not None])

            basic_features.append(feats)

        return pd.DataFrame(basic_features)

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

    def get_word_vector_features(self, sentences, train=False, pca=True):
        if train:
            word_vectors = self.vectorizer.fit_transform(sentences)
            if pca:
                word_vectors = self.word_vector_svd.fit_transform(word_vectors)
            else:
                word_vectors = word_vectors.todense()
        else:
            word_vectors = self.vectorizer.transform(sentences)
            if pca:
                word_vectors = self.word_vector_svd.transform(word_vectors)
            else:
                word_vectors = word_vectors.todense()
        
        wv_cols = [f"wv{i}" for i in range(word_vectors.shape[1])]
        return pd.DataFrame(word_vectors, columns=wv_cols)

    def get_word_embedding_features(self, sentences):
        we_features = []

        for sentence in sentences:
            doc = self.langauge_model(sentence)
            we_features.append(doc.vector)

            if len(we_features) % 1000 == 0:
                print(f"Embedding Progress: {len(we_features)} / {len(sentences)}", end="\r")

        we_features = np.array(we_features)

        print("                                            ", end="\r")
        we_cols = [f"we{i}" for i in range(we_features.shape[1])]
        return pd.DataFrame(we_features, columns = we_cols)

    def get_glove_embedding_features(self, tokens):
        embed_features = []
        for token in tokens:
            embeds = self.glove.embed_tokens(token)
            embeds = np.array(embeds)
            mean = np.mean(embeds, axis=0)
            embed_features.append(mean)

        embed_features = np.array(embed_features)

        embed_cols = [f"glove_we{i}" for i in range(self.glove.dims)]
        return pd.DataFrame(embed_features, columns = embed_cols)

    def extract_features(self, sentences, train=False):
        tokens = tokenize_sentences(sentences, self.tokenizer)

        feature_ls = []
        if "basic" in self.feature_list:
            basic_feats = self.get_basic_features(tokens)
            feature_ls.append(basic_feats)
        if "pos_tag" in self.feature_list:
            pos_tag_feats = self.get_pos_tag_features(tokens)
            feature_ls.append(pos_tag_feats)
        if "word_vector" in self.feature_list:
            wv_feats = self.get_word_vector_features(sentences, train=train, pca=self.vector_pca)
            feature_ls.append(wv_feats)
        if "word_embedding" in self.feature_list:
            we_feats = self.get_word_embedding_features(sentences)
            feature_ls.append(we_feats)
        if "glove_embedding" in self.feature_list:
            glove_feats = self.get_glove_embedding_features(tokens)
            feature_ls.append(glove_feats)

        feature_df = feature_ls[0]
        for feature in feature_ls[1:]:
            feature_df = feature_df.join(feature)

        return feature_df

