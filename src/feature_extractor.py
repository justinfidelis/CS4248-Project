import pandas as pd
import spacy
import numpy as np
from re import finditer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
word_vector_svd = TruncatedSVD(n_components=100)

def regex_tokenize(string):
    tokenizer_pattern = r"\b[A-Za-z]+(-[A-Za-z]*)*\b" + r"|" +\
                    r"(?<=[a-z])['][a-z]+" + r"|" +\
                    r"\$[\d]+(,[\d]+)*(\.[\d]+)?" + r"|" +\
                    r"\b[\d]+(\.[\d]+)?%" + r"|" +\
                    r"\b[\d]+(\.[\d]+)?\b" + r"|" +\
                    r"\.\.\.|[()\\,;:.?!_'\"“”[\]<>\/\-*#–—]"
    
    tokens = [match.group() for match in finditer(tokenizer_pattern, string)]

    return tokens

def tokenize_sentences(sentences, tokenizer="regex"):
    tokens_list = []

    for sentence in sentences:
        if tokenizer == "nltk":
            tokens_list.append(word_tokenize(sentence))
        else:
            tokens_list.append(regex_tokenize(sentence))

    return tokens_list

def get_pos_tag_features(tokens, type="count"):
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

def get_word_vector_features(sentences, vectorizer="count", train=False):
    if vectorizer == "tfidf":
        vect = tfidf_vectorizer
    else:
        vect = count_vectorizer
    
    if train:
        word_vectors = vect.fit_transform(sentences)
        wv_features = word_vector_svd.fit_transform(word_vectors)
    else:
        word_vectors = vect.transform(sentences)
        wv_features = word_vector_svd.transform(word_vectors)

    wv_cols = [f"wv{i}" for i in range(wv_features.shape[1])]
    return pd.DataFrame(wv_features, columns=wv_cols)

def get_word_embedding_features(sentences):
    we_features = []

    nlp_lg = spacy.load("en_core_web_lg")
    for sentence in sentences:
        doc = nlp_lg(sentence)
        we_features.append(doc.vector)

        if len(we_features) % 1000 == 0:
            print(f"Embedding Progress: {len(we_features)} / {len(sentences)}", end="\r")

    we_features = np.array(we_features)

    print("                                            ", end="\r")
    we_cols = [f"we{i}" for i in range(we_features.shape[1])]
    return pd.DataFrame(we_features, columns = we_cols)

def extract_features(sentences, feature_selection, train=False):
    feature_selection = set(feature_selection)

    tokens = tokenize_sentences(sentences)

    feature_ls = []
    if "pos_tag" in feature_selection:
        pos_tag_feats = get_pos_tag_features(tokens)
        feature_ls.append(pos_tag_feats)
    if "word_vector" in feature_selection:
        wv_feats = get_word_vector_features(sentences, train=train)
        feature_ls.append(wv_feats)
    if "word_embedding" in feature_selection:
        we_feats = get_word_embedding_features(sentences)
        feature_ls.append(we_feats)

    feature_df = feature_ls[0]
    for feature in feature_ls[1:]:
        feature_df = feature_df.join(feature)

    return feature_df

