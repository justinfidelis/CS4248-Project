import json
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from matplotlib import pyplot as plt
import seaborn as sns


class SciCiteDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout, embedding_matrix, attention=False):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embeddings.weight.requires_grad = False  # freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention_layer = None
        if attention:
            self.attention_layer = SelfAttentionLayer(hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        output, (ht, ct) = self.lstm(x)
        if self.attention_layer is not None:
            output, _ = self.attention_layer.forward(ht[-1])
            return self.linear(output)
        else:
            return self.linear(ht[-1])


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout, embedding_matrix, attention=False):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention_layer = None
        if attention:
            self.attention_layer = SelfAttentionLayer(hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # rint(x.size())
        h_embedding = self.embedding(x)
        # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        if self.attention_layer is not None:
            h_lstm, _ = self.attention_layer.forward(h_lstm)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out


def clean_numbers(text):
    text = re.sub(r'[0-9]+', '', text)
    return text


def clean_text(text, remove_stop_words=True):
    text = text.lower()
    # text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = clean_numbers(text)

    # text = re.sub(r'\b([a-z]*ha+h[ha]*)\b', 'haha', text, flags=re.I)
    # text = re.sub(r'\b(o?l+o+l+[ol]*)\b', 'lol', text, flags=re.I)

    text = re.sub(r'([aeiou])\1{2,}', r'\1\1', text, flags=re.I)
    text = re.sub(r'([b-df-hj-np-tv-z])\1{2,}', r'\1', text, flags=re.I)

    valid_chars_list = [i for i in text if i.isalnum() or i.isspace()]
    text = ''.join(valid_chars_list)

    text = ' '.join([w for w in text.split() if len(w) > 1])

    # Replace multiple spaces with one space
    text = re.sub(r' +', ' ', text)
    text = ''.join(text)
    text = text.strip()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for word, tag in pos_tag(word_tokenize(text)):
        if tag.startswith('J'):
            w = lemmatizer.lemmatize(word, pos='a')
        elif tag.startswith('V'):
            w = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('N'):
            w = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('R'):
            w = lemmatizer.lemmatize(word, pos='r')
        else:
            w = word

        if remove_stop_words:
            if w not in stop_words:
                tokens.append(w)
        else:
            tokens.append(w)

    return ' '.join(tokens)


def build_vocab(X):
    word_counts = Counter()
    for text in X:
        word_counts.update(text.split())

    # creating vocabulary
    vocab2index = {'': 0, 'UNK': 1}
    words = ['', 'UNK']
    for word in word_counts:
        vocab2index[word] = len(words)
        words.append(word)
    vocab_size = len(words)
    return word_counts, vocab_size, vocab2index


def encode_sentence(text, vocab2index, encoding_length=30):
    tokenized = text.split()
    encoded = np.zeros(encoding_length, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index['UNK']) for word in tokenized])
    length = min(encoding_length, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded


def is_float(string):
    if string.replace('.', '').isnumeric():
        return True
    else:
        return False


def load_glove_vectors(glove_file="../data/glove_embeddings/model.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:] if is_float(x.replace('-', '')) is True])
    return word_vectors


def get_emb_matrix(word_counts, word_vecs, emb_size=300):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype='float32')
    W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25, 0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx


def convert_label_to_index(label):
    label_to_index_dict = {'background': 0, 'method': 1, 'result': 2}
    return label_to_index_dict[label]


def get_models_dict(vocab_size, dropout, pretrained_weights, attention):
    dl_models_dict = {}
    attention_text = ''
    if attention:
        attention_text = 'WithAttention'
    ########################## LSTM #############################################
    model_name = 'LSTM'
    model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=64, num_classes=3, dropout=dropout, embedding_matrix=pretrained_weights, attention=attention)
    model_checkpoint_path_template = 'saved_models/training/LSTM' + attention_text + '/dropout_' + str(dropout) + '/trained_model_LSTM' + attention_text
    model_path_for_test = 'saved_models/trained_model_LSTMWithAttention_dropout_0.5_epoch_26.pth'  # Substitute the name of model checkpoint you want to test (can be found in src/lstm/saved_models)

    dl_models_dict[model_name] = {'model': model, 'model_checkpoint_path_template': model_checkpoint_path_template, 'model_path_for_test': model_path_for_test}
    #############################################################################

    ########################## BiLSTM #############################################
    model_name = 'BiLSTM'
    model = BiLSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=64, num_classes=3, dropout=dropout, embedding_matrix=pretrained_weights, attention=attention)
    model_checkpoint_path_template = 'saved_models/training/BiLSTM' + attention_text + '/dropout_' + str(dropout) + '/trained_model_BiLSTM' + attention_text
    model_path_for_test = 'saved_models/trained_model_BiLSTMWithAttention_dropout_0.5_epoch_9.pth'  # Substitute the name of model checkpoint you want to test (can be found in src/lstm/saved_models)

    dl_models_dict[model_name] = {'model': model, 'model_checkpoint_path_template': model_checkpoint_path_template, 'model_path_for_test': model_path_for_test}
    #############################################################################

    return dl_models_dict


def load_dataset():
    paths = [r"../../data/scicite/train.jsonl",
             r"../../data/scicite/dev.jsonl",
             r"../../data/scicite/test.jsonl"]

    data = []

    for path in paths:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.json_normalize(data)
    df.drop(columns=["label2", "label2_confidence"], inplace=True)

    data_x, data_y = df["string"], df["label"]
    return data_x, data_y


def prepare_data():
    data_X, data_y = load_dataset()

    data_clean_X = data_X.apply(clean_text)
    data_y = data_y.apply(convert_label_to_index)

    X_train, X_test, y_train, y_test = train_test_split(data_clean_X, data_y, test_size=0.2, random_state=42, stratify=data_y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # print(f"Minimum training text length: {np.min(X_train.apply(lambda s: len(s.split(' '))))}")
    # print(f"Mean training text length: {np.mean(X_train.apply(lambda s: len(s.split(' '))))}")
    # print(f"Maximum training text length: {np.max(X_train.apply(lambda s: len(s.split(' '))))}")
    # print(f"90th quartile training text length: {np.quantile(X_train.apply(lambda s: len(s.split(' '))), 0.9)}")

    word_counts, vocab_size, vocab2index = build_vocab(X_train)

    X_train = X_train.apply(lambda s: np.array(encode_sentence(s, vocab2index)))
    X_val = X_val.apply(lambda s: np.array(encode_sentence(s, vocab2index)))
    X_test = X_test.apply(lambda s: np.array(encode_sentence(s, vocab2index)))

    train_ds = SciCiteDataset(np.array(X_train.to_numpy().tolist()), y_train.to_numpy())
    val_ds = SciCiteDataset(np.array(X_val.to_numpy().tolist()), y_val.to_numpy())
    test_ds = SciCiteDataset(np.array(X_test.to_numpy().tolist()), y_test.to_numpy())

    train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=10, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)

    if os.path.isfile('saved_models/pretrained_weights.npy'):
        pretrained_weights = np.load('saved_models/pretrained_weights.npy')
        print('Loaded pretrained_weights.npy from disk...')
    else:
        word_vecs = load_glove_vectors()
        pretrained_weights, vocab, vocab2index = get_emb_matrix(word_counts, word_vecs)
        np.save('saved_models/pretrained_weights.npy', pretrained_weights)
        print('Wrote pretrained_weights.npy to disk...')

    return train_dl, val_dl, test_dl, vocab_size, pretrained_weights


def classify(classifier_name, mode, epochs, lr, dropout, attention):
    print('Running text classification with {}, mode = {}, lr = {}, dropout = {}, attention = {} ...'.format(classifier_name, mode, lr, dropout, attention))
    train_dl, val_dl, test_dl, vocab_size, pretrained_weights = prepare_data()
    dl_models_dict = get_models_dict(vocab_size, dropout, pretrained_weights, attention)
    model = dl_models_dict[classifier_name]['model']
    model_checkpoint_path_template = dl_models_dict[classifier_name]['model_checkpoint_path_template']
    if mode == 'train':
        trained_model, train_losses, train_f1_scores, val_losses, val_f1_scores = train_model(model, train_dl, val_dl, model_checkpoint_path_template, epochs=epochs, lr=lr)
        generate_plots(classifier_name, dropout, attention, train_losses, train_f1_scores, val_losses, val_f1_scores)
    elif mode == 'test':
        trained_model = model
        model_path_for_test = dl_models_dict[classifier_name]['model_path_for_test']
        if os.path.isfile(model_path_for_test):
            print('Loading {} ...'.format(model_path_for_test))
            state_dict = torch.load(model_path_for_test)
            trained_model.load_state_dict(state_dict)
            preds_y, reals_y = test_model(trained_model, test_dl)
            # np.save('visualizations/preds_y_' + classifier_name + '_dropout_' + str(dropout) + '.npy', preds_y)
            # np.save('visualizations/reals_y_' + classifier_name + '_dropout_' + str(dropout) + '.npy', reals_y)
            calculate_and_display_metrics(classifier_name, preds_y, reals_y, dropout, attention)
        else:
            print('Model checkpoint file ' + model_path_for_test + ' does not exist! Please train the model first. Exiting...')
            return None
    else:
        print('Invalid mode provided! Expected values for mode are train or test. Exiting...')
        return None


def generate_plots(classifier_name, dropout, attention, train_losses, train_f1_scores, val_losses, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label='Training Loss')
    axes[0].plot(epochs, val_losses, label='Validation Loss')
    axes[0].legend(loc='upper right')
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(nbins=10, integer=True))
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss v Epoch')

    axes[1].plot(epochs, train_f1_scores, label='Training f1-score')
    axes[1].plot(epochs, val_f1_scores, label='Validation f1-score')
    axes[1].legend(loc='lower right')
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(nbins=10, integer=True))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('f1-score')
    axes[1].set_title('Training & Validation f1-score vs Epoch')

    attention_text = ''
    if attention:
        attention_text = 'WithAttention'
    fig.savefig('visualizations/dropout_' + str(dropout) + '/' + classifier_name + attention_text + '_training_vs_validation_loss_f1_score.pdf', format='pdf', bbox_inches='tight')


def calculate_and_display_metrics(classifier_name, preds_y, reals_y, dropout, attention):
    accuracy = round(metrics.accuracy_score(reals_y, preds_y), 4)
    precision = round(metrics.precision_score(reals_y, preds_y, average='macro'), 4)
    recall = round(metrics.recall_score(reals_y, preds_y, average='macro'), 4)
    f1_score_macro = round(metrics.f1_score(reals_y, preds_y, average='macro'), 4)
    f1_score_micro = round(metrics.f1_score(reals_y, preds_y, average='micro'), 4)

    print('Test Metrics:')
    print('Accuracy = {}, Precision = {}, Recall = {}, F1 score (macro) = {}, F1 score (micro) = {}'.format(accuracy, precision, recall, f1_score_macro, f1_score_micro))
    print()

    cf_matrix = np.around(metrics.confusion_matrix(reals_y, preds_y, normalize='true'), decimals=3)
    cf_matrix_df = pd.DataFrame(cf_matrix,
                                index=['background', 'method', 'result'],
                                columns=['background', 'method', 'result'])
    plt.figure(figsize=(10, 9))
    sns.set(font_scale=1.2)
    sns.heatmap(cf_matrix_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    attention_text = ''
    if attention:
        attention_text = 'WithAttention'
    plt.savefig('visualizations/' + classifier_name + attention_text + '_dropout_' + str(dropout) + '_confusion_matrix.pdf', format='pdf', bbox_inches='tight')


def train_model(model, train_dl, val_dl, checkpoint_path, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    train_losses, train_f1_scores, val_losses, val_f1_scores = [], [], [], []
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        preds, targets = [], []
        for x, y in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            total += y.shape[0]
            preds.append(pred)
            targets.append(y)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        train_loss, train_acc, train_f1 = sum_loss / total, correct / total, metrics.f1_score(targets, preds, average='macro')
        val_loss, val_acc, val_f1 = validation_metrics(model, val_dl)
        train_losses.append(train_loss)
        train_f1_scores.append(train_f1)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        print('epoch[%d/%d] train loss = %.3f, train accuracy = %.3f, train f1-score = %.3f, val loss = %.3f, val accuracy = %.3f, val f1-score = %.3f' % (i + 1, epochs, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1))
        torch.save(model.state_dict(), checkpoint_path + '_epoch_' + str(i + 1) + '.pth')
    return model, train_losses, train_f1_scores, val_losses, val_f1_scores


def validation_metrics(model, val_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    preds, targets = [], []
    for x, y in val_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        preds.append(pred)
        targets.append(y)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return sum_loss / total, correct / total, metrics.f1_score(targets, preds, average='macro')


def test_model(model, test_dl):
    model.eval()
    correct = 0
    total = 0
    preds, targets = [], []
    for x, y in test_dl:
        x = x.long()
        y = y.long()
        y_pred = model(x)
        pred = torch.max(y_pred, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        preds.append(pred)
        targets.append(y)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets


if __name__ == '__main__':
    #################### Define Classifier and Training Setup ####################
    classifier = 'BiLSTM'
    run_mode = 'test'
    n_epochs = 30
    lr = 0.00005  # 0.0001 for LSTM, 0.001 for LSTM with Attention, 0.00005 for BiLSTM, 0.00005 for BiLSTM with Attention
    classify(classifier_name=classifier, mode=run_mode, epochs=n_epochs, lr=lr, dropout=0.3, attention=True)