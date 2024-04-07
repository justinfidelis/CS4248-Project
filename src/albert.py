import utils
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from datetime import datetime
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import json
from matplotlib import pyplot as plt


MODEL_NAME = 'albert-base-v2'
MAX_SEQUENCE_LENGTH = 500


class SciCiteDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AlbertClassification(nn.Module):
    def __init__(self, model_pretrained):
        super(AlbertClassification, self).__init__()
        self.albert = model_pretrained
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(4096, 64)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(64, len(class_labels))

    def forward(self, x):
        x = self.albert(input_ids=x)
        x = x['last_hidden_state'][:, 0, :]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLu(x)
        logits = self.linear2(x)
        # No need for a softmax, because it is already included in the CrossEntropyLoss
        return logits


df_X, df_y = utils.load_dataset()
# df_X_tokens = df_X.apply(get_num_tokens)
# print(max(list(df_X_tokens)))
class_labels = list(df_y.unique())

tokenizer = transformers.AlbertTokenizer.from_pretrained(MODEL_NAME)
X_list = df_X.to_list()
X_pt = tokenizer(X_list, padding='max_length', max_length=MAX_SEQUENCE_LENGTH, truncation=True, return_tensors='pt')['input_ids']
y_number_list = list(np.unique(df_y, return_inverse=True)[1])
y_pt = torch.Tensor(y_number_list).long()

X_pt_train, X_pt_test, y_pt_train, y_pt_test = train_test_split(X_pt, y_pt, test_size=0.2, random_state=42, stratify=y_pt)

print(f'Size of training dataset: {len(X_pt_train)}')
print(f'Size of test dataset: {len(X_pt_test)}')
print(f'Classes: {class_labels}')

# Get train and test data in form of Dataset class
train_data_pt = SciCiteDataset(X=X_pt_train, y=y_pt_train)
test_data_pt = SciCiteDataset(X=X_pt_test, y=y_pt_test)

# Get train and test data in form of Dataloader class
train_loader_pt = DataLoader(train_data_pt, batch_size=32)
test_loader_pt = DataLoader(test_data_pt, batch_size=32)

config = transformers.AlbertConfig(dropout=0.2, attention_dropout=0.2)
albert_pt = transformers.AlbertModel.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True)

device = torch.device('mps')
model_pt = AlbertClassification(albert_pt).to(device)
print(f'Using {device} device')
# print(model_pt)

for param in model_pt.albert.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model_pt.parameters())
total_params_trainable = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
print(f'Number of parameters:{total_params}')
print(f'Number of trainable parameters: {total_params_trainable}')

epoch_start = 1
epoch_end = 5

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pt.parameters())
training_history = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'train_accuracy': [], 'valid_accuracy': []}

# Measure time for training
start_time = datetime.now()

checkpoints_path = '../albert_checkpoints'
checkpoint_filename_prefix = 'albert_epoch_'
checkpoint_metrics_filename = 'albert_training_history.txt'

for e in range(epoch_start, epoch_end):
    if e > 0:
        print(f"Starting training from epoch {(e + 1)}")
        model_pt = AlbertClassification(albert_pt).to(device)
        model_pt.load_state_dict(torch.load(f'{checkpoints_path}/{checkpoint_filename_prefix}{e}.sd'))
        with open(f'{checkpoints_path}/{checkpoint_metrics_filename}', 'r') as f:
            training_history = json.load(f)
        print(f'Loaded state dictionaries and training metrics from epoch {e}')

    model_pt.train()

    train_loss = 0.0
    train_accuracy = []

    for X, y in tqdm(train_loader_pt):
        X = X.to(device)
        y = y.to(device)

        prediction = model_pt(X)
        loss = criterion(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index == y)
        train_accuracy += accuracy

    train_accuracy = (sum(train_accuracy) / len(train_accuracy)).item()

    model_pt.eval()

    valid_loss = 0.0
    valid_accuracy = []
    for X, y in test_loader_pt:
        X = X.to(device)
        y = y.to(device)

        prediction = model_pt(X)
        loss = criterion(prediction, y)

        valid_loss += loss.item()

        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index == y)
        valid_accuracy += accuracy

    valid_accuracy = (sum(valid_accuracy) / len(valid_accuracy)).item()

    training_history['epoch'].append(e + 1)
    training_history['train_loss'].append(train_loss / len(train_loader_pt))
    training_history['valid_loss'].append(valid_loss / len(test_loader_pt))
    training_history['train_accuracy'].append(train_accuracy)
    training_history['valid_accuracy'].append(valid_accuracy)

    print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_loader_pt) :10.3f} \t\t Validation Loss: {valid_loss / len(test_loader_pt) :10.3f}')
    print(f'\t\t Training Accuracy: {train_accuracy :10.3%} \t\t Validation Accuracy: {valid_accuracy :10.3%}')

    torch.save(model_pt.state_dict(), f'{checkpoints_path}/{checkpoint_filename_prefix}{(e + 1)}.sd')
    with open(f'{checkpoints_path}/{checkpoint_metrics_filename}', 'w') as f:
        f.write(json.dumps(training_history))

end_time = datetime.now()
training_time_pt = (end_time - start_time).total_seconds()

with open(f'{checkpoints_path}/{checkpoint_metrics_filename}', 'r') as f:
    training_history = json.load(f)
print(f'Loaded training metrics from epoch {epoch_end}')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax[0].set(title='Loss')
ax[0].plot(training_history['train_loss'], label='Training')
ax[0].plot(training_history['valid_loss'], label='Validation')
ax[0].legend(loc='upper right')

ax[1].set(title='Accuracy')
ax[1].plot(training_history['train_accuracy'], label='Training')
ax[1].plot(training_history['valid_accuracy'], label='Validation')
ax[1].legend(loc='lower right')


def get_num_tokens(text):
    return len(nltk.word_tokenize(text))
