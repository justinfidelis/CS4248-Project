import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor
from sklearn.metrics import f1_score, accuracy_score

from utils import load_dataset

class CitationDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.features, self.labels = torch.tensor(x, dtype=torch.float32).cuda(), self.label_to_idx(y).cuda()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def label_to_idx(self, y):
        mapping = {"background": 0, "method": 1, "result": 2}
        out = []
        for i in y:
            out.append(mapping[i])
        return torch.tensor(out, dtype=torch.long)
            

class MLP(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 3)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = nn.CrossEntropyLoss()

    def train(self, dataloader, epochs=30):
        self.model.train()

        for t in range(epochs):
            for batch, (x, y) in enumerate(dataloader):
                pred = self.model(x)
                loss = self.loss(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"Epoch: {t} / {epochs} Loss: {loss.item()}")

    def forward(self, X):
        return self.model(X.cuda()).cpu()

    def predict(self, data):
        self.model.eval()

        output = []
        for X, y in data:
            logits = self.model(X).cpu()
            pred = np.argmax(logits.detach().numpy())
            output.append(pred)

        return np.array(output)


def idx_to_label(y):
    mapping = ["background", "method", "result"]
    out = []
    for i in y:
        out.append(mapping[i])
    return out

if __name__ == "__main__":
    data_x, data_y = load_dataset()
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

    feature_list = {"glove_embedding", "word_vector"}
    vect = "count"
    vect_pca = True

    feat_ext = FeatureExtractor(feature_list=feature_list, word_vectorizer=vect, vector_pca=True)
    train_feat = feat_ext.extract_features(train_x, train=True).values
    test_feat = feat_ext.extract_features(test_x).values

    train_data = CitationDataset(train_feat, train_y.values)
    test_data = CitationDataset(test_feat, test_y.values)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data)

    model = MLP(200)
    model.cuda()

    model.train(train_dataloader)
    test_pred = idx_to_label(model.predict(test_dataloader))

    print(f"Model: MLP_torch", f", Features: {feature_list}")
    if "word_vector" in feature_list:
        print(f"Vectorizer: {vect}") 
        print(f"Vector PCA: {vect_pca}")
    print("Accuracy: ", accuracy_score(test_y, test_pred))
    print("F1 Score: ", f1_score(test_y, test_pred, average="macro"))

    

