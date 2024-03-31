import pandas as pd
import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from koila import lazy
import transformers
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm
import os
import time

# # INIT MODEL
batch_size = 8
learning_rate = 1e-5
epochs = 4
max_len1 = 512
seed_val = 28
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

config = BertConfig(
    max_length = max_len1,
    max_position_embeddings = max_len1,
)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 3
)

# # LOAD DATA
input_ids_core = torch.load('input_ids_core.pt')
attention_masks_core = torch.load('attention_masks_core.pt')
labels_core = torch.load('labels_core.pt')
input_ids_test = torch.load('input_ids_test.pt')
attention_masks_test = torch.load('attention_masks_test.pt')
labels_test = torch.load('labels_test.pt')
train_idx, val_idx = np.arange(50), np.arange(start=100,stop=120)
val_inputs, val_attention, val_labels = input_ids_core[val_idx], attention_masks_core[val_idx], labels_core[val_idx] 
(input_ids_c, labels_c) = lazy(input_ids_core[train_idx], labels_core[train_idx])
(input_ids_t, labels_t) = lazy(input_ids_test, labels_test)
# train_data = TensorDataset(input_ids_core[train_idx],attention_masks_core[train_idx],labels_core[train_idx])
# test_data = TensorDataset(input_ids_test,attention_masks_test,labels_test)
train_data = TensorDataset(input_ids_c,attention_masks_core[train_idx],labels_c)
test_data = TensorDataset(input_ids_t,attention_masks_test,labels_t)
print('Data Processed')

def train_model(model,rank,val_inputs=None,val_attention=None,val_labels=None):
    start_time = time.perf_counter()
    dist.init_process_group(backend='nccl',init_method='env://')
    device = f'cuda:{rank}'
    model.to(device)
    ddp_model = DDP(model, device_ids=[device])
    train_dataloader = DataLoader(
        train_data,
        sampler = torch.utils.data.distributed.DistributedSampler(train_data),
        batch_size = batch_size,
        num_workers = 0
    )
    optimizer = torch.optim.Adam(
        ddp_model.parameters(),
        lr = learning_rate,
        eps = 1e-8 #epsilon, to prevent division by zero
    )
    total_steps = len(train_dataloader) * epochs
    
    for epoch in tqdm(range(0,epochs)):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            result = model(
                batch_input_ids,
                token_type_ids = None, #KIV
                attention_mask = batch_attention_masks,
                labels = batch_labels,
                return_dict = True
            )
            loss = result.loss
            logits = result.logits #KIV
            loss.backward()
            clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if step % 100 == 0:
                print(f'Step {step}, time elapsed:{time.perf_counter()-start_time}')
        dist.destroy_process_group()
        
        # Evaluation
        model.eval()

        result = model(
                    val_inputs.to(device),
                    token_type_ids = None, #KIV
                    attention_mask = val_attention.to(device),
                    labels = val_labels.to(device),
                    return_dict = True
                )
        val_loss = result.loss.item()
        val_pred = result.logits.detach().cpu().numpy()
        val_pred = np.argmax(val_pred,axis=1).flatten()
        accuracy = accuracy_score(val_labels.numpy(),val_pred)
        precision,recall,f1,_ = precision_recall_fscore_support(val_labels,val_pred)
        print(f'Epoch:{epoch}, val_loss:{val_loss}, accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1:{f1}')
    new_path = r'/home/jupyter/CS4248-Project/trained_models'
    if not os.path.exists(new_path):
        model.save_pretrained(new_path)     

rank = os.environ["LOCAL_RANK"]
train_model(model,rank,val_inputs=val_inputs,val_attention=val_attention,val_labels=val_labels)