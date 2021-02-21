import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class CDAE(nn.Module):
    def __init__(self, device, batch_size, n_users, n_items, k_size=100, corrupt_ratio=0.5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.k_size = k_size
        self.device = device
        self.batch_size = batch_size
        self.corrupt_ratio= corrupt_ratio
        self.user_emb = nn.Embedding(n_users, k_size)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=self.n_items),
            nn.Linear(self.n_items, self.k_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.k_size, self.n_items),
            nn.Sigmoid()
        )
        self.to(self.device)
    def forward(self, user_id, interaction_matrix):
        out = F.dropout(input=interaction_matrix, p=self.corrupt_ratio)
        out = self.encoder(out) + self.user_emb(user_id)
        out = nn.Sigmoid()(out)
        return self.decoder(out)

    def predict(self, eval_pos):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray()).to(self.device)
            pred_result_matrix = np.zeros_like(input_matrix)

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / self.batch_size))
            perm = list(range(num_data))
        
            for batch in range(num_batches):
                if (batch + 1) * self.batch_size >= num_data:
                    batch_idx = perm[batch * self.batch_size:]
                else:
                    batch_idx = perm [batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_matrix = input_matrix[batch_idx]
                batch_idx = torch.LongTensor(batch_idx).to(self.device)
                batch_pred_result_matrix = self.forward(batch_idx, batch_matrix)
                pred_result_matrix[batch_idx] = batch_pred_result_matrix 

        pred_result_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_result_matrix

    def train_one_epoch(self, epoch, dataset, optimizer, criterion, verbose=True):
        self.train()
        loss = 0.0
        step = 0.0
        train_matrix = dataset.train_matrix

        num_to_train = train_matrix.shape[0]
        num_batches = int(np.ceil(num_to_train / self.batch_size))

        perm = np.random.permutation(num_to_train)
        for batch in range(num_batches):
            optimizer.zero_grad()

            if (batch + 1) * self.batch_size >= num_to_train:
                batch_idx = perm[batch * self.batch_size:]
            else:
                batch_idx = perm[batch * self.batch_size: (batch + 1) * self.batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
            batch_idx = torch.LongTensor(batch_idx).to(self.device)

            pred_matrix = self.forward(batch_idx, batch_matrix)

            batch_loss = criterion(pred_matrix, batch_matrix)
            batch_loss.backward()
            optimizer.step()

            step += 1.0
            loss += batch_loss
            if verbose==True and batch % 10 == 0:
                print (f'Epoch {epoch} ({batch * self.batch_size} / {num_to_train}) loss = {batch_loss:.4f}')
        return loss / step