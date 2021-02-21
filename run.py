from collections import OrderedDict, defaultdict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from Data import data_preprocess
from models.CDAE import CDAE
from utils.Evaluator import Evaluator
# Settings
path_1m = 'data/movielens_1m/ratings.dat'
n_users, n_items = (6040, 3706)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
# Dataset & DataLoader
class ImplicitMovielensDataset(Dataset):
    def __init__(self, path, train_ratio = 0.8):
        super().__init__()
        self.train_ratio = train_ratio
        self.path = path
        self.train_matrix, self.test_matrix = data_preprocess(self.path, train_ratio=self.train_ratio)

    def sparse_to_dict(self, sparse_matrix):
        ret = defaultdict(list)
        n_users = sparse_matrix.shape[0]

        for u in range(n_users):
            items_per_u = sparse_matrix.indices[sparse_matrix.indptr[u]:sparse_matrix.indptr[u+1]]
            ret[u] = items_per_u.tolist()
        return ret

    def eval_data(self):
        return self.train_matrix, self.sparse_to_dict(self.test_matrix)
    def __len__(self):
        return self.train_matrix.shape[0]
    def __getitem__(self, index: int):
        return self.train_matrix.toarray()[index], index
        

dataset = ImplicitMovielensDataset(path=path_1m)
eval_pos, eval_target = dataset.eval_data()

# Model
model = CDAE(device=device, batch_size=batch_size, n_users=n_users, n_items=3706, k_size=100, corrupt_ratio=0.5) # train 2965 test 741
# Criterion & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train & Top_k_recommend & Evaluate

evaluator = Evaluator(model=model, k=100, target=eval_target, eval_pos=eval_pos)

for epoch in range(20):
    model.train_one_epoch(epoch, dataset, optimizer, criterion, verbose=True)
    scores = evaluator.evaluate()
    print (scores)
        