from collections import OrderedDict, defaultdict
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.ScoreAverage import ScoreAverage

class Evaluator:
    def __init__(self, model, k, target, eval_pos):
        self.k = k
        self.model = model
        self.target = target
        self.eval_pos = eval_pos
    def evaluate(self):
        self.model.eval()
        # predict 
        pred_matrix = self.model.predict(self.eval_pos)
        # top_k
        topk = self.get_topk_recommend(pred_matrix, self.k)
        # print (f'topk {topk} idx {idx} shape {topk.shape}')
        # evaluate
        scores = self.prec_recall(topk, self.target)
        score_dict = OrderedDict()
        for metric in scores:
            score_dict[f'{metric}@100'] = f'{scores[metric][self.k].mean:.6f}'
        return score_dict
    def get_topk_recommend(self, preds, k):
        preds = torch.Tensor(preds)
        topk = torch.topk(preds, k, dim=1)
        return topk.indices
    def prec_recall(self, topk, target):
        prec = {100: ScoreAverage()}
        recall = {100: ScoreAverage()}
        scores = {
            'Prec': prec,
            'Recall': recall,
        }
        for idx, user in enumerate(target):
            pred_u = topk[user]
            target_u = target[user]
            num_target_items = len(target_u)
            hits_k = [(i + 1, item) for i, item in enumerate(pred_u) if item in target_u]
            num_hits = len(hits_k)
            prec_k = num_hits / self.k
            recall_k = num_hits / min(num_target_items, self.k)
            scores['Prec'][self.k].update(prec_k)
            scores['Recall'][self.k].update(recall_k)

        return scores