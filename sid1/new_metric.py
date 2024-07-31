import torch
import numpy as np
def get_recall(indices, targets, mask):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
        
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()
    
    recall = float(hits.size(0)) / float( mask.int().sum() )

    return torch.tensor(recall)


def get_mrr(indices, targets, mask):

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices)
    hits *= mask.view(-1, 1).expand_as(indices)
    
    hits = hits.nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / float( mask.int().sum() )
    return mrr

def get_ndcg(indices, targets, mask):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices)
    hits *= mask.view(-1, 1).expand_as(indices)
    
    hits = hits.nonzero()
    ranks = hits[:, -1] + 2
    ranks = ranks.float()
    rranks = torch.reciprocal(torch.log2(ranks))
    ndcg = torch.sum(rranks).data / float( mask.int().sum() )
    return ndcg


def evaluate(logits, targets, mask, recalls=[5, 20]):
    metrics = {}
    for k in recalls:

        _, indices = torch.topk(logits, k, -1)# indices be the top k item indices

        metrics['Recall@%d' % k] = get_recall(indices, targets, mask)
        metrics['mrr@%d' % k] = get_mrr(indices, targets, mask)
        metrics['ndcg@%d' % k] = get_ndcg(indices, targets, mask)
    return metrics


def evaluate_part(logits, targets, mask, k=5):

    _, indices = torch.topk(logits, k, -1)# indices be the top k item indices

   
    recall = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)
    ndcg = get_ndcg(indices, targets, mask)
    return recall, mrr, ndcg


def newevaluate(indices, targets, mask):
    recall = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)
    ndcg = get_ndcg(indices, targets, mask)
    return recall, mrr, ndcg



