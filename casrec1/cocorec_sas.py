import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import *
import sys
import os
import datetime
import pickle

from new_utils import *





def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
import argparse
import datetime



parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', default=128, type=int) #mat1(a,b)->a
parser.add_argument('--beam_size', type=int, default=10)

parser.add_argument('--config_files',  default='None', type=str)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data_name', default=None, type=str)

parser.add_argument('--dataset',  default='None', type=str)
parser.add_argument('--datasets',  default='retail', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")

parser.add_argument('--gamma', default=.5, type=float)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--incatemaxlen', type=int, default=10)


parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--lr', default=.0001, type=float)
parser.add_argument('--l2_emb', default=0.005, type=float)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--model', type=str, default='casr')


parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--num_blocks', default=2, type=int)


parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--recall', default=[5, 10, 20, 50], type=list)



parser.add_argument("--seed", type=int, default=5,
                     help="Seed for random initialization")


parser.add_argument("-sigma", type=float, default=None,
                     help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--time_bucket', type=int, default=64)
parser.add_argument('--time_span', type=int, default=64)
parser.add_argument('--topkc', default=1, type=int)


args, _ = parser.parse_known_args()

np.random.seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def init_model(model):
    if args.sigma is not None:
        for p in model.parameters():
            if args.sigma != -1 and args.sigma != -2:
                sigma = args.sigma
                p.data.uniform_(-sigma, sigma)
            elif len(list(p.size())) > 1:
                sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                if args.sigma == -1:
                    p.data.uniform_(-sigma, sigma)
                else:
                    p.data.uniform_(0, sigma)

def count_parameters(model):
    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameter_num", parameter_num) 


batch_size = args.batch_size
lr = args.lr





st = datetime.datetime.now()


if args.datasets =='sub_taobao':
    args.dataset = './data/subtb_10_50.csv'
    User = dataall_partition_last(args.dataset)


if args.datasets =='ijcai':
    args.dataset = './data/mergedijcai_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 32
    args.beam_size=10
if args.datasets =='bigtaobao':
    args.dataset = './data/bigtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 32
    args.beam_size = 20

if args.datasets =='midtaobao':
    args.dataset = './data/midtaobao_20_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 64
    args.beam_size = 20

if args.datasets =='retail':
    args.dataset = './data/retail_10_100.csv'
    # args.lr = 0.001
    User = dataall_partition_last(args.dataset)
    args.batch_size = 512

    args.beam_size=5

if args.datasets =='beer':
    args.dataset = './data/beer_10_100.csv'
    # args.lr = 0.001
    User = dataall_partition_last(args.dataset)
    args.batch_size = 512
    args.beam_size=5
    args.patience = 200
if args.datasets =='smalltaobao':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/taobao/mergedmultismalltb_10_40.csv'
    User = dataall_partition_last(args.dataset)

if args.datasets =='taobao':
    args.dataset = './data/subtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    args.beam_size=5
    args.patience = 200

directory = './dataset/'+args.datasets+'/'

if not os.path.exists(directory):
    os.mkdir(directory)  # 如果目录不存在，则创建它
    print(f"'{directory}' created" )
else:
    print(f" '{directory}' exists")

item_res_train, cate_res_train,  item_res_valid, cate_res_valid, item_res_test, cate_res_test,  usernum,  itemnum, catenum= datatimenormalizelast(User)

num_batch = len(item_res_train) // args.batch_size
cc = 0.0
for u in item_res_train:
    cc += len(item_res_train[u])
print('average item sequence length: %.2f' % (cc / len(item_res_train)))



item_res_all = dict()
cate_res_all = dict()

item_res_eval = dict()
cate_res_eval = dict()
for user in range(1,usernum+1):
    item_res_all[user] = item_res_train[user] + item_res_valid[user] + item_res_test[user]
    cate_res_all[user] = cate_res_train[user] + cate_res_valid[user] + cate_res_test[user]

for user in range(1,usernum+1):
    item_res_eval[user] = item_res_train[user] + item_res_valid[user] 
    cate_res_eval[user] = cate_res_train[user] + cate_res_valid[user] 



# time_point = generate_timepointfromtime(time_res_all, args.time_bucket)



et = datetime.datetime.now()
print("duration ", et-st)
dataloader = lastslow_newDataloader()
try:
    print('try')
    with open('./dataset/%s/dataloader_%d_train%d.pickle'%(args.datasets, args.maxlen,  args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    dataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init1')
    dataloader.init(item_res_train, cate_res_train,  usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen)

    with open('./dataset/%s/dataloader_%d_train%d.pickle'%(args.datasets, args.maxlen, args.batch_size),"wb") as f:
        pickle.dump(dataloader.dataset, f)

[usetensors, itemytensors, cateytensors,  itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors, itemyseqtensors, cateyseqtensors] = zip(*dataloader.dataset)

traindataset = mydataset(usetensors, itemytensors, cateytensors, itemseqtensors, cateseqtensors,  padincateitemtensors,  padcatesettensors, itemyseqtensors, cateyseqtensors, args)

evaldataloader = lastslow_newevalDataloader()
try:
    print('try')
    with open('./dataset/%s/dataloader_%d_eval%d.pickle'%(args.datasets, args.maxlen, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    evaldataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init2')
    evaldataloader.init(item_res_eval, cate_res_eval,  item_res_valid, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen)
    with open('./dataset/%s/dataloader_%d_eval%d.pickle'%(args.datasets, args.maxlen, args.batch_size),"wb") as f:
            pickle.dump(evaldataloader.dataset, f)

[usetensors, itemytensors, cateytensors, itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors,itemyseqtensors, cateyseqtensors] = zip(*evaldataloader.dataset)

evaldataset = mydataset(usetensors, list(itemytensors), list(cateytensors), itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors, itemyseqtensors,  cateyseqtensors,  args)

testdataloader = lastslow_newevalDataloader()
try:
    print('try')
    with open('./dataset/%s/dataloader_%d_test%d.pickle'%(args.datasets, args.maxlen, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    testdataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init2')
    testdataloader.init(item_res_all, cate_res_all, item_res_test, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen)
    with open('./dataset/%s/dataloader_%d_test%d.pickle'%(args.datasets, args.maxlen, args.batch_size),"wb") as f:
            pickle.dump(testdataloader.dataset, f)







[usetensors, itemytensors, cateytensors, itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors,itemyseqtensors, cateyseqtensors] = zip(*testdataloader.dataset)

testdataset = mydataset(usetensors, list(itemytensors), list(cateytensors), itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors, itemyseqtensors,  cateyseqtensors,  args)


mydata = DataModule(traindataset, testdataset, testdataset, args)



if args.datasets=='retail':
    args.config_files = './configs/retail.yaml'
    args.beam_size=5
elif args.datasets == 'sub_taobao':
    args.config_files = './configs/sub_taobao.yaml'
elif args.datasets =='taobao' or args.datasets =='bigtaobao' or args.datasets =='midtaobao' :
    args.config_files = './configs/taobao.yaml'
elif args.datasets == 'sub_retail':
    args.config_files = './configs/sub_retail.yaml'
    args.beam_size=5
elif args.datasets == 'ijcai':
    args.config_files = './configs/ijcai.yaml'
elif args.datasets == 'beer':
    args.config_files = './configs/beer.yaml'

parameter_dict = {
        'neg_sampling': None,
        'gpu_id':5
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
parameter_dict['gpu_id'] = args.cuda
import yaml
def _load_config_files(file_list):
    file_config_dict = dict()
    if file_list:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                file_config_dict.update(yaml.load(f.read(), Loader=yaml.FullLoader))
    return file_config_dict


config = _load_config_files([args.config_files])



import json
import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn.init import xavier_uniform_, constant_, normal_

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class cudaSASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(cudaSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.sigmoid = torch.nn.Sigmoid()
        self.emb = args.hidden_units
        self.embedding_size = args.hidden_units
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.beha_emb = torch.nn.Embedding(5, args.hidden_units)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units)

        # define the module feature gating need
        self.w1 = nn.Linear(args.hidden_units, args.hidden_units)
        self.w2 = nn.Linear(args.hidden_units, args.hidden_units)
        self.b = nn.Parameter(torch.zeros(args.hidden_units), requires_grad=True)

        # define the module instance gating need
        self.w3 = nn.Linear(args.hidden_units, 1, bias=False)
        self.w4 = nn.Linear(args.hidden_units, args.maxlen, bias=False)
        self.w5 = nn.Linear(args.hidden_units, args.hidden_units)
        self.w6 = nn.Linear(args.hidden_units, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.feed_forward = PointWiseFeedForward(args.hidden_units, args.dropout_rate)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.dropout = torch.nn.Dropout(p = args.dropout_rate)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def log2feats(self, seqs, mask):
        # seqs = self.item_emb(log_seqs)

        # beha = self.beha_emb(beha_seqs)

        seqs  = seqs

        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = mask
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices) # (U, I, C)

        logits = torch.einsum('bcd,bd -> bc',item_embs, final_feat)
        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return  logits # preds # (U, I)
    
    def catepredict(self, user_ids, log_seqs, all_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(all_indices) # (U, I, C)

        logits = torch.einsum('cd,bd -> bc',item_embs, final_feat)
        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return  logits, final_feat # preds # (U, I)
    
    def head(self, logits, indices):
        if indices is None:
            indices = torch.arange(0, self.item_num+1).to(self.dev)
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('cd,bd -> bc',item_embs, logits)
        else:
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('bcd,bd -> bc',item_embs, logits)
        return logits

    def trainq(self, log_seqs, seq_y, seq_neg):
        if seq_neg is None:
            seq_neg = torch.arange(0,self.item_num+1).to(self.dev)
            item_embs = self.item_emb(seq_neg)
            logits = torch.einsum('bnd, cd->bnc',log_seqs, item_embs)
        else:
            neg_embs = self.item_emb(seq_neg)
            seqy_embs = self.item_emb(seq_y)
            neg_logits = torch.einsum('bnd, cd -> bnc', log_seqs, neg_embs)
            pos_logits = torch.einsum('bnd, bnd->bn', log_seqs, seqy_embs)
            logits = torch.cat((pos_logits.unsqueeze(-1), neg_logits), -1)
        return logits


    def feature_gating(self, seq_item_embedding, user_embedding):
        """

        choose the features that will be sent to the next stage(more important feature, more focus)


        """

        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        # print(user_embedding.shape)
        user_embedding = user_embedding.repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.sigmoid(seq_item_embedding + user_embedding + self.b)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item



    def feature_attention(self, seq_item_embedding, user_embedding):
        """

        choose the features that will be sent to the next stage(more important feature, more focus)

        """

        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.sigmoid(seq_item_embedding + user_embedding + self.b)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item
    
    def instance_attention(self,  user_item, user_embedding, mask, item_seq):
        """

        choose the last click items that will influence the prediction( more important more chance to get attention)
        """

        user_embedding_value = user_item

        user_item = self.w6(user_item)
        # batch_size * seq_len * embedding_size

        user_embedding = self.w5(user_embedding)
        # batch_size * embedding_size
        # print(user_item.shape, user_embedding.shape)
        score = torch.einsum('bld,bhd->blh',user_item, user_embedding.repeat(1, user_item.shape[1],1))



        score += mask
        score = self.dropout(nn.functional.softmax(score, dim=-1))


        # instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1

        output = torch.einsum('blh,bhd->bld',score, user_embedding_value + item_seq)
        # output = torch.mul(score.unsqueeze(2), user_embedding_value)
        # batch_size * seq_len * embedding_size


        # output = torch.div(output.sum(dim=1), instance_score.sum(dim=1).unsqueeze(1))
            # batch_size * embedding_size
        # else:
            # # for max_pooling
            # index = torch.max(instance_score, dim=1)[1]
            # # batch_size * 1
            # output = self.gather_indexes(output, index)
            # # batch_size * seq_len * embedding_size ==>> batch_size * embedding_size
        output = self.feed_forward(output)

        return output


    def instance_gating(self, user_item, user_embedding):
        """

        choose the last click items that will influence the prediction( more important more chance to get attention)
        """

        user_embedding_value = user_item

        user_item = self.w3(user_item)
        # batch_size * seq_len * 1

        user_embedding = self.w4(user_embedding).unsqueeze(2)
        
        # batch_size * seq_len * 1

        instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1
        output = torch.mul(instance_score.unsqueeze(2), user_embedding_value)
        # batch_size * seq_len * embedding_size


        output = torch.div(output.sum(dim=1), instance_score.sum(dim=1).unsqueeze(1))
            # batch_size * embedding_size
        # else:
            # # for max_pooling
            # index = torch.max(instance_score, dim=1)[1]
            # # batch_size * 1
            # output = self.gather_indexes(output, index)
            # # batch_size * seq_len * embedding_size ==>> batch_size * embedding_size

        return output


    def heads(self, logits, indices):
        if indices is None:
            indices = torch.arange(0, self.item_num+1).to(self.dev)
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('cd,bhd -> bhc',item_embs, logits)
        else:
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('bcd,bhd -> bhc',item_embs, logits)
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 1 / self.embedding_size)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
    def losshead(self, logits, indices):
        if indices is None:
            indices = torch.arange(0, self.item_num+1).to(self.dev)
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('cd,bnd -> bnc',item_embs, logits)
        else:
            item_embs = self.item_emb(indices) # (U, I, C)
            logits = torch.einsum('bcd,bnd -> bnc',item_embs, logits)
        return logits


import torch.nn.functional as fn


# -*- coding: utf-8 -*-
# @Time    : 2022/02/22 19:32
# @Author  : Peilin Zhou, Yueqi Xie
# @Email   : zhoupl@pku.edu.cn
r"""
SASRecD
################################################

Reference:
    Yueqi Xie and Peilin Zhou et al. "Decouple Side Information Fusion for Sequential Recommendation"
    Submited to SIGIR 2022.
"""

import torch
from torch import gather, nn


class GatedLongRecSimplest(nn.Module):
    def __init__(self,  input_size, hidden_size=args.hidden_size, output_size=args.hidden_size, embedding_dim=args.hidden_size, cate_input_size=1113, cate_output_size=0, cate_embedding_dim=args.hidden_size, cate_hidden_size=args.hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.25, dropout_input=0.25, use_cuda=False, shared_embedding=True, cate_shared_embedding=True):
        
        # #the parameter of the star transformer
        # self.hidden_size = hidden_size
        # self.embedding_dim = embedding_dim
        # self.cate_embedding_dim = cate_embedding_dim
        # self.dropout_rate = dropout_hidden


        print('gated long rec init')
        super(GatedLongRecSimplest, self).__init__()

        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.itemnum = input_size
        self.catenum = cate_input_size
        print(self.catenum)
        ### long-term encoder and short-term encoder
        self.m_itemNN = ITEMNNSimplest(input_size, hidden_size, output_size, embedding_dim, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, shared_embedding)

        ### gating network
        self.m_cateNN = CATENNSimplest(self.catenum, self.catenum, cate_embedding_dim, cate_hidden_size, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, cate_shared_embedding)

        self.itemembs = self.m_itemNN.look_up
        self.cateembs = self.m_cateNN.m_cate_embedding



        self.fc = nn.Linear(hidden_size*2, hidden_size)

        # if shared_embedding:
        #     message = "share embedding"
        #     self.m_log.addOutput2IO(message)
        #     self.m_ss.params.weight = self.m_itemNN.look_up.weight

        # self = self.to(self.device)

    def forward(self, action_cate_long_batch, cate_long_batch,  actionNum_cate_long_batch, action_short_batch, cate_short_batch,  actionNum_short_batch, y_cate_batch):
        print('gated long rec forward')
        seq_cate_short_input = self.m_cateNN(cate_short_batch, actionNum_short_batch)
        logit_cate_short = self.m_cateNN.m_cate_h2o(seq_cate_short_input)

        seq_cate_input, seq_short_input = self.m_itemNN(action_cate_long_batch, cate_long_batch,  actionNum_cate_long_batch, action_short_batch, actionNum_short_batch, y_cate_batch)

        mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
        fc_output = self.fc(mixture_output)

        return fc_output, logit_cate_short

class ITEMNNSimplest(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0,use_cuda=False, shared_embedding=True):
        print('itemnn init')
        super(ITEMNNSimplest, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.use_cuda = use_cuda
        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        print('input_size:',input_size)
        print('embedding_dim:',embedding_dim)

        self.look_up = nn.Embedding(input_size+1, self.embedding_dim, padding_idx=0)
        # self.look_up = nn.Embedding(4000, self.embedding_dim) damn it

        ### long-term encoder
        self.m_cate_session_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        #output,h_n=gru(input,hidden)
        #D=2 if bidirectional=True otherwise 1
        #If batch_first=True then the input and output tensors are provided as(batch, seq, feature)
        #input=(batch_size, sequence_lenth, input_size=embedding_dim) batch_size,sequence_lenth在输入中引入
        #hidden=(numlayers*bidirectional, batch_size, hidden_size=hidden_size)
        #output=(batchsize, sequence_lenth, hidden_size=hidden_size)
        #h_n=(num_layers, batch_size, hidden_size=hidden_size)

        ### short-term encoder
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)

    def forward(self, action_cate_long_batch, cate_long_batch, actionNum_cate_long_batch, action_short_batch, actionNum_short_batch, y_cate_batch):

        y_cate_batch = y_cate_batch.reshape(-1, 1)

        ### retrieve cate concerning target cate
        y_cate_index = (cate_long_batch == y_cate_batch).float()
        y_cate_index = y_cate_index.unsqueeze(-1)

        ### long-term dependence
        #(batchsize, ,len)

        action_cate_long_batch = action_cate_long_batch.reshape(y_cate_batch.size(0), -1, action_cate_long_batch.size(1))
        action_cate_long_batch_mask = action_cate_long_batch*y_cate_index.long()

        action_cate_long_batch_mask = torch.sum(action_cate_long_batch_mask, dim=1)

        action_cate_input = action_cate_long_batch_mask 
        # print('action_cate_input:',action_cate_input)
        # print('action_cate_input shape:',action_cate_input.shape)
        # print(action_cate_input.min())
        # print(action_cate_input.max())
        action_cate_embedded = self.look_up(action_cate_input)

        action_cate_batch_size = action_cate_embedded.size(0)
        action_cate_hidden = self.init_hidden(action_cate_batch_size, self.hidden_size)

        ### action_cate_embedded: batch_size*action_num_cate*hidden_size
        action_cate_output, action_cate_hidden = self.m_cate_session_gru(action_cate_embedded, action_cate_hidden) # (sequence, B, H)




        # actionNum_cate_long_batch = actionNum_cate_long_batch.reshape(y_cate_batch.size(0),-1)
        # actionNum_cate_long_batch = torch.from_numpy(actionNum_cate_long_batch).to(self.device)

        # ### actionNum_cate_long_batch: batch_size*cate_num
        # actionNum_cate_long_batch = actionNum_cate_long_batch*y_cate_index.squeeze(-1).long()
        # actionNum_cate_long_batch = torch.sum(actionNum_cate_long_batch, dim=1)

        # pad_subseqLen_cate_batch = np.array([i-1 if i > 0 else 0 for i in subseqLen_cate_batch])
        first_dim_index = torch.arange(action_cate_batch_size).to('cuda')
        # second_dim_index = actionNum_cate_long_batch

        ### long-term dependence: seq_cate_input
        ### batch_size*hidden_size
        # seq_cate_input = action_cate_output[first_dim_index, second_dim_index, :]
        seq_cate_input = action_cate_output[first_dim_index, -1, :]
        #####################
        ### short-range actions
        action_short_input = action_short_batch.long()
        action_short_embedded = self.look_up(action_short_input)
        
        short_batch_size = action_short_embedded.size(0) 
        
        action_short_hidden = self.init_hidden(short_batch_size, self.hidden_size)
        action_short_output, action_short_hidden = self.m_short_gru(action_short_embedded, action_short_hidden)


        action_short_output_mask = action_short_output

        # pad_seqLen_batch = [i-1 if i > 0 else 0 for i in seqLen_batch]
        first_dim_index = torch.arange(short_batch_size).to('cuda')
        # second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)

        ### short-term dependence: seq_short_input
        ### batch_size*hidden_size
        # seq_short_input = action_short_output_mask[first_dim_index, second_dim_index, :]
        seq_short_input = action_short_output_mask[first_dim_index, -1, :]
        # output_seq, hidden_seq = self.m_short_gru(input_seq, hidden_seq)

        return seq_cate_input, seq_short_input

    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to('cuda')
        return h0

class CATENNSimplest(nn.Module):
    def __init__(self, cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0, use_cuda=False, cate_shared_embedding=True):
        print('catenn init')
        super(CATENNSimplest, self).__init__()
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden

        self.m_cate_input_size = cate_input_size
        self.m_cate_embedding_dim = cate_embedding_dim
        self.m_cate_hidden_size = cate_hidden_size
        self.m_cate_output_size = cate_output_size

        self.m_cate_embedding = nn.Embedding(self.m_cate_input_size+1, self.m_cate_embedding_dim, padding_idx=0)
        self.m_cate_gru = nn.GRU(self.m_cate_embedding_dim, self.m_cate_hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        #input=(batch_size, sequencelength, embedding_dim)
        #h0=(num_layers, batch_size, hidden_size)
        self.m_cate_h2o = nn.Linear(self.m_cate_hidden_size, self.m_cate_output_size+1)
        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        if cate_shared_embedding:
            print('*******************cate shared embedding: True************************')
            self.m_cate_h2o.weight = self.m_cate_embedding.weight

    def forward(self, cate_short_batch, actionNum_short_batch):
        cate_short_embedded = self.m_cate_embedding(cate_short_batch)
        short_batch_size = cate_short_embedded.size(0)
        cate_short_hidden = self.init_hidden(short_batch_size, self.m_cate_hidden_size)

        print(cate_short_embedded.shape, cate_short_hidden.shape)
        cate_short_output, cate_short_hidden = self.m_cate_gru(cate_short_embedded, cate_short_hidden)



        first_dim_index = torch.arange(short_batch_size).to('cuda')
        # second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)
 
        # seq_cate_short_input = cate_short_output[first_dim_index, second_dim_index, :]
        seq_cate_short_input = cate_short_output[first_dim_index, -1, :]
        return seq_cate_short_input
    
    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to('cuda')
        return h0


import pytorch_lightning as pl
from collections import defaultdict
from new_metric import *

class casr(pl.LightningModule):
    def __init__(self, itemnum, catenum,  args
        ):
        super().__init__()
        self.teachingflag = True
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.m_logsoftmax2 = torch.nn.LogSoftmax(dim=2)

        self.loss = torch.nn.CrossEntropyLoss()
        self.args = args


        self.train_epoch = 0
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.m_teacher_forcing_flag = True
        self.m_logsoftmax = nn.LogSoftmax(dim=1)
        self.m_cate_loss_func = nn.CrossEntropyLoss()
        self.model =  GatedLongRecSimplest(input_size=itemnum, cate_input_size=catenum)
        self.catemodel = cudaSASRec(usernum, catenum, args)
        self.train_epoch = 0
        self.max_cate_epoch = 0

        self.max_cate_recall = 0


    def get_attention_mask(self,item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def training_step(self, batch, batch_idx):
        args = self.args
        # self.train_epoch +=1
        # if self.train_epoch>8:
        #     self.teachingflag = False
        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)

        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        label_numpy = valid_numpy.nonzero()
        # item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49
        # loss = self.itemmodel.calculate_loss(item_seq_batch, item_seq_len, cate_seq_batch, item_y_batch, cate_y_batch)


        pad_incateitem_seq_batch = pad_incateitem_seq_batch.squeeze()
        pad_cateset_seq_batch = pad_cateset_seq_batch.squeeze()

        # self.train_epoch+=1

        # item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, catey_seq_batch = item_y_batch.squeeze(0), cate_y_batch.squeeze(0), beha_y_batch.squeeze(0), item_seq_batch.squeeze(0), cate_seq_batch.squeeze(0), beha_seq_batch.squeeze(0), time_seq_batch.squeeze(0), pad_incateitem_seq_batch.squeeze(0), pad_incatetime_seq_batch.squeeze(0), pad_cateset_seq_batch.squeeze(0), catey_seq_batch.squeeze(0)

        ### negative samples
        # sample_values = self.m_sampler.sample(self.m_nsampled, np.array(item_y_batch).reshape(batch_size))
        # sample_ids, true_freq, sample_freq = sample_values
        # item_idx = []
        # for _ in range(1000):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in item_y_batch: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        # item_test = np.repeat(np.array(item_idx).reshape(1,-1), args.batch_size, 0)


        # all = torch.cat((item_y_batch.reshape(-1,1), torch.from_numpy(item_test).to('cuda')),dim=1)
        # all_embs = self.model.m_itemNN.look_up(all)
        

        item_idx = np.arange(0,itemnum+1)
        item_test1 = torch.from_numpy(item_idx).to('cuda')
        test_embs = self.model.m_itemNN.look_up(item_test1)

        seq_cate_short_input = self.model.m_cateNN(cate_seq_batch, -1)
        # logit_cate_short = self.model.m_cateNN.m_cate_h2o(seq_cate_short_input)


     
        mask = (item_seq_batch==0)


        cate_logits = self.catemodel .log2feats(self.catemodel.item_emb(cate_seq_batch), mask)  
        cate_input = cate_logits[torch.arange(cate_logits.size(0)), -1, :] 
        logit_cate_short = self.catemodel.head(cate_input, None).squeeze()

        pred_item_prob = None
        pred_item_prob1 = None

        if self.m_teacher_forcing_flag:
            # seq_cate_input, seq_short_input = self.model.m_itemNN(torch.from_numpy(np.array(incateitem_seq_batch).reshape(np.array(incateitem_seq_batch).shape[0]*np.array(incateitem_seq_batch).shape[1],-1)).to('cuda'), torch.from_numpy(np.array(cateset_seq_batch)).to('cuda'),  0, torch.from_numpy(np.array(item_seq_batch)).to('cuda'),  0, torch.from_numpy(np.array(cate_y_batch)).to('cuda'))

            seq_cate_input, seq_short_input = self.model.m_itemNN(pad_incateitem_seq_batch.reshape(pad_incateitem_seq_batch.shape[0]*pad_incateitem_seq_batch.shape[1],-1), pad_cateset_seq_batch,  0, item_seq_batch,  0, cate_y_batch)
            #seq_short_input.shape seq_cate_input.shape  [100,16] embedding dim or hidden dim
            #mixture_output.shape [100,32]

            mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
            #fc_output.shape [100,16]
            fc_output = self.model.fc(mixture_output)


            test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)
            test_prob_batch = self.m_logsoftmax(test_logit_batch)
            pred_item_prob1 = test_prob_batch 
        
        else:
            log_prob_cate_short = self.m_logsoftmax(logit_cate_short)
            log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)

            pred_cate_index = pred_cate_index.detach()
            log_prob_cate_short = log_prob_cate_short

            for beam_index in range(args.beam_size):
                pred_cate_beam = pred_cate_index[:, beam_index]
                prob_cate_beam = log_prob_cate_short[:, beam_index]
                #only in itemNN x_long_cate_action is used
                seq_cate_input, seq_short_input = self.model.m_itemNN(pad_incateitem_seq_batch.reshape(pad_incateitem_seq_batch.shape[0]*pad_incateitem_seq_batch.shape[1],-1), pad_cateset_seq_batch,  0, item_seq_batch,  0, pred_cate_beam)

                mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
                fc_output = self.model.fc(mixture_output)
                #sampled_logit_batch [256,10001]  sampled_target_batch torch.zeros(256)
                # sampled_logit_batch, sampled_target_batch = self.model.m_ss(fc_output, torch.from_numpy(np.array(item_y_batch)).to('cuda'),  sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match)
            

                test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)

                test_prob_batch = self.m_logsoftmax(test_logit_batch)
                if pred_item_prob1 is None:
                    pred_item_prob1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob1 = pred_item_prob1.unsqueeze(-1)
                else:
                    pred_item_prob_beam1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob_beam1 = pred_item_prob_beam1.unsqueeze(-1)
                    pred_item_prob1 = torch.cat((pred_item_prob1, pred_item_prob_beam1), dim=-1)
            pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)


        #在teaching flag true时，sampled_target_batch 全为0
        #因为真值为第一个，因此sampled_target_batch全为0，也就是第一个
        # 输入为prob时，与输入logit，损失函数不同
        # item_loss = -torch.mean(pred_item_prob[:, 0])
        valid_labels = torch.zeros((pred_item_prob1.shape[0])).long().to('cuda')
        item_loss = self.m_cate_loss_func(pred_item_prob1 , item_y_batch)
        cate_loss = self.m_cate_loss_func(logit_cate_short, cate_y_batch)
        loss = (1-args.gamma)*item_loss + args.gamma*cate_loss 
        
        # loss = self.loss(pred_item_prob, valid_labels)
        # loss = loss.unsqueeze(0)
        # recall_5, mrr_5 = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(item_y_batch).to('cuda'), k=5)
        cate_recall, cate_mrr, cate_ndcg = evaluate_part(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=10)

        return {'loss':loss, 'cate_recall': cate_recall}


    def training_epoch_end(self, training_step_outputs):
        self.train_epoch+=1
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)
        

        cate_recall = torch.stack([o['cate_recall'] for o in training_step_outputs], 0).mean()
        if cate_recall > self.max_cate_recall:
            self.max_cate_recall = cate_recall
            self.max_cate_epoch = self.train_epoch

        elif cate_recall <self.max_cate_recall:
            self.m_teacher_forcing_flag = False
        # keys = training_step_outputs[0]['metrics'].keys()
        # metric =  defaultdict(list)
        # for o in training_step_outputs:
        #     for k in keys:
        #         metric[k].append(o['metrics'][k])

        # for k in keys:
        #     self.log(f'Train:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        args = self.args
        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        pad_incateitem_seq_batch = pad_incateitem_seq_batch.squeeze(0)
        pad_cateset_seq_batch = pad_cateset_seq_batch.squeeze(0)




        item_idx = np.arange(0, itemnum+1)
        item_test = torch.from_numpy(item_idx).to('cuda')
        test_embs = self.model.m_itemNN.look_up(item_test)


        pred_item_prob1 = None
             
        mask = (item_seq_batch==0)
        cate_logits = self.catemodel .log2feats(self.catemodel.item_emb(cate_seq_batch), mask)  
        cate_input = cate_logits[torch.arange(cate_logits.size(0)), -1, :] 
        logit_cate_short = self.catemodel.head(cate_input, None).squeeze()



        log_prob_cate_short = self.m_logsoftmax(logit_cate_short)
        log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)

        pred_cate_index = pred_cate_index.detach()
        log_prob_cate_short = log_prob_cate_short


        mask = (item_seq_batch>0)

        for beam_index in range(args.beam_size):
            pred_cate_beam = pred_cate_index[:, beam_index]
            prob_cate_beam = log_prob_cate_short[:, beam_index]
            #only in itemNN x_long_cate_action is used
            seq_cate_input, seq_short_input = self.model.m_itemNN(pad_incateitem_seq_batch.reshape(pad_incateitem_seq_batch.shape[0]*pad_incateitem_seq_batch.shape[1],-1), pad_cateset_seq_batch,  0, item_seq_batch,  0, pred_cate_beam)

            # star_output = self.starmodel(self.model.m_itemNN.look_up(item_seq_batch), self.model.m_cateNN.m_cate_embedding(cate_seq_batch), self.model.m_cateNN.m_cate_embedding(pred_cate_beam), mask)

            mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
            fc_output = self.model.fc(mixture_output)

        

            test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)
            test_prob_batch = self.m_logsoftmax(test_logit_batch)

            # startest_logit_batch = torch.einsum('bd, cd->bc',star_output, test_embs)
            # test_prob_batch = self.m_logsoftmax(test_logit_batch+ startest_logit_batch)

            if pred_item_prob1 is None:
                pred_item_prob1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob1 = pred_item_prob1.unsqueeze(-1)
            else:
                pred_item_prob_beam1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob_beam1 = pred_item_prob_beam1.unsqueeze(-1)
                pred_item_prob1 = torch.cat((pred_item_prob1, pred_item_prob_beam1), dim=-1)
        
    
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)



        valid_labels = torch.zeros((pred_item_prob1.shape[0])).long().to('cuda')


        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)

        #cate 只有十多个
        # catemetrics = evaluate(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)
        cate_recall, cate_mrr, cate_ndcg = evaluate_part(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=5)




        return {'metrics':metrics,  'validcate_recall':cate_recall}

    


    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in validation_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])

        for k in keys:
            self.log(f'Val:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

        validcate_recall = torch.stack([o['validcate_recall'] for o in validation_step_outputs], 0).mean()
        self.log('validcate_recall', validcate_recall, on_epoch=True)

        
    def test_step(self, batch, batch_idx):
        args = self.args
        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        pad_incateitem_seq_batch = pad_incateitem_seq_batch.squeeze(0)
        pad_cateset_seq_batch = pad_cateset_seq_batch.squeeze(0)
        # item_seq_len = torch.arange(0, 50).unsqueeze(0).repeat(item_seq_batch.shape[0],1).to(item_seq_batch.device)

        # item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49


        # logit = self.itemmodel.full_sort_predict(item_seq_batch, item_seq_len, cate_seq_batch)
    
        # valid_labels = torch.ones((logit.shape[0])).long().to(args.device)


        # metrics = evaluate(logit, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)


        # cate_recall_5, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), k=5)



        item_idx = np.arange(0, itemnum+1)
        item_test = torch.from_numpy(item_idx).to('cuda')
        test_embs = self.model.m_itemNN.look_up(item_test)



        # seq_cate_short_input = self.model.m_cateNN(cate_seq_batch, -1)

        # logit_cate_short = self.model.m_cateNN.m_cate_h2o(seq_cate_short_input)
     
        mask = (item_seq_batch==0)

        
        cate_logits = self.catemodel .log2feats(self.catemodel.item_emb(cate_seq_batch), mask)  
        cate_input = cate_logits[torch.arange(cate_logits.size(0)), -1, :] 
        logit_cate_short = self.catemodel.head(cate_input, None).squeeze()


        pred_item_prob1 = None

        log_prob_cate_short = self.m_logsoftmax(logit_cate_short)
        log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)

        pred_cate_index = pred_cate_index.detach()
        log_prob_cate_short = log_prob_cate_short


        mask = (item_seq_batch>0)

        for beam_index in range(args.beam_size):
            pred_cate_beam = pred_cate_index[:, beam_index]
            prob_cate_beam = log_prob_cate_short[:, beam_index]
            #only in itemNN x_long_cate_action is used
            seq_cate_input, seq_short_input = self.model.m_itemNN(pad_incateitem_seq_batch.reshape(pad_incateitem_seq_batch.shape[0]*pad_incateitem_seq_batch.shape[1],-1), pad_cateset_seq_batch,  0, item_seq_batch,  0, pred_cate_beam)

            # star_output = self.starmodel(self.model.m_itemNN.look_up(item_seq_batch), self.model.m_cateNN.m_cate_embedding(cate_seq_batch), self.model.m_cateNN.m_cate_embedding(pred_cate_beam), mask)

            mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
            fc_output = self.model.fc(mixture_output)

        

            test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)
            test_prob_batch = self.m_logsoftmax(test_logit_batch)

            # startest_logit_batch = torch.einsum('bd, cd->bc',star_output, test_embs)
            # test_prob_batch = self.m_logsoftmax(test_logit_batch+ startest_logit_batch)

            if pred_item_prob1 is None:
                pred_item_prob1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob1 = pred_item_prob1.unsqueeze(-1)
            else:
                pred_item_prob_beam1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob_beam1 = pred_item_prob_beam1.unsqueeze(-1)
                pred_item_prob1 = torch.cat((pred_item_prob1, pred_item_prob_beam1), dim=-1)
        
    
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)



        valid_labels = torch.zeros((pred_item_prob1.shape[0])).long().to('cuda')


        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)

        #cate 只有十多个
        # catemetrics = evaluate(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)
        cate_recall, cate_mrr, cate_ndcg = evaluate_part(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=10)




        return {'metrics':metrics,  'testcate_recall':cate_recall}
    


    def test_epoch_end(self, test_step_outputs):
        keys = test_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in test_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])

        for k in keys:
            self.log(f'Test:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

        testcate_recall = torch.stack([o['testcate_recall'] for o in test_step_outputs], 0).mean()
        self.log('testcate_recall', testcate_recall, on_epoch=True)

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        return optimizer


# model = SASRecD(config, usernum+1, itemnum+1, catenum+1)

# # itemmodel = cudaSASRec(usernum, itemnum, args)

# catemodel = cudaSASRec(usernum, catenum, args )

recmodel = casr(itemnum, catenum, args)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@20', mode='max')

trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/cocorec_sas/recall{}/lr{}'.format(args.datasets , args.recall, args.lr))

##test test
trainer.fit(recmodel, mydata)

print(recmodel.max_cate_epoch, recmodel.max_cate_recall)
trainer.test(recmodel, mydata, ckpt_path='best')