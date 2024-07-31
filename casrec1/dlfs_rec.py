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
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--incatemaxlen', type=int, default=10)


parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--lr', default=.001, type=float)
parser.add_argument('--l2_emb', default=0.005, type=float)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--model', type=str, default='casr')


parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--num_blocks', default=2, type=int)


parser.add_argument('--optimizer_type', default='Adagrad', type=str)
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
    args.batch_size = 128
    args.beam_size=50

if args.datasets =='bigtaobao':
    args.dataset = './data/bigtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 32
    args.beam_size = 20

# if args.datasets =='midtaobao':
#     args.dataset = './data/midtaobao_20_100.csv'
#     User = dataall_partition_last(args.dataset)
#     args.batch_size = 64
#     args.beam_size = 20

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
    args.beam_size=10
if args.datasets =='smalltaobao':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/taobao/mergedmultismalltb_10_40.csv'
    User = dataall_partition_last(args.dataset)

if args.datasets =='taobao':
    args.dataset = './data/subtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 512
    args.beam_size=50

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


mydata = DataModule(traindataset, evaldataset, testdataset, args)



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

import torch.nn.functional as fn
class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



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


import copy


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class DistFilterLayer(nn.Module):
    def __init__(self, args):
        super(DistFilterLayer, self).__init__()
        self.mean_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.cov_complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_mean_tensor, input_cov_tensor):
        batch, seq_len, hidden = input_mean_tensor.shape

        mean_x = torch.fft.rfft(input_mean_tensor, dim=1, norm='ortho')
        mean_weight = torch.view_as_complex(self.mean_complex_weight)
        mean_x = mean_x * mean_weight
        mean_sequence_emb_fft = torch.fft.irfft(mean_x, n=seq_len, dim=1, norm='ortho')
        mean_hidden_states = self.out_dropout(mean_sequence_emb_fft)
        mean_hidden_states = self.layer_norm(mean_hidden_states + input_mean_tensor)

        cov_x = torch.fft.rfft(input_cov_tensor, dim=1, norm='ortho')
        cov_weight = torch.view_as_complex(self.cov_complex_weight)
        cov_x = cov_x * cov_weight
        cov_sequence_emb_fft = torch.fft.irfft(cov_x, n=seq_len, dim=1, norm='ortho')
        cov_hidden_states = self.out_dropout(cov_sequence_emb_fft)
        cov_hidden_states = self.layer_norm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filter_layer = DistFilterLayer(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states):
        mean_filter_output, cov_filter_output = self.filter_layer(mean_hidden_states, cov_hidden_states)
        return mean_filter_output, self.activation_func(cov_filter_output) + 1


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            mean_hidden_states, cov_hidden_states = layer_module(mean_hidden_states, cov_hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states])
        return all_encoder_layers


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights

def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1 = mean1.unsqueeze(dim=1)
    cov1 = cov1.unsqueeze(dim=1)
    mean1_2 = torch.sum(mean1 ** 2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2 ** 2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(1, 2)) + mean1_2 + mean2_2.transpose(1, 2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)

    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)),
                                torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(1, 2)) + cov1_2 + cov2_2.transpose(
        1, 2)

    return ret + cov_ret


class DLFSRecModel(nn.Module):
    def __init__(self, args):
        super(DLFSRecModel, self).__init__()
        self.args = args
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.side_mean_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)
        self.side_cov_dense = nn.Linear(args.feature_size, args.attribute_hidden_size)

        if args.fusion_type == 'concat':
            self.mean_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = nn.Linear(args.attribute_hidden_size + args.hidden_size, args.hidden_size)

        elif args.fusion_type == 'gate':
            self.mean_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)
            self.cov_fusion_layer = VanillaAttention(args.hidden_size, args.hidden_size)

        self.mean_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.cov_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        self.elu = torch.nn.ELU()

        self.apply(self.init_weights)

    def forward(self, input_ids, input_context):

        mean_id_emb = self.item_mean_embeddings(input_ids)
        cov_id_emb = self.item_cov_embeddings(input_ids)

        input_attrs = self.args.items_feature[input_ids]
        mean_side_dense = self.side_mean_dense(torch.cat((input_context, input_attrs), dim=2))
        cov_side_dense = self.side_cov_dense(torch.cat((input_context, input_attrs), dim=2))

        if self.args.fusion_type == 'concat':
            mean_sequence_emb = self.mean_fusion_layer(torch.cat((mean_id_emb, mean_side_dense), dim=2))
            cov_sequence_emb = self.cov_fusion_layer(torch.cat((cov_id_emb, cov_side_dense), dim=2))
        elif self.args.fusion_type == 'gate':
            mean_concat = torch.cat(
                [mean_id_emb.unsqueeze(-2), mean_side_dense.unsqueeze(-2)], dim=-2)
            mean_sequence_emb, _ = self.mean_fusion_layer(mean_concat)
            cov_concat = torch.cat(
                [cov_id_emb.unsqueeze(-2), cov_side_dense.unsqueeze(-2)], dim=-2)
            cov_sequence_emb, _ = self.cov_fusion_layer(cov_concat)
        else:
            mean_sequence_emb = mean_id_emb + mean_side_dense
            cov_sequence_emb = cov_id_emb + cov_side_dense

        mask = (input_ids > 0).long().unsqueeze(-1).expand_as(mean_sequence_emb)
        mean_sequence_emb = mean_sequence_emb * mask
        cov_sequence_emb = cov_sequence_emb * mask

        mean_sequence_emb = self.dropout(self.mean_layer_norm(mean_sequence_emb))
        cov_sequence_emb = self.elu(self.dropout(self.cov_layer_norm(cov_sequence_emb))) + 1

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                output_all_encoded_layers=True)
        sequence_mean_output, sequence_cov_output = item_encoded_layers[-1]

        return sequence_mean_output, sequence_cov_output

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, pos_cxt, neg_ids, neg_cxt):

        pos_mean_emb = self.item_mean_embeddings(pos_ids)
        pos_cov_emb = self.item_cov_embeddings(pos_ids)
        neg_mean_emb = self.item_mean_embeddings(neg_ids)
        neg_cov_emb = self.item_cov_embeddings(neg_ids)

        pos_attrs = self.args.items_feature[pos_ids]
        neg_attrs = self.args.items_feature[neg_ids]

        if self.args.side_info_fused:

            pos_mean_side_dense = self.side_mean_dense(torch.cat((pos_cxt, pos_attrs), dim=1))
            pos_cov_side_dense = self.side_cov_dense(torch.cat((pos_cxt, pos_attrs), dim=1))
            neg_mean_side_dense = self.side_mean_dense(torch.cat((neg_cxt, neg_attrs), dim=1))
            neg_cov_side_dense = self.side_cov_dense(torch.cat((neg_cxt, neg_attrs), dim=1))

            if self.args.fusion_type == 'concat':
                pos_mean_emb = self.mean_fusion_layer(torch.cat((pos_mean_emb, pos_mean_side_dense), dim=1))
                pos_cov_emb = self.cov_fusion_layer(torch.cat((pos_cov_emb, pos_cov_side_dense), dim=1))
                neg_mean_emb = self.mean_fusion_layer(torch.cat((neg_mean_emb, neg_mean_side_dense), dim=1))
                neg_cov_emb = self.cov_fusion_layer(torch.cat((neg_cov_emb, neg_cov_side_dense), dim=1))

            elif self.args.fusion_type == 'gate':
                pos_mean_concat = torch.cat(
                    [pos_mean_emb.unsqueeze(-2), pos_mean_side_dense.unsqueeze(-2)], dim=-2)
                pos_mean_emb, _ = self.mean_fusion_layer(pos_mean_concat)
                pos_cov_concat = torch.cat(
                    [pos_cov_emb.unsqueeze(-2), pos_cov_side_dense.unsqueeze(-2)], dim=-2)
                pos_cov_emb, _ = self.cov_fusion_layer(pos_cov_concat)

                neg_mean_concat = torch.cat(
                    [neg_mean_emb.unsqueeze(-2), neg_mean_side_dense.unsqueeze(-2)], dim=-2)
                neg_mean_emb, _ = self.mean_fusion_layer(neg_mean_concat)
                neg_cov_concat = torch.cat(
                    [neg_cov_emb.unsqueeze(-2), neg_cov_side_dense.unsqueeze(-2)], dim=-2)
                neg_cov_emb, _ = self.cov_fusion_layer(neg_cov_concat)

            else:
                pos_mean_emb = pos_mean_emb + pos_mean_side_dense
                pos_cov_emb = pos_cov_emb + pos_cov_side_dense
                neg_mean_emb = neg_mean_emb + neg_mean_side_dense
                neg_cov_emb = neg_cov_emb + neg_cov_side_dense

        pos_cov_emb = self.elu(pos_cov_emb) + 1
        neg_cov_emb = self.elu(neg_cov_emb) + 1

        seq_mean_emb = seq_mean_out[:, -1, :]
        seq_cov_emb = seq_cov_out[:, -1, :]

        pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean_emb, pos_cov_emb)
        neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean_emb, neg_cov_emb)
        pos_vs_neg = wasserstein_distance(pos_mean_emb, pos_cov_emb, neg_mean_emb, neg_cov_emb)

        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits) + 1e-24) * istarget) / torch.sum(istarget)

        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(
            istarget)
        auc = torch.sum(((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict(self, seq_mean_out, seq_cov_out, test_ids, test_cxt):

        test_mean_emb = self.item_mean_embeddings(test_ids)
        test_cov_emb = self.item_cov_embeddings(test_ids)

        test_attrs = self.args.items_feature[test_ids]
        if self.args.side_info_fused:

            test_mean_side_dense = self.side_mean_dense(torch.cat((test_cxt, test_attrs), dim=2))
            test_cov_side_dense = self.side_cov_dense(torch.cat((test_cxt, test_attrs), dim=2))

            if self.args.fusion_type == 'concat':
                test_mean_emb = self.mean_fusion_layer(torch.cat((test_mean_emb, test_mean_side_dense), dim=2))
                test_cov_emb = self.cov_fusion_layer(torch.cat((test_cov_emb, test_cov_side_dense), dim=2))
            elif self.args.fusion_type == 'gate':
                test_mean_concat = torch.cat(
                    [test_mean_emb.unsqueeze(-2), test_mean_side_dense.unsqueeze(-2)], dim=-2)
                test_mean_emb, _ = self.mean_fusion_layer(test_mean_concat)
                test_cov_concat = torch.cat(
                    [test_cov_emb.unsqueeze(-2), test_cov_side_dense.unsqueeze(-2)], dim=-2)
                test_cov_emb, _ = self.cov_fusion_layer(test_cov_concat)
            else:
                test_mean_emb = test_mean_emb + test_mean_side_dense
                test_cov_emb = test_cov_emb + test_cov_side_dense

        test_item_mean_emb = test_mean_emb
        test_item_cov_emb = self.elu(test_cov_emb) + 1
        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)


class SASRecD(torch.nn.Module):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, usernum, itemnum, catenum):
        super(SASRecD, self).__init__()
        # self.USER_ID = config['USER_ID_FIELD']
        # self.ITEM_ID = config['ITEM_ID_FIELD']
        # self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = itemnum
        self.n_cates = catenum
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        # self.layer_norm_eps = config['layer_norm_eps']
        self.layer_norm_eps = 1e-12

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = 'cuda'
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['fusion_type']

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.fea_embedding = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.n_attributes = catenum
        # for attribute in self.selected_features:
        #     self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.hidden_size),
                                             nn.BatchNorm1d(num_features=self.hidden_size),
                                             nn.ReLU(),
                                             # final logits
                                             nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.n_attributes)
                                             )
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList(
                [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes))
                 for _ in self.selected_features])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
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
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len, cate_seq):
        item_emb = self.item_embedding(item_seq)
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)


        feature_emb = self.fea_embedding(cate_seq)
        # feature_table = []
        # for feature_embed_layer in self.feature_embed_layer_list:
        #     sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
        #     sparse_embedding = sparse_embedding['item']
        #     dense_embedding = dense_embedding['item']
        # # concat the sparse embedding and float embedding
        #     if sparse_embedding is not None:
        #         feature_table.append(sparse_embedding)
        #     if dense_embedding is not None:
        #         feature_table.append(dense_embedding)

        # feature_emb = feature_table
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb,[feature_emb],position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, cate_seq, pos_items, pos_cate):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, cate_seq)
        # pos_items = interaction[self.POS_ITEM_ID]
        if True:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            if self.attribute_predictor!='' and self.attribute_predictor!='not':
                loss_dic = {'item_loss':loss}
                attribute_loss_sum = 0
                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output)
                    attribute_labels = pos_cate
                    attribute_labels1 = attribute_labels.clone().detach()#zz crete
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes)

                    if len(attribute_labels.shape) > 2:
                        attribute_labels = attribute_labels.sum(dim=1)
                    attribute_labels = attribute_labels.float()
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                    attribute_loss = torch.mean(attribute_loss[:, 1:])
                    loss_dic[self.selected_features[i]] = attribute_loss
                if self.num_feature_field == 1:
                    total_loss = loss + self.lamdas[0] * attribute_loss
                    # print('total_loss:{}\titem_loss:{}\tattribute_{}_loss:{}'.format(total_loss, loss,self.selected_features[0],attribute_loss))
                else:
                    for i,attribute in enumerate(self.selected_features):
                        attribute_loss_sum += self.lamdas[i] * loss_dic[attribute]
                    total_loss = loss + attribute_loss_sum
                    loss_dic['total_loss'] = total_loss
                    # s = ''
                    # for key,value in loss_dic.items():
                    #     s += '{}_{:.4f}\t'.format(key,value.item())
                    # print(s)
            else:
                total_loss = loss
            return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, item_seq, item_seq_len, cate_seq):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, cate_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores

import pytorch_lightning as pl
from collections import defaultdict
from new_metric import *

class casr(pl.LightningModule):
    def __init__(self, itemmodel, catemodel,  args
        ):
        super().__init__()
        self.teachingflag = True
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.m_logsoftmax2 = torch.nn.LogSoftmax(dim=2)

        self.loss = torch.nn.CrossEntropyLoss()
        self.args = args
        self.itemmodel = itemmodel
        self.catemodel = catemodel

        self.train_epoch = 0
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

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
        self.train_epoch +=1
        if self.train_epoch>8:
            self.teachingflag = False
        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)

        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        label_numpy = valid_numpy.nonzero()
        item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49




        loss = self.itemmodel.calculate_loss(item_seq_batch, item_seq_len, cate_seq_batch, item_y_batch, cate_y_batch)



        # for param in self.itemmodel.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        return {'loss':loss}

    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)
        
        # keys = training_step_outputs[0]['metrics'].keys()
        # metric =  defaultdict(list)
        # for o in training_step_outputs:
        #     for k in keys:
        #         metric[k].append(o['metrics'][k])

        # for k in keys:
        #     self.log(f'Train:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

    def validation_step(self, batch, batch_idx):

        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)

        # item_seq_len = torch.arange(0, 50).unsqueeze(0).repeat(item_seq_batch.shape[0],1).to(item_seq_batch.device)

        item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49


        logit = self.itemmodel.full_sort_predict(item_seq_batch, item_seq_len, cate_seq_batch)
    
        valid_labels = torch.ones((logit.shape[0])).long().to(args.device)


        metrics = evaluate(logit, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)


        # cate_recall_5, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), k=5)




        return {'metrics':metrics}

    


    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in validation_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])

        for k in keys:
            self.log(f'Val:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

        # validcate_recall_5 = torch.stack([o['validcate_recall_5'] for o in validation_step_outputs], 0).mean()
        # self.log('validcate_recall_5', validcate_recall_5, on_epoch=True)

        
    def test_step(self, batch, batch_idx):

        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        # item_seq_len = torch.arange(0, 50).unsqueeze(0).repeat(item_seq_batch.shape[0],1).to(item_seq_batch.device)
        item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49



        logit = self.itemmodel.full_sort_predict(item_seq_batch, item_seq_len, cate_seq_batch)

    
        valid_labels = torch.ones((logit.shape[0])).long().to(args.device)


        metrics = evaluate(logit, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)



        return {'metrics':metrics}

    


    def test_epoch_end(self, test_step_outputs):
        keys = test_step_outputs[0]['metrics'].keys()
        metric =  defaultdict(list)
        for o in test_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])

        for k in keys:
            self.log(f'Test:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)

        # testcate_recall_5 = torch.stack([o['testcate_recall_5'] for o in test_step_outputs], 0).mean()
        # self.log('testcate_recall_5', testcate_recall_5, on_epoch=True)

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        return optimizer


model = SASRecD(config, usernum+1, itemnum+1, catenum+1)

# itemmodel = cudaSASRec(usernum, itemnum, args)

catemodel = cudaSASRec(usernum, catenum, args )

recmodel = casr(model, catemodel, args)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=20, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@20', mode='max')

trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/difsr/recall{}/lr{}'.format(args.datasets , args.recall, args.lr))

##test test
trainer.fit(recmodel, mydata)

trainer.test(recmodel, mydata, ckpt_path='best')