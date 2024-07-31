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

parser.add_argument('--pretrain', default=200, type=int)#finetune
parser.add_argument('--recall', default=[5, 10, 20, 50], type=list)



parser.add_argument("--seed", type=int, default=5,
                     help="Seed for random initialization")


parser.add_argument("-sigma", type=float, default=None,
                     help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--train_stage', default='pretrain', type=str)#finetune
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
    args.batch_size = 128
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



from torch.nn import functional as F
import random

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



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


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

import copy
class S3Rec(torch.nn.Module):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config):
        super(S3Rec, self).__init__()

        # load parameters info
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']

###sasrecd
        self.layer_norm_eps = 1e-12
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = args.device
        self.num_feature_field = len(config['selected_features'])

####
        # self.FEATURE_FIELD = config['item_attribute']
        # self.FEATURE_LIST = self.FEATURE_FIELD + config['LIST_SUFFIX']
        self.train_stage = args.train_stage  # pretrain or finetune
        self.pre_model_path = '' # We need this for finetune
        self.mask_ratio = 0.2
        self.aap_weight = 0.2
        self.mip_weight = 1.0
        self.map_weight = 1.0
        self.sp_weight = 0.5
# train_stage: 'pretrain'  # pretrain or finetune
# # pre_model_path:''  # We need this for finetune
# mask_ratio: 0.2
# aap_weight: 0.2
# mip_weight: 1.0
# map_weight: 1.0
# sp_weight: 0.5
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load dataset info
        self.n_items = config['itemnum']
        # self.n_cates = 
        self.n_users = config['usernum']


        # self.n_items = dataset.item_num + 1  # for mask token
        self.mask_token = self.n_items - 1
        self.n_features = config['catenum']
        # self.item_feat = dataset.get_item_feature()

        # define layers and loss
        # modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.feature_embedding = nn.Embedding(self.n_features, self.hidden_size, padding_idx=0)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # modules for pretrain
        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_fct = nn.BCELoss(reduction='none')

        # modules for finetune

        if self.loss_type == 'CE' and self.train_stage == 'finetune':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == 'finetune':
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ['pretrain', 'finetune']
        if self.train_stage == 'pretrain':
            self.apply(self._init_weights)
        # else:
        #     # load pretrained model for finetune
        #     pretrained = torch.load(self.pre_model_path)
        #     self.logger.info(f'Load pretrained model from {self.pre_model_path}')
        #     self.load_state_dict(pretrained['state_dict'])

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

    def _associated_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    def _masked_item_prediction(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(sequence_output.view([-1, sequence_output.size(-1)]))  # [B*L H]
        target_item_emb = target_item_emb.view([-1, sequence_output.size(-1)])  # [B*L H]
        score = torch.mul(sequence_output, target_item_emb)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]

    def _masked_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, sequence_output.size(-1), 1])  # [B*L H 1]
        # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
        score = torch.matmul(feature_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L feature_num]

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]

    def get_attention_mask(self, sequence, bidirectional=True):
        """
        In the pre-training stage, we generate bidirectional attention mask for multi-head attention.

        In the fine-tuning stage, we generate left-to-right uni-directional attention mask for multi-head attention.
        """
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(sequence.device)
            extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask



    def pretrain(
        self, features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment
    ):
        """Pretrain out model using four pre-training tasks:

            1. Associated Attribute Prediction

            2. Masked Item Prediction

            3. Masked Attribute Prediction

            4. Segment Prediction
        """
        # Encode masked sequence
        sequence_output = self.forward(masked_item_sequence)

        feature_embedding = self.feature_embedding.weight
        # AAP
        aap_score = self._associated_attribute_prediction(sequence_output, feature_embedding)
        aap_loss = self.loss_fct(aap_score, features.view(-1, self.n_features).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.mask_token).float() * \
                   (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)
        pos_score = self._masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self._masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.loss_fct(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.mask_token).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self._masked_attribute_prediction(sequence_output, feature_embedding)
        map_loss = self.loss_fct(map_score, features.view(-1, self.n_features).float())
        map_mask = (masked_item_sequence == self.mask_token).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        # take the last position hidden as the context
        segment_context = self.forward(masked_segment_sequence)[:, -1, :]  # [B H]
        pos_segment_emb = self.forward(pos_segment)[:, -1, :]
        neg_segment_emb = self.forward(neg_segment)[:, -1, :]  # [B H]
        pos_segment_score = self._segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self._segment_prediction(segment_context, neg_segment_emb)
        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)
        sp_loss = torch.sum(self.loss_fct(sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)))

        pretrain_loss = self.aap_weight * aap_loss \
                        + self.mip_weight * mip_loss \
                        + self.map_weight * map_loss \
                        + self.sp_weight * sp_loss

        return pretrain_loss

    def _neg_sample(self, item_set):  # [ , ]
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_zero_at_left(self, sequence):
        # had truncated according to the max_length
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence
        return sequence

    def reconstruct_pretrain_data(self, item_seq, item_seq_len, cate_seq):
        """Generate pre-training data for the pre-training stage."""
        device = item_seq.device
        batch_size = item_seq.size(0)

        # We don't need padding for features
        item_feature_seq = cate_seq

        end_index = item_seq_len.cpu().numpy().tolist()
        item_seq = item_seq.cpu().numpy().tolist()
        item_feature_seq = item_feature_seq.cpu().numpy().tolist()

        # we will padding zeros at the left side
        # these will be train_instances, after will be reshaped to batch
        sequence_instances = []
        associated_features = []  # For Associated Attribute Prediction and Masked Attribute Prediction
        long_sequence = []
        for i, end_i in enumerate(end_index):
            sequence_instances.append(item_seq[i][:end_i])
            long_sequence.extend(item_seq[i][:end_i])
            # padding feature at the left side
            associated_features.extend([[0] * self.n_features] * (self.max_seq_length - end_i))
            for indexes in item_feature_seq[i][:end_i]:
                features = [0] * self.n_features
                try:
                    # multi class
                    for index in indexes:
                        if index >= 0:
                            features[index] = 1
                except:
                    # single class
                    features[indexes] = 1
                associated_features.append(features)

        # Masked Item Prediction and Masked Attribute Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = instance.copy()
            neg_item = instance.copy()
            for index_id, item in enumerate(instance):
                prob = random.random()
                if prob < self.mask_ratio:
                    masked_sequence[index_id] = self.mask_token
                    neg_item[index_id] = self._neg_sample(instance)
            masked_item_sequence.append(self._padding_zero_at_left(masked_sequence))
            pos_items.append(self._padding_zero_at_left(pos_item))
            neg_items.append(self._padding_zero_at_left(neg_item))

        # Segment Prediction
        masked_segment_list = []
        pos_segment_list = []
        neg_segment_list = []
        for instance in sequence_instances:
            if len(instance) < 2:
                masked_segment = instance.copy()
                pos_segment = instance.copy()
                neg_segment = instance.copy()
            else:
                sample_length = random.randint(1, len(instance) // 2)
                start_id = random.randint(0, len(instance) - sample_length)
                neg_start_id = random.randint(0, len(long_sequence) - sample_length)
                pos_segment = instance[start_id:start_id + sample_length]
                neg_segment = long_sequence[neg_start_id:neg_start_id + sample_length]
                masked_segment = instance[:start_id] + [self.mask_token] * sample_length \
                                 + instance[start_id + sample_length:]
                pos_segment = [self.mask_token] * start_id + pos_segment + \
                              [self.mask_token] * (len(instance) - (start_id + sample_length))
                neg_segment = [self.mask_token] * start_id + neg_segment + \
                              [self.mask_token] * (len(instance) - (start_id + sample_length))
            masked_segment_list.append(self._padding_zero_at_left(masked_segment))
            pos_segment_list.append(self._padding_zero_at_left(pos_segment))
            neg_segment_list.append(self._padding_zero_at_left(neg_segment))

        associated_features = torch.tensor(associated_features, dtype=torch.long, device=device)
        associated_features = associated_features.view(-1, self.max_seq_length, self.n_features)

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        masked_segment_list = torch.tensor(masked_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        pos_segment_list = torch.tensor(pos_segment_list, dtype=torch.long, device=device).view(batch_size, -1)
        neg_segment_list = torch.tensor(neg_segment_list, dtype=torch.long, device=device).view(batch_size, -1)

        return associated_features, masked_item_sequence, pos_items, neg_items, \
               masked_segment_list, pos_segment_list, neg_segment_list


    def forward(self, item_seq, bidirectional=True):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq, bidirectional=bidirectional)
        trm_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = trm_output[-1]  # [B L H]
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, cate_seq, pos_items):




        if self.train_stage == 'pretrain':
            features, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment \
                = self.reconstruct_pretrain_data(item_seq, item_seq_len, cate_seq)

            loss = self.pretrain(
                features, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment
            )
        # finetune
        else:

            # we use uni-directional attention in the fine-tuning stage
            seq_output = self.forward(item_seq, bidirectional=False)
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)

            # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


    def full_sort_predict(self, item_seq, item_seq_len):

        seq_output = self.forward(item_seq, bidirectional=False)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items - 1]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores

import pytorch_lightning as pl
from collections import defaultdict
from new_metric import *

class casr(pl.LightningModule):
    def __init__(self, itemmodel,  args
        ):
        super().__init__()
        self.teachingflag = True
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.m_logsoftmax2 = torch.nn.LogSoftmax(dim=2)

        self.loss = torch.nn.CrossEntropyLoss()
        self.args = args
        self.itemmodel = itemmodel

        self.train_epoch = 0
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)


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
        user_batch = torch.tensor(user_seq_batch).to(self.args.device).squeeze()
        # valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        # label_numpy = valid_numpy.nonzero()
        item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49




        loss = self.itemmodel.calculate_loss(item_seq_batch, item_seq_len, cate_seq_batch, item_y_batch)



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
        if args.train_stage =='finetune':
            user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


            item_seq_batch = item_seq_batch.squeeze(0)
            item_y_batch = item_y_batch.squeeze(0)
            cate_seq_batch = cate_seq_batch.squeeze(0)
            cate_y_batch = cate_y_batch.squeeze(0)
            catey_seq_batch = catey_seq_batch.squeeze(0)
            itemy_seq_batch = itemy_seq_batch.squeeze(0)
            user_batch = torch.tensor(user_seq_batch).to(self.args.device).squeeze()
            item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49


            logit = self.itemmodel.full_sort_predict(item_seq_batch, item_seq_len)
        
            valid_labels = torch.ones((logit.shape[0])).long().to(args.device)


            metrics = evaluate(logit, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)


            # cate_recall_5, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), k=5)




            return {'metrics':metrics}
        else:
            return 0

    


    def validation_epoch_end(self, validation_step_outputs):
        if args.train_stage =='finetune':
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
        item_seq_len = torch.ones(item_y_batch.size()).long().to(item_seq_batch.device)*49


        logit = self.itemmodel.full_sort_predict(item_seq_batch, item_seq_len)
    
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


config['usernum'] = usernum+1
config['itemnum'] = itemnum+1
config['catenum'] = catenum+1

model = S3Rec(config)

# itemmodel = cudaSASRec(usernum, itemnum, args)

# catemodel = cudaSASRec(usernum, catenum, args )




from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=20, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@20', mode='max')


if args.train_stage =='pretrain':
    print(args.train_stage)
    args.lr=0.002
    recmodel = casr(model,  args)
    trainer = pl.Trainer(devices=[args.cuda],  accelerator="gpu", accumulate_grad_batches=1, max_epochs=50,  default_root_dir='None')
    torch.save(model.state_dict(), './savemodel/{}_model.pth'.format(args.datasets))

##test test
    trainer.fit(recmodel, mydata)

if args.train_stage =='finetune':
    print(args.train_stage)
    model.load_state_dict(torch.load('./savemodel/{}_model.pth'.format(args.datasets)))
    recmodel = casr(model,  args)

    trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/s3rec/recall{}/lr{}'.format(args.datasets , args.recall, args.lr))
    trainer.fit(recmodel, mydata)
    trainer.test(recmodel, mydata, ckpt_path='best')