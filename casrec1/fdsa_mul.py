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
parser.add_argument('--datasets',  default='beer', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--incatemaxlen', type=int, default=10)


parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--lr', default=.002, type=float)
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


parser.add_argument('--fdsa_alpha', default=.2, type=float)
parser.add_argument('--temperature', default=1., type=float)

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



if args.datasets =='beer':
    args.dataset = './data/beer_10_100.csv'
    # args.lr = 0.001
    # args.lr = 0.0001
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    args.beam_size=10
    args.hidden_units = 64


if args.datasets =='taobao':
    args.dataset = './data/subtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    args.beam_size=20
    args.hidden_unit=128


if args.datasets =='randomijcai':
    args.dataset = './data/randomijcai_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    args.beam_size=20
    args.hidden_unit=128

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
elif args.datasets =='taobao' or args.datasets =='bigtaobao' or args.datasets =='midtaobao' or args.datasets =='randomijcai' :
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



class FDSA_MUL(torch.nn.Module):
    r"""
    FDSA is similar with the GRU4RecF implemented in RecBole, which uses two different Transformer encoders to
    encode items and features respectively and concatenates the two subparts' outputs as the final output.

    """

    def __init__(self, config, usernum, itemnum, catenum):
        super(FDSA_MUL, self).__init__()

        # # load parameters info
        # self.n_layers = config['n_layers']
        # self.n_heads = config['n_heads']
        # self.hidden_size = config['hidden_size']  # same as embedding_size
        # self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        # self.hidden_dropout_prob = config['hidden_dropout_prob']
        # self.attn_dropout_prob = config['attn_dropout_prob']
        # self.hidden_act = config['hidden_act']
        # self.layer_norm_eps = config['layer_norm_eps']

        # self.selected_features = config['selected_features']
        # self.pooling_mode = config['pooling_mode']
        # self.device = config['device']
        # self.num_feature_field = len(config['selected_features'])

        # self.initializer_range = config['initializer_range']
        # self.loss_type = config['loss_type']

        self.temperature = config['temperature']


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
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        # For simplicity, we use same architecture for item_trm and feature_trm
        self.feature_trm_encoder = TransformerEncoder(
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
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if self.loss_type == 'BPR':
            self.loss_fct = None
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer']

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

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)


        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)

        feature_emb = self.feature_embed_layer(cate_seq)

        feature_emb = feature_emb + position_embedding
        
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.dropout(feature_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        item_output = self.gather_indexes(item_output, item_seq_len )  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len )  # [B H]

        self.item_output = item_output
        self.feature_output = feature_output

        
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]

        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)
        return seq_output  # [B H]

    def calculate_loss(self,  item_seq, item_seq_len, cate_seq, pos_items, pos_cate):
 
        seq_output = self.forward(item_seq, item_seq_len, cate_seq)
        # pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        test_cate_emb = self.feature_embed_layer.weight
        cate_logits = torch.matmul(seq_output, test_cate_emb.transpose(0, 1))
        cateloss = self.loss_fct(cate_logits, pos_cate)


        #cl1
        # permuted_indices = torch.randperm(seq_output.shape[0])
        # feature_output_new = self.feature_output[permuted_indices]


        #[b,b]
        sim11 = torch.matmul(self.item_output, self.feature_output.T)/self.temperature
        cl1_label = torch.arange(sim11.shape[0], dtype=torch.long, device=sim11.device)

        cl1_loss = self.loss_fct(sim11, cl1_label)



        return loss + config['fdsa_alpha']*(cl1_loss+cateloss)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, item_seq, item_seq_len, cate_seq):

        seq_output = self.forward(item_seq, item_seq_len, cate_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


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

config['fdsa_alpha'] = args.fdsa_alpha
config['temperature'] = args.temperature

model = FDSA_MUL(config, usernum+1, itemnum+1, catenum+1)

# itemmodel = cudaSASRec(usernum, itemnum, args)

# catemodel = cudaSASRec(usernum, catenum, args )

recmodel = casr(model, None, args)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=50, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@20', mode='max')

trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/fdsamul/recall{}/lr{}'.format(args.datasets , args.recall, args.lr))

##test test
trainer.fit(recmodel, mydata)

trainer.test(recmodel, mydata, ckpt_path='best')