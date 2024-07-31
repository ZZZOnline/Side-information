#added with additionalry task as auxiliary task with mmoe model.
#two ffn for cate and item

#gating is determined by nn.linear(d,1) with input mean value of item sequence and category sequence.

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import *
import sys
import os
import datetime
import pickle

from new_utils import *
from torch.nn.init import xavier_uniform_, constant_, normal_




def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
import argparse
import datetime



parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', default=128, type=int) #mat1(a,b)->a
parser.add_argument('--beam_size', type=int, default=10)


parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data_name', default=None, type=str)

parser.add_argument('--dataset',  default='None', type=str)
parser.add_argument('--datasets',  default='retail', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")

parser.add_argument('--gamma', default=1, type=float)
# parser.add_argument('--gamma1', default=0, type=float)
# parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
# parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--incatemaxlen', type=int, default=10)
parser.add_argument('--integra', type=str, default='gating')

parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--lr', default=.001, type=float)
parser.add_argument('--l2_emb', default=0.005, type=float)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--model', type=str, default='casr')


parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--num_blocks', default=2, type=int)


parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--patience', default=50, type=int)
parser.add_argument('--preference', default='gating', type=str)

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
    args.beam_size=20

    args.patience=20

if args.datasets =='retail':
    args.dataset = './data/retail_10_100.csv'

    User = dataall_partition_last(args.dataset)
    args.batch_size = 512

    args.beam_size=5

if args.datasets =='midtaobao':
    args.dataset = './data/midtaobao_20_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 64
    args.beam_size = 20

if args.datasets =='smalltaobao':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/taobao/mergedmultismalltb_10_40.csv'
    User = dataall_partition_last(args.dataset)

if args.datasets =='randomijcai':
    args.dataset = './data/randomijcai_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    # args.beam_size=20
    # args.hidden_unit=128
    args.patience = 50
    # args.gamma = 0.95

if args.datasets =='taobao':
    args.dataset = './data/subtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    # args.beam_size=20

    args.patience = 50

if args.datasets =='bigtaobao':
    args.dataset = './data/bigtaobao_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 32
    args.beam_size = 20

if args.datasets =='beer':
    args.dataset = './data/beer_10_100.csv'

    # args.hidden_units = 64
    # args.beam_size=10

    User = dataall_partition_last(args.dataset)

    args.batch_size = 128
    # args.beam_size=10
    args.patience = 50

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




import json
import numpy as np
import torch
import torch.nn as nn
import math



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
    
class cudaSASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(cudaSASRec, self).__init__()




        self.user_num = user_num
        self.item_num = item_num
        self.cateall = torch.tensor([imap[i] for i in np.arange(0, itemnum+1)]) 
        self.gating_network = VanillaAttention(args.hidden_units, args.hidden_units)


        self.cateitem_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.useritem_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.dev = args.device
        self.sigmoid = torch.nn.Sigmoid()
        self.emb = args.hidden_units
        self.embedding_size = args.hidden_units
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.cate_emb = torch.nn.Embedding(catenum+1, args.hidden_units, padding_idx=0)
        self.catecate_emb = torch.nn.Embedding(catenum+1, args.hidden_units, padding_idx=0)

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.beha_emb = torch.nn.Embedding(5, args.hidden_units)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units)

        # define the module feature gating need
        self.w1 = nn.Linear(args.hidden_units, args.hidden_units)
        self.w2 = nn.Linear(args.hidden_units, args.hidden_units)


        # define the module instance gating need
        self.w3 = nn.Linear(args.hidden_units, 1, bias=False)
        self.w4 = nn.Linear(args.hidden_units, args.maxlen, bias=False)
        self.w5 = nn.Linear(args.hidden_units, args.hidden_units)
        self.w6 = nn.Linear(args.hidden_units, args.hidden_units)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
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

    def gating(self, item_logits, fea_logits):
        
        input = torch.cat([item_logits.unsqueeze(-2), fea_logits.unsqueeze(-2)], dim=-2)
        attn, weigh =self.gating_network(input)
        # seq_item_embedding = self.w5(item_logits)
        # # batch_size * seq_len * embedding_size
        # user_embedding = self.w6(fea_logits)
        # # batch_size * embedding_size
        # user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        # a = self.gelu(seq_item_embedding + user_embedding)

        return attn+item_logits
    
    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)


        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
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
    def concat(self, item, cate):
        all = torch.cat((item,cate), dim=-1)
        # output = self.w5(all) + self.b
        return all
    def feature_gating(self, seq_item_embedding, user_embedding):
        """

        choose the features that will be sent to the next stage(more important feature, more focus)
        #user_embedding(b,d)

        """

        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.gelu(seq_item_embedding + user_embedding)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item

    def feature_fuse(self, fea_y, fea_logits):

        seq_item_embedding = self.w1(fea_y)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(fea_logits)
        # batch_size * embedding_size
        # user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        a = self.gelu(seq_item_embedding + user_embedding)

        return torch.mul(fea_y, a)

    def instance_attention(self,  user_item, user_embedding, mask, item_seq):
        """

        choose the last click items that will influence the prediction( more important more chance to get attention)
        user_embedding(b,d)
        """

        user_embedding_value = user_item

        user_item = self.w6(user_item)
        # batch_size * seq_len * embedding_size

        user_embedding = self.w5(user_embedding)
        # batch_size * embedding_size
        # print(user_item.shape, user_embedding.shape)
        score = torch.einsum('bld,bd->bl',user_item, user_embedding)

        score =   score \
                / math.sqrt(self.embedding_size) 
        # mask = mask.repeat(self.head_num, 1 ).unsqueeze(1)
        # print(mask.shape)
        if mask is not None:

            if score.dtype == torch.float16:
                score = score.masked_fill(mask == 0, -65500)
            else:
                score = score.masked_fill(mask == 0, -torch.inf)
        
        # if torch.rand(1)>0.99:    
        #     print('mask:',mask)
        #     print('score:',score)
        score = self.dropout(nn.functional.softmax(score, dim=-1))


        # instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1

        output = torch.einsum('bl,bld->bd',score, user_embedding_value + item_seq)
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
        output = self.feed_forward(output.unsqueeze(1))

        return output.squeeze()


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


import pytorch_lightning as pl
from collections import defaultdict
from new_metric import *

class starcateseq(torch.nn.Module):
    def __init__(self, embedding_dim, cate_embedding_dim,  dropout_rate,  args):
        super(starcateseq, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.lastk =args.lastk
        self.norm = torch.nn.LayerNorm(args.hidden_units)
        self.embedding_dim = embedding_dim

        self.cate_embedding_dim = cate_embedding_dim

        self.cate_querynetwork = nn.Linear(embedding_dim, args.hidden_units)
        self.cate_keynetwork = nn.Linear(embedding_dim, args.hidden_units)
        self.item_valuenetwork = nn.Linear(embedding_dim, embedding_dim)
        # #不需要归零化
        # self.weights = nn.Parameter(torch.randn( self.lastk, hidden_size), requires_grad=True)
        
        self.feed_forward = PointWiseFeedForward(args.hidden_units, dropout_rate)

        self.hidden_size = embedding_dim

        self.head_num = args.num_heads


        assert self.hidden_size%self.head_num ==0
        self.head_size = args.hidden_units // self.head_num

        self.apply(self._init_weights)

        # self.cate_querynetwork.weight.data.normal_(mean=0.0, std=0.02)
        # self.item_keynetwork.weight.data.normal_(mean=0.0, std=0.02)
        # self.item_valuenetwork.weight.data.normal_(mean=0.0, std=0.02)
        # self.weights.data.normal_(mean=0.0, std=0.02)



    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0., 1 / self.embedding_size)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, item_seq, cate_seq, cate_y, mask):
        cate_query = self.cate_querynetwork(cate_y)
        
        cate_key = self.cate_keynetwork(cate_seq)

        item_value = self.item_valuenetwork(item_seq)+item_seq



        cate_query_ = torch.cat(torch.split(cate_query, self.head_size, dim=2), dim=0)
        cate_key_ = torch.cat(torch.split(cate_key, self.head_size, dim=2), dim=0)
        item_value_ = torch.cat(torch.split(item_value, self.head_size, dim=2), dim=0)

        # test_item_emb = self.item_embedding.weight

        att_0 = torch.einsum('bnd, bhd -> bnh', cate_query_, cate_key_)

        scores = att_0 \
                / math.sqrt(self.cate_embedding_dim) 

        # 2. dealing with padding and softmax.
        # get padding masks
        # mask = (x > 0)

        mask = mask.repeat(self.head_num, 1 ).unsqueeze(1)
        # print(mask.shape)
        if mask is not None:
            # assert len(mask.shape) == 2
            #mask [a,b] ->[a,1,b,b]
            # mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            if scores.dtype == torch.float16:
                scores = scores.masked_fill(mask == 0, -65500)
            else:
                scores = scores.masked_fill(mask == 0, -torch.inf)

        c_attn = self.dropout(nn.functional.softmax(scores, dim=-1))


        att_out = torch.einsum('bnh, bhd -> bnd', c_attn, item_value_)

        att_out = torch.cat(torch.split(att_out, item_seq.shape[0], dim=0), dim=2) # div batch_size
        # print(att_out.shape)


        att_value = self.feed_forward(att_out)


        return att_value

class PASR(pl.LightningModule):
    def __init__(self,  args , usernum, itemnum, catenum):
        super().__init__()

        self.args = args

        self.usernum = usernum
        
        self.itemnum = itemnum

        self.catenum = catenum

    


        self.itemmodel = cudaSASRec(self.usernum, self.itemnum, args)
        self.catemodel = cudaSASRec(self.usernum, self.catenum, args)
        

                
        self.teachingflag = True
        self.m_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.m_logsoftmax2 = torch.nn.LogSoftmax(dim=2)
        self.loss = torch.nn.CrossEntropyLoss()
        self.loss_type = 'CE'
        self.beloss = torch.nn.BCEWithLogitsLoss()
        self.args = args



        self.train_epoch = 0
        self.n_e_sh = 1
        self.n_e_sp = 1
        self.itemgates = nn.Parameter(torch.randn(args.hidden_units, self.n_e_sh + self.n_e_sp), requires_grad=True)
        self.categates = nn.Parameter(torch.randn(args.hidden_units, self.n_e_sh + self.n_e_sp), requires_grad=True)


        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.starmodel = starcateseq(args.hidden_units,args.hidden_units, args.dropout_rate, args)
        self.catefeed = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        self.itemfeed = PointWiseFeedForward(args.hidden_units, args.dropout_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.shared_experts = nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units))
        self.catespecific_experts = nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units)) 
        self.itemspecific_experts = nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units)) 

        # self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units)) for i in range(self.n_e_sh)])
        # self.catespecific_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units)) for i in range(self.n_e_sp)])
        # self.itemspecific_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units)) for i in range(self.n_e_sp)])


        # self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units), nn.ReLU(),nn.Linear(args.hidden_units, args.hidden_units)) for i in range(self.n_e_sh)])
        # self.specific_experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_units, args.hidden_units), nn.ReLU(),nn.Linear(args.hidden_units, args.hidden_units)) for i in range(self.n_b * self.n_e_sp)])
        # self.w_gates = nn.Parameter(torch.randn(self.n_b, args.hidden_units, self.n_e_sh + self.n_e_sp), requires_grad=True)
        # self.token_embeddings = token_embeddings
        self.ln = nn.LayerNorm(args.hidden_units)


    def training_step(self, batch, batch_idx):
        # print('train')
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
        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        label_numpy = valid_numpy.nonzero()

        mask = (item_seq_batch>0)


        cate_logits = self.catemodel.log2feats(cate_seq_batch)

        cate_input = cate_logits

        

        cate_outputs = self.catemodel.heads(cate_input, None)      
        floss = self.loss(cate_outputs[:,-1,:], cate_y_batch)

        # user_cate = self.catemodel.feature_gating(self.catemodel.item_emb(cate_seq_batch), self.catemodel.user_emb(user_batch))[:,-1,:].unsqueeze(1)

        #behasum1
        if args.preference =='gating':
            cate_fuse = self.catemodel.feature_fuse(self.catemodel.item_emb(cate_y_batch.view(-1,1)), cate_logits[:, -1, :].unsqueeze(1))
        if args.preference =='add':
            cate_fuse = self.catemodel.item_emb(cate_y_batch.view(-1,1)) + cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='dynamic':
            cate_fuse =  cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='direct':
            cate_fuse = self.catemodel.item_emb(cate_y_batch.view(-1,1))  

        # cate_fuse = self.catemodel.feature_fuse(self.catemodel.item_emb(cate_y_batch.view(-1,1)), cate_logits[:, -1, :].unsqueeze(1))

        # print(cate_fuse.shape)

        if args.integra =='gating':
            iteminput = self.itemmodel.gating(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

 
        if args.integra =='add':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch)+ self.catemodel.item_emb(cate_seq_batch)

        if args.integra =='concat':
            iteminput = self.itemmodel.feature_fuse1(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

        if args.integra =='item':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch)  

        cate_input = self.starmodel(iteminput, self.catemodel.item_emb(cate_seq_batch), cate_fuse , mask)

        starinput = self.ln(cate_input.squeeze())


        itemloss = None

        allinput  = self.itemmodel.cateitem_emb.weight
        cateallinput = self.catemodel.item_emb.weight
        
        # shared_starinput = self.shared_experts(starinput)
        # itemspe_starinput = self.itemspecific_experts(starinput)
        # catespe_starinput = self.catespecific_experts(starinput)

        # shared_experts_o = [e(x) for e in self.shared_experts]
        # specific_experts_o = [e(x) for e in self.specific_experts]
        gates_i = self.softmax(torch.einsum('bd, de->be', starinput, self.itemgates))
        gates_c = self.softmax(torch.einsum('bd, de->be', starinput, self.categates))



        # rearange
        experts_i_tensor = torch.cat((self.shared_experts(starinput).unsqueeze(1), self.itemspecific_experts(starinput).unsqueeze(1)),dim=1 )
        experts_c_tensor = torch.cat((self.shared_experts(starinput).unsqueeze(1), self.catespecific_experts(starinput).unsqueeze(1)),dim=1 )
        # torch.stack([torch.stack(shared_experts_o+specific_experts_o[i*2: (i+1)*2]) for i in range(4)])
        i_output = torch.einsum('bed, be->bd', experts_i_tensor, gates_i)
        c_output = torch.einsum('bed, be->bd', experts_c_tensor, gates_c)
        # outputs = torch.cat([torch.zeros_like(x).unsqueeze(0), output])

        


        logits = torch.einsum('cd,bd -> bc', allinput , starinput+self.ln(i_output))

        itemloss = self.loss(logits, item_y_batch)
        # print(allinput.shape, starinput.shape)
        # print(cateallinput.shape, self.catefeed(starinput.unsqueeze(1)).shape)
        catelogits = torch.einsum('cd, bd ->bc', cateallinput, starinput+self.ln(c_output))
        cateloss = self.loss(catelogits, cate_y_batch)


        # print(torch.mean(floss), torch.mean(itemloss))
        # print(torch.mean(floss), torch.mean(itemloss),  torch.mean(cateloss))
        loss = args.gamma*(floss+cateloss) + itemloss 


        return {'loss':loss}


    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)
        # train_recall_5 = torch.stack([o['recall_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_recall_5', train_recall_5)
        # train_mrr_5 = torch.stack([o['mrr_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_mrr_5', train_mrr_5)
        # train_recall_5 = torch.stack([o['recall_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_recall_5', train_recall_5)
        # train_mrr_5 = torch.stack([o['mrr_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_mrr_5', train_mrr_5)


    def validation_step(self, batch, batch_idx):


        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        user_batch = torch.tensor(user_seq_batch).to(self.args.device).squeeze()
        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0
        label_numpy = valid_numpy.nonzero()

        mask = (item_seq_batch>0)


        # print(next(self.featmodel[num].parameters()).device)
        cate_logits = self.catemodel.log2feats(cate_seq_batch)



        cate_input = cate_logits[torch.arange(cate_logits.size(0)), -1, :] 
        
        # if batch_idx %100==0:
        #     print('cate: user_input:{}, cate_logits:{}'.format(userinput, cate_logits[torch.arange(cate_logits.size(0)), item_len_batch, :]))
        
        cate_outputs = self.catemodel.head(cate_input, None).squeeze()

        pred_item_prob1 = None
        

        log_prob_cate_short = self.m_logsoftmax(cate_outputs)
        log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)
        
        pred_cate_index = pred_cate_index.detach()
        log_prob_cate_short = log_prob_cate_short

        prob_cate_short = self.m_logsoftmax(cate_outputs)
        
        prob_cate_short, pred_cate_index = torch.topk(prob_cate_short, args.beam_size, dim=-1)
        
        pred_cate_index = pred_cate_index.detach()
        # prob_cate_short = log_prob_cate_short

        pred_cate_emb = self.catemodel.item_emb(pred_cate_index)

        prob_cate_beam = prob_cate_short[:, :args.beam_size]

        if args.preference =='gating':
            cate_fuse = self.catemodel.feature_fuse(pred_cate_emb, cate_logits[:, -1, :].unsqueeze(1))
        if args.preference =='add':
            cate_fuse = pred_cate_emb + cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='dynamic':
            cate_fuse =  cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='direct':
            cate_fuse = pred_cate_emb 
   

        if args.integra =='gating':
            iteminput = self.itemmodel.gating(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

 
        if args.integra =='add':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch)+ self.catemodel.item_emb(cate_seq_batch)

        if args.integra =='concat':
            iteminput = self.itemmodel.feature_fuse1(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

        if args.integra =='item':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch) 


        cate_input = self.ln(self.starmodel(iteminput, self.catemodel.item_emb(cate_seq_batch) , cate_fuse, mask))


        # starinput = cate_input


        allinput  = self.itemmodel.cateitem_emb.weight

        # catetest_cate_emb = self.catemodel.item_emb(self.itemmodel.cateall.to(self.args.device))
        gates_i = self.softmax(torch.einsum('bhd, de->bhe', cate_input, self.itemgates))
        # gates_c = self.softmax(torch.einsum('bd, de->be', starinput, self.categates))



        # rearange
        experts_i_tensor = torch.cat((self.shared_experts(cate_input).unsqueeze(1), self.itemspecific_experts(cate_input).unsqueeze(1)),dim=1 )
        # experts_c_tensor = torch.cat((self.shared_experts(starinput).unsqueeze(1), self.catespecific_experts(starinput).unsqueeze(1)),dim=1 )
        # torch.stack([torch.stack(shared_experts_o+specific_experts_o[i*2: (i+1)*2]) for i in range(4)])
        i_output = torch.einsum('behd, bhe->bhd', experts_i_tensor, gates_i)
        # c_output = torch.einsum('bed, be->bd', experts_c_tensor, gates_c)

        catelogits = torch.einsum('cd, bhd -> bhc', allinput  , self.ln(i_output)+cate_input)

        

        # logit_2 = self.itemmodel.heads(output, None)

        logit_2_1 = self.m_logsoftmax2(catelogits)

        outputs = logit_2_1 + prob_cate_beam[:,:,None]

        pred_item_prob1 = outputs
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim = 1)

        valid_labels = torch.ones((pred_item_prob1.shape[0])).long().to(args.device)

        # metrics = evaluate(pred, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)
        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)

        cate_recall, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=10)



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
        
        user_seq_batch, item_y_batch, cate_y_batch,  item_seq_batch, cate_seq_batch,  pad_incateitem_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch


        item_seq_batch = item_seq_batch.squeeze(0)
        item_y_batch = item_y_batch.squeeze(0)
        cate_seq_batch = cate_seq_batch.squeeze(0)
        cate_y_batch = cate_y_batch.squeeze(0)
        catey_seq_batch = catey_seq_batch.squeeze(0)
        itemy_seq_batch = itemy_seq_batch.squeeze(0)
        user_batch = torch.tensor(user_seq_batch).to(self.args.device).squeeze()
        valid_numpy = np.array(itemy_seq_batch.to('cpu'))>0

        mask = (item_seq_batch>0)


        # print(next(self.featmodel[num].parameters()).device)
        cate_logits = self.catemodel.log2feats(cate_seq_batch)


        cate_input = cate_logits[torch.arange(cate_logits.size(0)), -1, :] 
        
        # if batch_idx %100==0:
        #     print('cate: user_input:{}, cate_logits:{}'.format(userinput, cate_logits[torch.arange(cate_logits.size(0)), item_len_batch, :]))
        
        cate_outputs = self.catemodel.head(cate_input, None).squeeze()



        pred_item_prob1 = None
        

        log_prob_cate_short = self.m_logsoftmax(cate_outputs)
        log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, args.beam_size, dim=-1)
        
        pred_cate_index = pred_cate_index.detach()
        log_prob_cate_short = log_prob_cate_short

        prob_cate_short = self.m_logsoftmax(cate_outputs)
        
        prob_cate_short, pred_cate_index = torch.topk(prob_cate_short, args.beam_size, dim=-1)
        
        pred_cate_index = pred_cate_index.detach()
        # prob_cate_short = log_prob_cate_short

        pred_cate_emb = self.catemodel.item_emb(pred_cate_index)

        prob_cate_beam = prob_cate_short[:, :args.beam_size]

        if args.preference =='gating':
            cate_fuse = self.catemodel.feature_fuse(pred_cate_emb, cate_logits[:, -1, :].unsqueeze(1))
        if args.preference =='add':
            cate_fuse = pred_cate_emb + cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='dynamic':
            cate_fuse =  cate_logits[:, -1, :].unsqueeze(1)
        if args.preference =='direct':
            cate_fuse = pred_cate_emb 

        if args.integra =='gating':
            iteminput = self.itemmodel.gating(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

        if args.integra =='add':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch)+ self.catemodel.item_emb(cate_seq_batch)

        if args.integra =='concat':
            iteminput = self.itemmodel.feature_fuse1(self.itemmodel.cateitem_emb(item_seq_batch), self.catemodel.item_emb(cate_seq_batch))

        if args.integra =='item':
            iteminput = self.itemmodel.cateitem_emb(item_seq_batch) 
        cate_input = self.ln(self.starmodel(iteminput, self.catemodel.item_emb(cate_seq_batch) , cate_fuse, mask))


        # starinput = cate_input

        allinput = self.itemmodel.cateitem_emb.weight


        gates_i = self.softmax(torch.einsum('bhd, de->bhe', cate_input, self.itemgates))
        # gates_c = self.softmax(torch.einsum('bd, de->be', starinput, self.categates))



        # rearange
        experts_i_tensor = torch.cat((self.shared_experts(cate_input).unsqueeze(1), self.itemspecific_experts(cate_input).unsqueeze(1)),dim=1 )

        i_output = torch.einsum('behd, bhe->bhd', experts_i_tensor, gates_i)
        # c_output = torch.einsum('bed, be->bd', experts_c_tensor, gates_c)

        catelogits = torch.einsum('cd, bhd -> bhc', allinput  , self.ln(i_output)+cate_input)

        logit_2_1 = self.m_logsoftmax2(catelogits)

        outputs = logit_2_1 + prob_cate_beam[:,:,None]

        pred_item_prob1 = outputs
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim = 1)

        valid_labels = torch.ones((pred_item_prob1.shape[0])).long().to(args.device)

        # metrics = evaluate(pred, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)
        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)

        cate_recall, cate_mrr, cate_ndcg = evaluate_part(cate_outputs, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=10)



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
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=.0001,
        #                                    weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001,weight_decay= 0.000001)
        #from SASREC Main.py
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98))
        return optimizer

imap, imaps = catemap(args.dataset)


recmodel = PASR(args, usernum, itemnum, catenum)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=50, verbose=False, mode="max")

modelcheckpoint = ModelCheckpoint(monitor="Val:Recall@20", mode="max")


#只用cate训练时基于catey_seq_batch
# if args.datasets=='taobao':
#     trainer = pl.Trainer(devices=[args.cuda], callbacks=[modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=100,  default_root_dir='./final/{}/cu_allshared_gamma1/gating/beam{}/gamma{}/recall{}/lr{}'.format(args.datasets ,args.beam_size, args.gamma, args.recall, args.lr))
# else:
trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./fffinal/{}/newcasrecmul2hidden/{}/pre{}/beam{}/gamma{}/recall{}/lr{}'.format(args.datasets, args.integra, args.preference, args.beam_size, args.gamma, args.recall, args.lr))

###############headnum=1

trainer.fit(recmodel, mydata)

trainer.test(recmodel, mydata, ckpt_path="best")