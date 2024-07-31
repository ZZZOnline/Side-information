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
parser.add_argument('--integra', type=str, default='gating')

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

parser.add_argument('--shared', default='shared', type=str)
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

if args.datasets =='randomijcai':
    args.dataset = './data/randomijcai_10_100.csv'
    User = dataall_partition_last(args.dataset)
    args.batch_size = 128
    args.beam_size=20
    args.hidden_unit=128
    
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


mydata = DataModule(traindataset, testdataset, testdataset, args)



if args.datasets=='retail':
    args.config_files = './configs/retail.yaml'
    args.beam_size=5
elif args.datasets == 'sub_taobao':
    args.config_files = './configs/sub_taobao.yaml'
elif args.datasets =='taobao' or args.datasets =='bigtaobao' or args.datasets =='midtaobao' or args.datasets =='kuai' or args.datasets =='kuai_pure'or args.datasets =='randomijcai':
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


class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            # print(attribute_tensor.shape)
            # print(attribute_query(attribute_tensor).shape)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)

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
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class TrueLastDIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(TrueLastDIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    #[b,h,l,d]

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask, true_item):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(true_item)

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)

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
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)+ input_tensor
        return hidden_states


class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class TrueLastDIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(TrueLastDIFTransformerLayer, self).__init__()
        self.multi_head_attention = TrueLastDIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask, true_item):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask, true_item)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output



class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
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
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TrueLastDIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(TrueLastDIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers-1)])
        self.layer.append(TrueLastDIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        ))

    def forward(self, output_states, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
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
        true_states = hidden_states
        for i in range(len(self.layer)-1):
            hidden_states = self.layer[i](hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        output = self.layer[i+1](hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask, output_states)
        if output_all_encoded_layers:
                all_encoder_layers.append(output)
        if not output_all_encoded_layers:
            all_encoder_layers.append(output)
        return all_encoder_layers
    



class QSASRecD(torch.nn.Module):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, usernum, itemnum, catenum, args):
        super(QSASRecD, self).__init__()
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
        # self.attribute_predictor = config['attribute_predictor']
        self.attribute_predictor = 'not'

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.fea_embedding = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)
        self.fea_embedding1 = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TrueLastDIFTransformerEncoder(
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


        self.args = args
        self.cateall = torch.tensor([imap[i] for i in np.arange(0, itemnum)]) 
        # define the module feature gating need
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.b = nn.Parameter(torch.zeros(args.hidden_units), requires_grad=True).to(self.dev)

        # define the module instance gating need
        self.w3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.w4 = nn.Linear(self.hidden_size, args.maxlen, bias=False)
        self.w5 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.w6 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.gating_network = VanillaAttention(self.hidden_size, self.hidden_size)


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

    def gating(self, item_logits, fea_logits):
        # print(item_logits.shape, fea_logits.shape)
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
    
    def feature_fuse1(self, item_logits, fea_logits):

        all = torch.cat((item_logits, fea_logits), dim=-1)
        output = self.w5(all) 
        return output

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
        if args.integra =='gating':
            iteminput = self.gating(item_emb, feature_emb)


        if args.integra =='add':
            iteminput = item_emb+ feature_emb


        if args.integra =='concat':
            iteminput = self.feature_fuse1(item_emb, feature_emb)


        if args.integra =='item':
            iteminput = item_emb

        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(iteminput, input_emb, [feature_emb],position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, cate_seq, pos_items, pos_cate):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, cate_seq)
        # pos_items = interaction[self.POS_ITEM_ID]
        if self.args.shared=='shared':
            fea_embedding = self.fea_embedding
        else:
            fea_embedding = self.fea_embedding1
        if args.integra =='gating':
            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = self.gating(catetest_item_emb, catetest_cate_emb) 

        if args.integra =='add':
            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = catetest_item_emb + catetest_cate_emb 

        if args.integra =='concat':
            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = self.feature_fuse1(catetest_item_emb, catetest_cate_emb) 

        if args.integra =='item':
            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb

        if True:  # self.loss_type = 'CE'
            # test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, allinput.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            if self.attribute_predictor!='' and self.attribute_predictor!='not':
                loss_dic = {'item_loss':loss}
                attribute_loss_sum = 0
                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output)
                    attribute_labels = pos_cate
                    attribute_labels1 = attribute_labels.clone().detach()#zz crete
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes)
                    if torch.rand(1)>0.99:
                        with open('result.txt','w') as file:
                            file.write(str(pos_cate.shape))
                            file.write(str(attribute_labels.shape))
                        # print(pos_cate.shape, attribute_labels.shape)
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
        # test_items_emb = self.item_embedding.weight
        if self.args.shared=='shared':
            fea_embedding = self.fea_embedding
        else:
            fea_embedding = self.fea_embedding1
    
        if args.integra =='gating':

            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = self.gating(catetest_item_emb, catetest_cate_emb) 

        if args.integra =='add':

            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = catetest_item_emb + catetest_cate_emb 

        if args.integra =='concat':

            catetest_item_emb = self.item_embedding.weight
            catetest_cate_emb = fea_embedding(self.cateall.to(self.args.device))
            allinput = self.feature_fuse1(catetest_item_emb, catetest_cate_emb) 

        if args.integra =='item':

            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb

        scores = torch.matmul(seq_output, allinput.transpose(0, 1))  # [B, item_num]
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

        # if args.integra =='gating':
        #     iteminput = self.itemmodel.gating(self.itemmodel.item_emb(item_seq_batch), self.itemmodel.cate_emb(cate_seq_batch))
        #     catetest_item_emb = self.itemmodel.item_emb.weight
        #     catetest_cate_emb = self.itemmodel.cate_emb(self.cateall.to(self.args.device))
        #     allinput = self.itemmodel.gating(catetest_item_emb, catetest_cate_emb) 

        # if args.integra =='add':
        #     iteminput = self.itemmodel.item_emb(item_seq_batch)+ self.itemmodel.cate_emb(cate_seq_batch)
        #     catetest_item_emb = self.itemmodel.item_emb.weight
        #     catetest_cate_emb = self.itemmodel.cate_emb(self.cateall.to(self.args.device))
        #     allinput = catetest_item_emb + catetest_cate_emb 

        # if args.integra =='concat':
        #     iteminput = self.itemmodel.feature_fuse1(self.itemmodel.item_emb(item_seq_batch), self.itemmodel.cate_emb(cate_seq_batch))
        #     catetest_item_emb = self.itemmodel.item_emb.weight
        #     catetest_cate_emb = self.itemmodel.cate_emb(self.cateall.to(self.args.device))
        #     allinput = self.itemmodel.feature_fuse1(catetest_item_emb, catetest_cate_emb) 

        # if args.integra =='item':
        #     iteminput = self.itemmodel.item_emb(item_seq_batch)  
        #     catetest_item_emb = self.itemmodel.item_emb.weight
        #     allinput = catetest_item_emb


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

imap, imaps = catemap(args.dataset)
model = QSASRecD( config, usernum+1, itemnum+1, catenum+1, args)

# itemmodel = cudaSASRec(usernum, itemnum, args)

# catemodel = cudaSASRec(usernum, catenum, args )

recmodel = casr(model, None, args)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@10", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@10', mode='max')

trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/difsr_decode/integra{}/shared{}/recall{}/lr{}'.format(args.datasets, args.integra, args.shared, args.recall, args.lr))

##test test
trainer.fit(recmodel, mydata)

trainer.test(recmodel, mydata, ckpt_path='best')