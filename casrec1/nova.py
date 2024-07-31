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


parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data_name', default=None, type=str)

parser.add_argument('--dataset',  default='None', type=str)
parser.add_argument('--datasets',  default='retail', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")
parser.add_argument('--hidden_units', default=128, type=int)
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



if args.datasets =='beer':
    args.dataset = './data/beer_10_100.csv'
    # args.lr = 0.001
    args.lr = 0.0001
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
        self.cate_emb = torch.nn.Embedding(catenum+1, args.hidden_units, padding_idx=0)
        self.cateall = torch.tensor([imap[i] for i in np.arange(0, itemnum+1)]) 

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
                                                            args.dropout_rate,batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)

    def log2feats(self, item_seqs, fea_seqs,  mask):

        Q = item_seqs +fea_seqs
        positions = np.tile(np.array(range(item_seqs.shape[1])), [item_seqs.shape[0], 1])
        pos_seq = self.pos_emb(torch.LongTensor(positions).to(item_seqs.device))
        Q = self.emb_dropout(Q+pos_seq)



        timeline_mask = mask
        Q *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = item_seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # iseqs = torch.transpose(iseqs, 0, 1)
        #iseqs is item representation
        iseqs = item_seqs

        for i in range(len(self.attention_layers)):

            # Q = seqs +fea_seqs

            Q = self.attention_layernorms[i](Q)
            mha_outputs, _ = self.attention_layers[i](Q, Q, iseqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            iseqs = mha_outputs + iseqs

            iseqs = self.forward_layernorms[i](iseqs)

            iseqs = self.forward_layers[i](iseqs)

            iseqs *=  ~timeline_mask.unsqueeze(-1)

            Q = iseqs + fea_seqs +pos_seq

        log_feats = self.last_layernorm(iseqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def log2feats_6(self, seqs, cate_seq, mask):
        # seqs = self.item_emb(log_seqs)

        # beha = self.beha_emb(beha_seqs)
        fuse_emb = seqs+cate_seq

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
            mha_outputs, att = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        output = torch.einsum('bln, bnd->bld', att, fuse_emb)
        output = self.forward_layers[i](output)
        # seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(output) # (U, T, C) -> (U, -1, C)

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




        item_seq = self.itemmodel.item_emb(item_seq_batch)
        cate_seq = self.itemmodel.cate_emb(cate_seq_batch)
        mask = (item_seq_batch==0)

        item_logits = self.itemmodel.log2feats(item_seq, cate_seq, mask)


        catetest_item_emb = self.itemmodel.item_emb.weight

        catetest_cate_emb = self.itemmodel.cate_emb(self.itemmodel.cateall.to(self.args.device))

        #老版本的实现，貌似不对
        # logits = torch.einsum('cd,bd -> bc', catetest_item_emb+catetest_cate_emb, item_logits[:,-1,:])

        logits = torch.einsum('cd,bd -> bc', catetest_item_emb, item_logits[:,-1,:])
        itemloss = self.loss(logits, item_y_batch)

        # outputs = self.itemmodel.head(item_logits[:,-1,:], None)

        # loss = self.loss(outputs, item_y_batch)


        # for param in self.itemmodel.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        return {'loss':itemloss}

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

        item_seq = self.itemmodel.item_emb(item_seq_batch)

        mask = (item_seq_batch==0)
        cate_seq = self.itemmodel.cate_emb(cate_seq_batch)
        item_logits = self.itemmodel.log2feats(item_seq, cate_seq, mask)


        catetest_item_emb = self.itemmodel.item_emb.weight

        # catetest_cate_emb = self.itemmodel.cate_emb(self.itemmodel.cateall.to(self.args.device))

        
        #老版本的实现，貌似不对
        # logit = torch.einsum('cd,bd -> bc', catetest_item_emb+catetest_cate_emb, item_logits[:,-1,:])

        logit = torch.einsum('cd,bd -> bc', catetest_item_emb, item_logits[:,-1,:])
    
        valid_labels = torch.ones((logit.shape[0])).long().to(args.device)


        metrics = evaluate(logit, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=args.recall)


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

        item_seq = self.itemmodel.item_emb(item_seq_batch)

        mask = (item_seq_batch==0)

        cate_seq = self.itemmodel.cate_emb(cate_seq_batch)
        item_logits = self.itemmodel.log2feats(item_seq, cate_seq, mask)


        catetest_item_emb = self.itemmodel.item_emb.weight

        # catetest_cate_emb = self.itemmodel.cate_emb(self.itemmodel.cateall.to(self.args.device))

        #老版本的实现，貌似不对
        # logit = torch.einsum('cd,bd -> bc', catetest_item_emb+catetest_cate_emb, item_logits[:,-1,:])

        logit = torch.einsum('cd,bd -> bc', catetest_item_emb, item_logits[:,-1,:])

    
        valid_labels = torch.ones((logit.shape[0])).long().to(args.device)*100


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
itemmodel = cudaSASRec(usernum, itemnum, args)

catemodel = cudaSASRec(usernum, catenum, args )

recmodel = casr(itemmodel, catemodel, args)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


early_stop_callback = EarlyStopping(monitor="Val:Recall@20", min_delta=0.00, patience=50, verbose=False, mode="max")
modelcheckpoint = ModelCheckpoint(monitor='Val:Recall@20', mode='max')

trainer = pl.Trainer(devices=[args.cuda], callbacks=[early_stop_callback, modelcheckpoint], accelerator="gpu", accumulate_grad_batches=1, max_epochs=200,  default_root_dir='./final/{}/nova/recall{}/lr{}'.format(args.datasets , args.recall, args.lr))

##test test
trainer.fit(recmodel, mydata)
trainer.test(recmodel, mydata, ckpt_path='best')