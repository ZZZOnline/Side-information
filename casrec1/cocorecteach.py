from matplotlib.scale import LogitTransform

from new_utils import *
from new_metric import *

import torch
import numpy as np
import pytorch_lightning as pl
#构造一个必须有buy行为的情况,buy需要不在最后一个

from torch import nn
#bert 写成sasrec了

from torch.utils.data import *


class mydataset(torch.utils.data.Dataset):
    def __init__(self, usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors, args):   # self 参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        self.usetensors, self.itemytensors, self.cateytensors, self.behaytensors, self.itemseqtensors, self.cateseqtensors, self.behaseqtensors, self.timeseqtensors, self.padincateitemtensors, self.padincatetimetensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors = usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors
        self.args = args

        
    def __len__(self):
        return len(self.usetensors)
    def __getitem__(self, idx):
        
        return self.usetensors[idx], self.itemytensors[idx], self.cateytensors[idx], self.behaytensors[idx], self.itemseqtensors[idx], self.cateseqtensors[idx], self.behaseqtensors[idx], self.timeseqtensors[idx], self.padincateitemtensors[idx], self.padincatetimetensors[idx], self.padcatesettensors[idx], self.itemyseqtensors[idx], self.cateyseqtensors[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, traindataset, testdataset, args):
        super().__init__()
        self.traindataset = traindataset
        self.testdataset = testdataset
        self.args = args

    # def setup(self, stage: str):


    def train_dataloader(self):
        return DataLoader(self.traindataset, 1, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.testdataset, 1, num_workers=0, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.testdataset, 1, num_workers=0, persistent_workers=False)


#retail is too small
import numpy as np
import torch
import sys

FLOAT_MIN = -sys.float_info.max
from tqdm import tqdm

import time
import os
import torch
import numpy as np

import sys

import datetime
import torch
import numpy as np
import pickle

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
import argparse
import datetime



parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', default=256, type=int) #mat1(a,b)->a
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--bptt', default=1, type=int)
parser.add_argument('--cate_embedding_dim', type=int, default=64 )#mat2(a,b)->a  [6 10]
parser.add_argument('--cate_hidden_size', type=int, default=64) #? []
parser.add_argument('--cate_shared_embedding', default=64, type=int)#default=1
parser.add_argument('--cate_window_size', default=5, type=int)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_ml')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--data_folder', default='/home/zzzou/ticasrec/dataset/taobao/', type=str)



parser.add_argument('--dataset',  default='None', type=str)

# parser.add_argument('--dataset',  default='/home/zzzou/multibehavior/dataset/ijcai/ijcai/mergedijcai_10_300.csv', type=str)

parser.add_argument('--datasets',  default='retail', type=str)


# parser.add_argument('--dataset',  default='/home/zzzou/multibehavior/dataset/taobao/mergedmultitaobao.csv', type=str)
# parser.add_argument('--datasets',  default='taobao', type=str)
# parser.add_argument('--data_time', default='mergedtime.pickle', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.2, type=float)
parser.add_argument('--dropout_rate', default=0.2, type=float)
# parser.add_argument("--embedding_dim", type=int, default=1114,
#                      help="using embedding")#default=0
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")#default=0
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--final_act', default='tanh', type=str)

parser.add_argument('--hidden_size', default=64, type=int)#好像没用 需要和category size在后面对齐，important，[6 8]
parser.add_argument('--hidden_units', default=64, type=int)
#和embedding_dim对齐？？。。
#input_size=output_size
# parser.add_argument('--hidden_size', default=64, type=int)#需要和后面对齐，important，

parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--incatemaxlen', type=int, default=20)
parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--loss_type', default='XE', type=str)
parser.add_argument('--lr', default=.0005, type=float)
parser.add_argument('--l2_emb', default=0.005, type=float)

parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--model_name', default="topkCascadeCategoryRNN", type=str)
parser.add_argument('--mode', type=str, default='simplest')
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--negative_num', default=1000, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)

parser.add_argument('--patience', default=1000, type=int)
# parser.add_argument("-seed", type=int, default=1,
#                      help="Seed for random initialization")

parser.add_argument('--recall', default=[5, 10, 20, 50], type=list)


parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument("-seed", type=int, default=5,
                     help="Seed for random initialization")

parser.add_argument('--shared_embedding', default=64, type=int)#default=None
parser.add_argument("-sigma", type=float, default=None,
                     help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument('--state_dict_path', default=None, type=str)


parser.add_argument('--test_observed', default=5, type=int)
parser.add_argument('--test_start_time', default=1512144000, type=int)
parser.add_argument('--time_bucket', type=int, default=64)
parser.add_argument('--time_span', type=int, default=64)
parser.add_argument('--topkc', default=3, type=int)
parser.add_argument('--train_dir', default='default', type=str)

parser.add_argument('--unknown', default=0, type=int)
parser.add_argument('--valid_start_time', default=1512057600, type=int)  
parser.add_argument('--warm_start', default=0, type=int)
parser.add_argument('--weight_decay', default=0, type=float)
# parser.add_argument('--window_size', default=30, type=int)
# parser.add_argument('--window_size', default=15, type=int)
# parser.add_argument('--cate_window_size', default=5, type=int)
parser.add_argument('--window_size', default=10, type=int)




# Get the arguments
args = parser.parse_args()
# args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)


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



hidden_size = args.hidden_size
num_layers = args.num_layers
batch_size = args.batch_size
dropout_input = args.dropout_input
dropout_hidden = args.dropout_hidden
embedding_dim = args.embedding_dim

final_act = args.final_act
loss_type = args.loss_type
optimizer_type = args.optimizer_type
lr = args.lr
weight_decay = args.weight_decay
momentum = args.momentum
eps = args.eps

n_epochs = args.n_epochs

window_size = args.window_size   #10
cate_window_size = args.cate_window_size   #5

valid_start_time = args.valid_start_time

test_start_time = args.test_start_time

observed_threshold = args.test_observed



st = datetime.datetime.now()


if args.datasets =='ijcai':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/ijcai/mergedijcai_10_300.csv'
    User, time_map, bmap = beha_dataall_partition_last(args.dataset, '1')
    args.batch_size=64

if args.datasets =='retail':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/Retail/mergedmultiretail_10_300.csv'
    # args.lr = 0.001
    User, time_map, bmap = beha_dataall_partition_last(args.dataset, 'transaction')
    args.beam_size = 3
    args.tokc = 1
    # args.beam_size = 2


if args.datasets =='smalltaobao':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/taobao/mergedmultismalltb_10_40.csv'
    User, time_map, bmap = beha_dataall_partition_last(args.dataset, 'buy')
    args.batch_size=128


if args.datasets =='taobao':
    args.dataset = '/dev-data/zzz/207/multibehavior/dataset/taobao/mergedmultitaobao.csv'
    User, time_map, bmap = beha_dataall_partition_last(args.dataset, 'buy')
    args.batch_size=128



item_res_train, cate_res_train,  beha_res_train, time_res_train, item_res_valid, cate_res_valid, beha_res_valid, time_res_valid, item_res_test, cate_res_test,  beha_res_test, time_res_test, usernum,  itemnum, catenum, timenum = beha_datatimenormalizelast(User, time_map)


num_batch = len(item_res_train) // args.batch_size
cc = 0.0
for u in item_res_train:
    cc += len(item_res_train[u])
print('average item sequence length: %.2f' % (cc / len(item_res_train)))



item_res_all = dict()
cate_res_all = dict()
beha_res_all = dict()
time_res_all = dict()
for user in range(1,usernum+1):
    item_res_all[user] = item_res_train[user] + item_res_valid[user] + item_res_test[user]
    cate_res_all[user] = cate_res_train[user] + cate_res_valid[user] + cate_res_test[user]
    beha_res_all[user] = beha_res_train[user] + beha_res_valid[user] + beha_res_test[user]
    time_res_all[user] = time_res_train[user] + time_res_valid[user] + time_res_test[user]
time_point = generate_timepointfromtime(time_res_all, args.time_bucket)

try:
    relation_matrix = pickle.load(open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctrelationmatrix/%s%d_train.pickle'%(args.datasets, args.time_bucket),'rb'))
    print("train ok")
except Exception as e:
    print(e.args)
    relation_matrix = lastGeneral_Relationfromtime(time_res_train, usernum, time_point)
    pickle.dump(relation_matrix, open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctrelationmatrix/%s%d_train.pickle'%(args.datasets, args.time_bucket),'wb'))

try:
    relation_matrixall = pickle.load(open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctrelationmatrix/%s%d_all.pickle'%(args.datasets, args.time_bucket),'rb'))
    print("eval ok")
except Exception as e:
    print(e.args)
    relation_matrixall = lastGeneral_Relationfromtime(time_res_all, usernum, time_point)
    pickle.dump(relation_matrixall, open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctrelationmatrix/%s%d_all.pickle'%(args.datasets, args.time_bucket),'wb'))


et = datetime.datetime.now()
print("duration ", et-st)
zzzdataloader = lastslow_newDataloader()
try:
    print('try')
    with open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctdataloader/new%s%d_time%d_trainslowsplit%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    zzzdataloader.load(wswe, args)
    # print('batchsize',zzzdataloader.batchsize)
except Exception as e:
    print(e.args)
    print('init')
    zzzdataloader.init(item_res_train, cate_res_train, beha_res_train, relation_matrix, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen, args.time_span)

    with open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctdataloader/new%s%d_time%d_trainslowsplit%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"wb") as f:
        pickle.dump(zzzdataloader.dataset, f)



zzzevaldataloader = lastslow_newevalDataloader()
try:
    print('try')
    with open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctdataloader/new%s%d_time%d_evalslowsplit%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    zzzevaldataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init')
    zzzevaldataloader.init(item_res_all, cate_res_all, beha_res_all, item_res_test, relation_matrixall, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen, args.time_span)
    with open('/dev-data/zzz/207/multibehavior/dataset/dataprocess/dataprocess/correctdataloader/new%s%d_time%d_evalslowsplit%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"wb") as f:
            pickle.dump(zzzevaldataloader.dataset, f)





[usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors] = zip(*zzzdataloader.dataset)

traindataset = mydataset(usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors, args)

[usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors,itemyseqtensors, cateyseqtensors] = zip(*zzzevaldataloader.dataset)

testdataset = mydataset(usetensors, list(itemytensors), list(cateytensors), behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors,  cateyseqtensors,  args)


mydata = DataModule(traindataset, testdataset, args)

class GatedLongRecSimplest(nn.Module):
    def __init__(self,  input_size, hidden_size=args.hidden_size, output_size=args.hidden_size, embedding_dim=args.hidden_size, cate_input_size=1113, cate_output_size=0, cate_embedding_dim=args.hidden_size, cate_hidden_size=args.hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.2, dropout_input=0, use_cuda=False, shared_embedding=True, cate_shared_embedding=True):
        
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



class coRecModel(pl.LightningModule):
    # def __init__(self, gatedmodel
    #     ):
    #     super().__init__()
    #     self.m_teacher_forcing_flag = True
    #     self.m_logsoftmax = nn.LogSoftmax(dim=1)
    #     self.m_cate_loss_func = nn.CrossEntropyLoss()
    #     self.model =  gatedmodel
    #     self.train_epoch = 0
    #     self.max_cate_epoch = 0
    def __init__(self, input_size, cate_input_size
        ):
        super().__init__()
        self.m_teacher_forcing_flag = True
        self.m_logsoftmax = nn.LogSoftmax(dim=1)
        self.m_cate_loss_func = nn.CrossEntropyLoss()
        self.model =  GatedLongRecSimplest(input_size=input_size, cate_input_size=cate_input_size)
        self.train_epoch = 0
        self.max_cate_epoch = 0

    def forward(self, input_ids, b_seq):
        return self.backbone(input_ids, b_seq)
        

    def training_step(self, batch, batch_idx):

        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch
        self.train_epoch+=1
        if self.train_epoch > 5:
            self.m_teacher_forcing_flag = False
        item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, catey_seq_batch = item_y_batch.squeeze(0), cate_y_batch.squeeze(0), beha_y_batch.squeeze(0), item_seq_batch.squeeze(0), cate_seq_batch.squeeze(0), beha_seq_batch.squeeze(0), time_seq_batch.squeeze(0), pad_incateitem_seq_batch.squeeze(0), pad_incatetime_seq_batch.squeeze(0), pad_cateset_seq_batch.squeeze(0), catey_seq_batch.squeeze(0)

        ### negative samples
        # sample_values = self.m_sampler.sample(self.m_nsampled, np.array(item_y_batch).reshape(batch_size))
        # sample_ids, true_freq, sample_freq = sample_values
        item_idx = []
        for _ in range(1000):
            t = np.random.randint(1, itemnum + 1)
            while t in item_y_batch: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        item_test = np.repeat(np.array(item_idx).reshape(1,-1), args.batch_size, 0)
        #
        # print(item_y_batch.shape)
        # print(item_test.shape)

        all = torch.cat((item_y_batch.reshape(-1,1), torch.from_numpy(item_test).to('cuda')),dim=1)
        all_embs = self.model.m_itemNN.look_up(all)
        

        item_idx = np.arange(0,itemnum+1)
        item_test1 = torch.from_numpy(item_idx).to('cuda')
        test_embs = self.model.m_itemNN.look_up(item_test1)

        seq_cate_short_input = self.model.m_cateNN(cate_seq_batch, -1)
        logit_cate_short = self.model.m_cateNN.m_cate_h2o(seq_cate_short_input)

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
            sampled_logit_batch = torch.einsum('bd,bcd->bc',fc_output,all_embs)

            sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch)
            pred_item_prob = sampled_prob_batch

            test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)
            test_prob_batch = self.m_logsoftmax(test_logit_batch)
            pred_item_prob1 = test_prob_batch 
        
        else:
            log_prob_cate_short = self.m_logsoftmax(logit_cate_short)
            log_prob_cate_short, pred_cate_index = torch.topk(log_prob_cate_short, 5, dim=-1)

            pred_cate_index = pred_cate_index.detach()
            log_prob_cate_short = log_prob_cate_short

            for beam_index in range(5):
                pred_cate_beam = pred_cate_index[:, beam_index]
                prob_cate_beam = log_prob_cate_short[:, beam_index]
                #only in itemNN x_long_cate_action is used
                seq_cate_input, seq_short_input = self.model.m_itemNN(pad_incateitem_seq_batch.reshape(pad_incateitem_seq_batch.shape[0]*pad_incateitem_seq_batch.shape[1],-1), pad_cateset_seq_batch,  0, item_seq_batch,  0, pred_cate_beam)

                mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
                fc_output = self.model.fc(mixture_output)
                #sampled_logit_batch [256,10001]  sampled_target_batch torch.zeros(256)
                # sampled_logit_batch, sampled_target_batch = self.model.m_ss(fc_output, torch.from_numpy(np.array(item_y_batch)).to('cuda'),  sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match)
                sampled_logit_batch = torch.einsum('bd,bcd->bc', fc_output, all_embs)
                sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch)


                if pred_item_prob is None:
                    pred_item_prob = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob = pred_item_prob.unsqueeze(-1)
                else:
                    pred_item_prob_beam = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob_beam = pred_item_prob_beam.unsqueeze(-1)
                    pred_item_prob = torch.cat((pred_item_prob, pred_item_prob_beam), dim=-1)
            

                test_logit_batch = torch.einsum('bd, cd->bc', fc_output, test_embs)

                test_prob_batch = self.m_logsoftmax(test_logit_batch)
                if pred_item_prob1 is None:
                    pred_item_prob1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob1 = pred_item_prob1.unsqueeze(-1)
                else:
                    pred_item_prob_beam1 = test_prob_batch+prob_cate_beam.reshape(-1, 1)
                    pred_item_prob_beam1 = pred_item_prob_beam1.unsqueeze(-1)
                    pred_item_prob1 = torch.cat((pred_item_prob1, pred_item_prob_beam1), dim=-1)
            pred_item_prob = torch.logsumexp(pred_item_prob, dim=-1)
            pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)


        #在teaching flag true时，sampled_target_batch 全为0
        #因为真值为第一个，因此sampled_target_batch全为0，也就是第一个
        # 输入为prob时，与输入logit，损失函数不同
        # item_loss = -torch.mean(pred_item_prob[:, 0])
        valid_labels = torch.zeros((pred_item_prob.shape[0])).long().to('cuda')
        item_loss = self.m_cate_loss_func(pred_item_prob , valid_labels)
        cate_loss = self.m_cate_loss_func(logit_cate_short, cate_y_batch)
        loss = 0.8*item_loss + 0.2*cate_loss 
        
        # loss = self.loss(pred_item_prob, valid_labels)
        # loss = loss.unsqueeze(0)
        # recall_5, mrr_5 = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(item_y_batch).to('cuda'), k=5)
        # cate_recall, cate_mrr, cate_ndcg = evaluate_part(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=5)

        return {'loss':loss}

        
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)
        # train_recall_5 = torch.stack([o['recall_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_recall_5', train_recall_5, on_epoch=True)
        # train_mrr_5 = torch.stack([o['mrr_5'] for o in training_step_outputs], 0).mean()
        # self.log('train_mrr_5', train_mrr_5, on_epoch=True)
        # cate_recall = torch.stack([o['cate_recall'] for o in training_step_outputs], 0).mean()
        # self.log('cate_recall', cate_recall, on_epoch=True)
        # if cate_recall > self.max_cate_epoch:
        #     self.max_cate_epoch = cate_recall
        # else:
        #     self.m_teacher_forcing_flag = False
        #     self.log('epoch', self.train_epoch)

            
    def validation_step(self, batch, batch_idx):

        user_seq_batch, item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, itemy_seq_batch, catey_seq_batch = batch
        item_y_batch, cate_y_batch, beha_y_batch, item_seq_batch, cate_seq_batch, beha_seq_batch, time_seq_batch, pad_incateitem_seq_batch, pad_incatetime_seq_batch, pad_cateset_seq_batch, catey_seq_batch = item_y_batch.squeeze(0), cate_y_batch.squeeze(0), beha_y_batch.squeeze(0), item_seq_batch.squeeze(0), cate_seq_batch.squeeze(0), beha_seq_batch.squeeze(0), time_seq_batch.squeeze(0), pad_incateitem_seq_batch.squeeze(0), pad_incatetime_seq_batch.squeeze(0), pad_cateset_seq_batch.squeeze(0), catey_seq_batch.squeeze(0)

        ### negative samples
        # sample_values = self.m_sampler.sample(self.m_nsampled, np.array(item_y_batch).reshape(batch_size))
        # sample_ids, true_freq, sample_freq = sample_values
        item_idx = []
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in item_y_batch: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        item_test = np.repeat(np.array(item_idx).reshape(1,-1), args.batch_size, 0)
        all = torch.cat((item_y_batch.reshape(-1,1), torch.from_numpy(item_test).to('cuda')),dim=1)
        part_embs = self.model.m_itemNN.look_up(all)




        item_idx = np.arange(0, itemnum+1)
        item_test = torch.from_numpy(item_idx).to('cuda')
        test_embs = self.model.m_itemNN.look_up(item_test)



        seq_cate_short_input = self.model.m_cateNN(cate_seq_batch, -1)

        logit_cate_short = self.model.m_cateNN.m_cate_h2o(seq_cate_short_input)

        pred_item_prob = None
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


            sampled_logit_batch = torch.einsum('bd,bcd->bc', fc_output, part_embs)
            sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch)
            # starsampled_logit_batch = torch.einsum('bd,bcd->bc',star_output,part_embs)
            # sampled_prob_batch = self.m_logsoftmax(sampled_logit_batch+ starsampled_logit_batch)


            if pred_item_prob is None:
                pred_item_prob = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob = pred_item_prob.unsqueeze(-1)
            else:
                pred_item_prob_beam = sampled_prob_batch+prob_cate_beam.reshape(-1, 1)
                pred_item_prob_beam = pred_item_prob_beam.unsqueeze(-1)
                pred_item_prob = torch.cat((pred_item_prob, pred_item_prob_beam), dim=-1)
        

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
        
        
        pred_item_prob = torch.logsumexp(pred_item_prob, dim=-1)
        pred_item_prob1 = torch.logsumexp(pred_item_prob1, dim=-1)



        valid_labels = torch.zeros((pred_item_prob.shape[0])).long().to('cuda')


        partmetrics = evaluate(pred_item_prob, valid_labels, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)

        metrics = evaluate(pred_item_prob1, item_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)

        #cate 只有十多个
        # catemetrics = evaluate(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), recalls=args.recall)
        cate_recall_5, cate_mrr, cate_ndcg = evaluate_part(logit_cate_short, cate_y_batch, torch.tensor([True]).expand_as(valid_labels).to('cuda'), k=5)



        # loss = self.loss(valid_logits, valid_labels)


        return {'metrics':metrics, 'partmetrics':partmetrics,  'validcate_recall_5':cate_recall_5}
    
    
    # def validation_epoch_end(self, validation_step_outputs):
    #     valid_recall = torch.stack([o['valid_recall'] for o in validation_step_outputs], 0).mean()
    #     self.log('valid_recall', valid_recall, on_epoch=True)
    #     valid_mrr = torch.stack([o['valid_mrr'] for o in validation_step_outputs], 0).mean()
    #     self.log('valid_mrr', valid_mrr, on_epoch=True)

    #     valid_recall = torch.stack([o['partvalid_recall'] for o in validation_step_outputs], 0).mean()
    #     self.log('partvalid_recall', valid_recall, on_epoch=True)
    #     valid_mrr = torch.stack([o['partvalid_mrr'] for o in validation_step_outputs], 0).mean()
    #     self.log('partvalid_mrr', valid_mrr, on_epoch=True)

    #     validcate_recall = torch.stack([o['validcate_recall'] for o in validation_step_outputs], 0).mean()
    #     self.log('validcate_recall', validcate_recall, on_epoch=True)
        
    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0]['metrics'].keys()
        partkeys = validation_step_outputs[0]['partmetrics'].keys()
        # catekeys = validation_step_outputs[0]['catemetrics'].keys()
        metric =  defaultdict(list)
        for o in validation_step_outputs:
            for k in keys:
                metric[k].append(o['metrics'][k])
            # for k in partkeys:
                metric['part'+k].append(o['partmetrics'][k])
            # for k in catekeys:
                # metric['cate'+k].append(o['catemetrics'][k])
        for k in keys:
            self.log(f'Val:{k}', torch.Tensor(metric[k]).mean(), on_epoch=True)
            self.log(f'partVal:{k}', torch.Tensor(metric['part'+k]).mean(), on_epoch=True)
            # self.log(f'cateVal:{k}', torch.Tensor(metric['cate'+k]).mean(), on_epoch=True)
        validcate_recall_5 = torch.stack([o['validcate_recall_5'] for o in validation_step_outputs], 0).mean()
        self.log('validcate_recall_5', validcate_recall_5, on_epoch=True)

        
        # for k in keys:
        #     tmp = []
        #     for o in validation_step_outputs:
        #         tmp.append(o[k])
        #     self.log(f'Val:{k}', torch.Tensor(tmp).mean())
        # keys = validation_step_outputs[0]['metrics'].keys()
        # for k in keys:
        #     tmp = []
        #     for o in validation_step_outputs:
        #         tmp.append(o[k])
        #     self.log(f'Val:{k}', torch.Tensor(tmp).mean())
        # keys = validation_step_outputs[0]['metrics'].keys()
        # for k in keys:
        #     tmp = []
        #     for o in validation_step_outputs:
        #         tmp.append(o[k])
        #     self.log(f'Val:{k}', torch.Tensor(tmp).mean())
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=.001,
        #                                    weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.98))
        return optimizer

if __name__ =='__main__':

    # #version1
    # model = GatedLongRecSimplest(input_size=itemnum, cate_input_size=catenum)
    # recmodel = coRecModel(model)
    
    # #version2
    # model = GatedLongRecSimplest(input_size=itemnum, cate_input_size=catenum)
    recmodel = coRecModel(input_size=itemnum, cate_input_size=catenum)
#之前的好像没有带teach！

    trainer = pl.Trainer(devices=[args.cuda], accelerator="gpu", accumulate_grad_batches=1, max_epochs=100,  default_root_dir='./log/{}/cocorec/teach/recall{}/lr{}/beamsize'.format(args.datasets,args.recall,args.lr, args.beam_size))
    trainer.fit(model=recmodel, datamodule=mydata)