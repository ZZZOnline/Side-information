import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import *
import sys
import os
import datetime
import pickle

from new_utils import *
from network import *
from plmodel import *






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
parser.add_argument('--datasets',  default='sub_taobao', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout_rate', default=0.25, type=float)
parser.add_argument("--embedding_dim", type=int, default=64,
                     help="using embedding")
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--incatemaxlen', type=int, default=10)


parser.add_argument('--lastk', type=int, default=5)
parser.add_argument('--lr', default=.001, type=float)
parser.add_argument('--l2_emb', default=0.005, type=float)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--model', type=str, default='casr')


parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=2, type=int)


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










args = parser.parse_args()

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
    User, time_map, bmap = beha_dataall_partition_last(args.dataset, 'buy')
else:
    print('wrong datasets')


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


directory = "./dataset/%s/"%(args.datasets)
if not os.path.exists(directory):
    os.makedirs(directory)
    print("directory created")


try:
    relation_matrix = pickle.load(open('./dataset/%s/timematrix_%d_train.pickle'%(args.datasets, args.time_bucket),'rb'))
    print("train ok")
except Exception as e:
    print(e.args)
    relation_matrix = lastGeneral_Relationfromtime(time_res_train, usernum, time_point)
    pickle.dump(relation_matrix, open('./dataset/%s/timematrix_%d_train.pickle'%(args.datasets, args.time_bucket),'wb'))

try:
    relation_matrixall = pickle.load(open('./dataset/%s/timematrix_%d_all.pickle'%(args.datasets, args.time_bucket),'rb'))
    print("eval ok")
except Exception as e:
    print(e.args)
    relation_matrixall = lastGeneral_Relationfromtime(time_res_all, usernum, time_point)
    pickle.dump(relation_matrixall, open('./dataset/%s/timematrix_%d_all.pickle'%(args.datasets, args.time_bucket),'wb'))


et = datetime.datetime.now()
print("duration ", et-st)
dataloader = lastslow_newDataloader()
try:
    print('try')
    with open('./dataset/%s/dataloader_%d_time%d_train%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    dataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init1')
    dataloader.init(item_res_train, cate_res_train, beha_res_train, relation_matrix, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen, args.time_span)

    with open('./dataset/%s/dataloader_%d_time%d_train%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"wb") as f:
        pickle.dump(dataloader.dataset, f)



evaldataloader = lastslow_newevalDataloader()
try:
    print('try')
    with open('./dataset/%s/dataloader_%d_time%d_test%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"rb") as f:
        wswe= pickle.load(f)
    evaldataloader.load(wswe, args)

except Exception as e:
    print(e.args)
    print('init2')
    evaldataloader.init(item_res_all, cate_res_all, beha_res_all, item_res_test, relation_matrixall, usernum, args.batch_size, 5, args.maxlen, args.incatemaxlen, args.time_span)
    with open('./dataset/%s/dataloader_%d_time%d_test%d.pickle'%(args.datasets, args.maxlen, args.time_bucket, args.batch_size),"wb") as f:
            pickle.dump(evaldataloader.dataset, f)





[usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors] = zip(*dataloader.dataset)

traindataset = mydataset(usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors, cateyseqtensors, args)

[usetensors, itemytensors, cateytensors, behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors,itemyseqtensors, cateyseqtensors] = zip(*evaldataloader.dataset)

evaldataset = mydataset(usetensors, list(itemytensors), list(cateytensors), behaytensors, itemseqtensors, cateseqtensors, behaseqtensors, timeseqtensors, padincateitemtensors, padincatetimetensors, padcatesettensors, itemyseqtensors,  cateyseqtensors,  args)


mydata = DataModule(traindataset, evaldataset, args)



imap, imaps = catemap(args.dataset)
args.imap = imap



if args.model =='casr':
    
    itemmodel = cudaSASRec( itemnum, args)
    catemodel = cudatqSASRec( catenum, args )
    recmodel = casrmodel(itemmodel, catemodel, args)


trainer = pl.Trainer(devices=[args.cuda], accelerator="gpu", accumulate_grad_batches=1, max_epochs=args.num_epochs,  default_root_dir='./script/{}/{}//beamsize{}/lr{}'.format(args.model, args.datasets, args.beam_size,  args.lr))
trainer.fit(recmodel, mydata)


