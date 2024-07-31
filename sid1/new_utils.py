
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils.data import *
import pytorch_lightning as pl

#replace timestamps with time intervals
def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: 
        time_map[time] = int(round(float(time-time_min)))
    return time_map


def interval_split(timearray, k):

    partnum = np.sum(timearray) // k
    time_point = np.zeros((k,), dtype=int)
    interactiionsum = 0
    count = 0

    for i in range(len(timearray)):
        if count>=k:
            break
        interactiionsum+=timearray[i]
        if interactiionsum < count*partnum:
            continue
        elif interactiionsum >= count*partnum:
            time_point[count] = i
            count+=1
    return time_point


# return k timepoints for generating time buckets; time intervals greater than 200000s are grouped together 
def generate_timepointfromtime(usertime_seq, k):
    timeinterval_total = []
    for user, items in usertime_seq.items():
        timelist = []
        for index in range(len(items)-2):
            timeinterval = items[index+1] - items[index]
            timelist.append(timeinterval)
        timeinterval_total.append(timelist)
    timedict = np.zeros((200001,))
    for i in timeinterval_total:
        for j in i:
            if j < 200000:
                timedict[j]+=1  
            else:
                timedict[200000]+=1

    timepoint = interval_split(timedict, k)
    return timepoint




#return time bucket for any time intervals
def timegeneralize(timeinterval, timedict):
    for i in range(len(timedict)):
        if timeinterval<timedict[i]:
            return i
    return len(timedict)

    

#return time bucket sequence with time interval sequence input
def lastcomputeGePos(time_seq, time_point):
    size = time_seq.shape[0]
    time_outseq = np.zeros([size], dtype=np.int32)
    for i in range(size):
            span = abs(time_seq[i]-time_seq[-1])
            time_outseq[i] = timegeneralize(span, time_point)
    return time_outseq


# return time bucket sequence of users
def lastGeneral_Relationfromtime(usertime_seq, usernum,  time_point):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.array(usertime_seq[user])
        data_train[user] = lastcomputeGePos(time_seq, time_point)
    return data_train




#generate datasets including behavior with the last sequence
def beha_dataall_partition_last(fname, target_beha):

    User = defaultdict(list)

    
    print('Preparing data...')
    f = open('%s'%fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    cate_count = defaultdict(int)
    beha_set = set()
    
    for line in f:
        try:
            u, i, c, timestamp, beha = line.rstrip().split(',')
        #when no \t is generated, in a naturall csv file ',' is the delimiter
        except:
            u, i, c, timestamp, beha = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)
        b = beha
        user_count[u]+=1
        item_count[i]+=1
        cate_count[c]+=1
        beha_set.add(b)
    bmap = dict()
    
    for i,b in enumerate(beha_set):
        if b in bmap.keys():
            continue
        else:
            bmap[b] = i+1
    for key, value in bmap.items():
        if value == 1 and key != target_beha:
            bmap[key] = 0
            replaced_key = key
            new_value = bmap["%s"%target_beha]
            bmap["%s"%target_beha] = 1
            bmap["%s"%replaced_key] = new_value
            break

    f.close()

    f = open('%s'%fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, c, timestamp, beha = line.rstrip().split(',')
        #when no \t is generated, in a naturall csv file ',' is the delimiter
        except:
            u, i, c, timestamp, beha = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)
        b = bmap[beha]

        timestamp = float(timestamp)

        time_set.add(timestamp)
        
        User[u].append([i, timestamp, c, b])
    f.close()
    time_map = timeSlice(time_set)


    return User, time_map, bmap

#generate datasets including behavior with the last sequence
def dataall_partition_last(fname):

    User = defaultdict(list)

    
    print('Preparing data...')
    f = open('%s'%fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    cate_count = defaultdict(int)
    beha_set = set()
    
    for line in f:
        try:
            u, i, c, timestamp, beha = line.rstrip().split(',')
        #when no \t is generated, in a naturall csv file ',' is the delimiter
        except:
            u, i, c, timestamp, beha = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)
        b = beha
        user_count[u]+=1
        item_count[i]+=1
        cate_count[c]+=1
        beha_set.add(b)

    f.close()

    f = open('%s'%fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, c, timestamp, beha = line.rstrip().split(',')
        #when no \t is generated, in a naturall csv file ',' is the delimiter
        except:
            u, i, c, timestamp, beha = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)

        # timestamp = float(timestamp)

        # time_set.add(timestamp)
        
        User[u].append([i, c])
    f.close()
    # time_map = timeSlice(time_set)


    return User




def beha_datatimenormalizelast(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    cate_set = set()
    time_max = set()
    #time normalize
    User_train = defaultdict(list)
    User_valid = defaultdict(list)
    User_test = defaultdict(list)
    for user, items in User.items():
        User_train[user]=items[:-2]
        User_valid[user]=[items[-2]]
        User_test[user]=[items[-1]]

    for user, items in User.items():
        time_list = list(map(lambda x: time_map[x[1]], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User[user] = list(map(lambda x: [x[0], int(round((time_map[x[1]]-time_min)/time_scale)+1), x[2], x[3]], items))
        User_train[user] = list(map(lambda x: [x[0], int(round((time_map[x[1]]-time_min)/time_scale)+1), x[2], x[3]], User_train[user]))
        User_valid[user] = list(map(lambda x: [x[0], int(round((time_map[x[1]]-time_min)/time_scale)+1), x[2], x[3]], User_valid[user]))
        User_test[user] = list(map(lambda x: [x[0], int(round((time_map[x[1]]-time_min)/time_scale)+1), x[2], x[3]], User_test[user]))
        time_max.add(max(set(map(lambda x: x[1], User[user]))))


    for user, datas in User.items():#datas:[item,time,cate]
        user_set.add(user)
        User_filted[user] = datas
        for data in datas:
            item_set.add(data[0])
            cate_set.add(data[2])
            

    item_res_train = dict()
    cate_res_train = dict()
    beha_res_train = dict()
    time_res_train = dict()
    # in_cateitem_res_train = defaultdict(dict)
    for user, datas in User_train.items():
        # catelist_train = defaultdict(list)
        item_res_train[user] = list(map(lambda x: x[0], datas))
        cate_res_train[user] = list(map(lambda x: x[2], datas))
        time_res_train[user] = list(map(lambda x: x[1], datas))
        beha_res_train[user] = list(map(lambda x: x[3], datas))
        # for data in datas:#data [item,time,cate]
        #     catelist_train[data[2]].append([data[0],data[1]])
        # in_cateitem_res_train[user] = catelist_train
    item_res_valid = dict()
    cate_res_valid = dict()
    beha_res_valid = dict()
    time_res_valid = dict()
    # in_cateitem_res_valid = defaultdict(dict)
    for user, datas in User_valid.items():
        # catelist_valid = defaultdict(list)
        item_res_valid[user] = list(map(lambda x: x[0], datas))
        cate_res_valid[user] = list(map(lambda x: x[2], datas))
        time_res_valid[user] = list(map(lambda x: x[1], datas))
        beha_res_valid[user] = list(map(lambda x: x[3], datas))
        # for data in datas:#data [item,time,cate]
        #     catelist_valid[data[2]].append([data[0],data[1]])
        # in_cateitem_res_valid[user] = catelist_valid
    item_res_test = dict()
    cate_res_test = dict()
    beha_res_test = dict()
    time_res_test = dict()
    # in_cateitem_res_test = defaultdict(dict)
    for user, datas in User_test.items():
        # catelist_test = defaultdict(list)
        item_res_test[user] = list(map(lambda x: x[0], datas))
        cate_res_test[user] = list(map(lambda x: x[2], datas))
        time_res_test[user] = list(map(lambda x: x[1], datas))
        beha_res_test[user] = list(map(lambda x: x[3], datas))

                
    return item_res_train, cate_res_train,  beha_res_train, time_res_train, item_res_valid, cate_res_valid,  beha_res_valid, time_res_valid, item_res_test, cate_res_test,  beha_res_test, time_res_test, len(user_set), len(item_set), len(cate_set), max(time_max)



def datatimenormalizelast(User):
    User_filted = dict()
    user_set = set()
    item_set = set()
    cate_set = set()
    time_max = set()
    #time normalize
    User_train = defaultdict(list)
    User_valid = defaultdict(list)
    User_test = defaultdict(list)
    for user, items in User.items():
        User_train[user]=items[:-2]
        User_valid[user]=[items[-2]]
        User_test[user]=[items[-1]]



    for user, datas in User.items():#datas:[item,time,cate]
        user_set.add(user)
        User_filted[user] = datas
        for data in datas:
            item_set.add(data[0])
            cate_set.add(data[1])
            

    item_res_train = dict()
    cate_res_train = dict()
    beha_res_train = dict()
    time_res_train = dict()
    # in_cateitem_res_train = defaultdict(dict)
    for user, datas in User_train.items():
        # catelist_train = defaultdict(list)
        item_res_train[user] = list(map(lambda x: x[0], datas))
        cate_res_train[user] = list(map(lambda x: x[1], datas))
        # time_res_train[user] = list(map(lambda x: x[1], datas))
        # beha_res_train[user] = list(map(lambda x: x[3], datas))
        # for data in datas:#data [item,time,cate]
        #     catelist_train[data[2]].append([data[0],data[1]])
        # in_cateitem_res_train[user] = catelist_train
    item_res_valid = dict()
    cate_res_valid = dict()
    beha_res_valid = dict()
    time_res_valid = dict()
    # in_cateitem_res_valid = defaultdict(dict)
    for user, datas in User_valid.items():
        # catelist_valid = defaultdict(list)
        item_res_valid[user] = list(map(lambda x: x[0], datas))
        cate_res_valid[user] = list(map(lambda x: x[1], datas))
        # time_res_valid[user] = list(map(lambda x: x[1], datas))
        # beha_res_valid[user] = list(map(lambda x: x[3], datas))
        # for data in datas:#data [item,time,cate]
        #     catelist_valid[data[2]].append([data[0],data[1]])
        # in_cateitem_res_valid[user] = catelist_valid
    item_res_test = dict()
    cate_res_test = dict()
    beha_res_test = dict()
    time_res_test = dict()
    # in_cateitem_res_test = defaultdict(dict)
    for user, datas in User_test.items():
        # catelist_test = defaultdict(list)
        item_res_test[user] = list(map(lambda x: x[0], datas))
        cate_res_test[user] = list(map(lambda x: x[1], datas))
        # time_res_test[user] = list(map(lambda x: x[1], datas))
        # beha_res_test[user] = list(map(lambda x: x[3], datas))

                
    return item_res_train, cate_res_train, item_res_valid, cate_res_valid, item_res_test, cate_res_test, len(user_set), len(item_set), len(cate_set)


# dataloader for training
class lastslow_newDataloader(object):
    def __init__(self):
        pass

    def init(self, item_sequence, cate_sequence, usernum, batchsize, threshstart, maxlen, incatemaxlen):
        cateset_seq = []
        # time_seq = []
        user_seq = []
        cate_seq = []
        #catey_seq is the y set in sarsrec which is the next seq of cate_seq, 
        catey_seq = []
        item_seq = []
        itemy_seq = []
        # beha_seq = []
        #cate_y is only the n+1 category of the sequence
        cate_y = []
        item_y = []
        # beha_y = []

        incateitem_seq = []
        # incatetime_seq = []
        # self.timespan = timespan
        seqlen = []
        for user in tqdm(range(1,usernum+1)):

            item_user = np.array(item_sequence[user])
            if(len(item_user)< threshstart):
                continue
            cate_user = np.array(cate_sequence[user])
            # beha_user = np.array(beha_sequence[user])
            # time_user = relation_matrix[user]
            cate_set = list(set(cate_user[:]))

            for index in range(len(item_user)-1, len(item_user)):

                if(index < maxlen):
                    new_itemseq = item_user[:index].tolist()
                    new_cateseq = cate_user[:index].tolist()
                    new_cateyseq = cate_user[1:index+1].tolist()
                    new_itemyseq = item_user[1:index+1].tolist()
                    # new_behaseq = beha_user[:index].tolist()
                    # new_timeseq = time_user[:index].tolist()
                else:
                    new_itemseq = item_user[index-maxlen:index].tolist()
                    new_cateseq = cate_user[index-maxlen:index].tolist()
                    new_cateyseq = cate_user[index-maxlen+1:index+1].tolist()
                    new_itemyseq = item_user[index-maxlen+1:index+1].tolist()
                    # new_behaseq = beha_user[index-maxlen:index].tolist()
                    # new_timeseq = time_user[index-maxlen:index].tolist()


                new_incateitemlist = []
                new_incatetimelist = []
                for cateexhaustion in cate_set:

                    incate_index = np.array(np.nonzero(cate_user[:index] ==cateexhaustion)).reshape(-1)
                    if len(incate_index)<incatemaxlen:
                        new_incateitemseq = np.squeeze(item_user[incate_index]).reshape(-1).tolist()
                        # new_incatetimeseq = time_user[incate_index].reshape(-1).tolist()

                    else:
                        new_incateitemseq = np.squeeze(item_user[incate_index]).reshape(-1).tolist()
                        # new_incatetimeseq = time_user[incate_index].reshape(-1).tolist()
                        new_incateitemseq = new_incateitemseq[-incatemaxlen:]
                        # new_incatetimeseq = new_incatetimeseq[-incatemaxlen:]
                    new_incateitemlist.append(new_incateitemseq)
                    # new_incatetimelist.append(new_incatetimeseq)
                seqlen.append(len(new_itemseq))
                item_seq.append(new_itemseq)
                cate_seq.append(new_cateseq)
                catey_seq.append(new_cateyseq)
                itemy_seq.append(new_itemyseq)
                # beha_seq.append(new_behaseq)
                # time_seq.append(new_timeseq)
                incateitem_seq.append(new_incateitemlist)
                # incatetime_seq.append(new_incatetimelist)
                cate_y.append(cate_user[index])
                item_y.append(item_user[index])
                # beha_y.append(beha_user[index])
                user_seq.append(user)
                cateset_seq.append(cate_set)

        self.seqlength = len(user_seq)
        self.batchnum = self.seqlength//batchsize
        self.batchsize = batchsize




        self.temp = sorted(list(zip(seqlen, user_seq, item_y, cate_y, item_seq, cate_seq, incateitem_seq, cateset_seq, itemy_seq, catey_seq )), reverse=False)
        self.seqlen, self.user_seq, self.item_y, self.cate_y, self.item_seq, self.cate_seq, self.incateitem_seq, self.cateset_seq, self.itemy_seq, self.catey_seq = zip(*self.temp)

        batchsize = self.batchsize
        seqlen, user_seq, item_y, cate_y, item_seq, cate_seq, incateitem_seq, cateset_seq, itemy_seq, catey_seq =self.seqlen, self.user_seq, self.item_y, self.cate_y, self.item_seq, self.cate_seq, self.incateitem_seq, self.cateset_seq, self.itemy_seq, self.catey_seq

        self.itemseqtensors = []
        self.itemytensors = []
        self.itemyseqtensors = []

        self.cateseqtensors = []
        self.cateytensors = []
        self.cateyseqtensors = []


        # self.behaseqtensors = []
        # self.behaytensors = []
        # self.timeseqtensors = []
        self.padincateitemtensors = []
        # self.padincatetimetensors = []
        self.padcatesettensors = []
        self.seqlentensors = []
        self.usetensors = []

        for batchidx in range(self.batchnum):
            seqlen_batch = seqlen[batchidx*batchsize:(batchidx+1)*batchsize]
            user_seq_batch = user_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            item_y_batch = item_y[batchidx*batchsize:(batchidx+1)*batchsize]
            cate_y_batch = cate_y[batchidx*batchsize:(batchidx+1)*batchsize]
            itemy_seq_batch = itemy_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            catey_seq_batch = catey_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # beha_y_batch = beha_y[batchidx*batchsize:(batchidx+1)*batchsize]
            item_seq_batch = item_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            cate_seq_batch = cate_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # beha_seq_batch = beha_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # time_seq_batch = time_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            incateitem_seq_batch = incateitem_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # incatetime_seq_batch = incatetime_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            cateset_seq_batch = cateset_seq[batchidx*batchsize:(batchidx+1)*batchsize]

            pad_item_seq_batch = [ [0]*(maxlen-len(i)) + i for i in item_seq_batch ]
            pad_itemy_seq_batch = [ [0]*(maxlen-len(i)) + i for i in itemy_seq_batch ]
            pad_cate_seq_batch = [ [0]*(maxlen-len(i)) + i for i in cate_seq_batch ]
            pad_catey_seq_batch = [ [0]*(maxlen-len(i)) + i for i in catey_seq_batch ]
            # pad_beha_seq_batch = [ [0]*(maxlen-len(i)) + i for i in beha_seq_batch ]
            # pad_time_seq_batch = [ [self.timespan]*(maxlen-len(i)) + i for i in time_seq_batch ]
            item_seq_batch = pad_item_seq_batch
            cate_seq_batch = pad_cate_seq_batch
            # beha_seq_batch = pad_beha_seq_batch
            # time_seq_batch = pad_time_seq_batch
            itemy_seq_batch = pad_itemy_seq_batch
            catey_seq_batch = pad_catey_seq_batch
            
            maxlen_cateset = max(len(i) for i in cateset_seq_batch)


            maxlen_incateseq = max([len(j) for i in incateitem_seq_batch for j in i])


            pad_cateset_seq_batch = [subseq + [-1]*(maxlen_cateset-len(subseq)) for subseq in cateset_seq_batch]
            pad_incateitem_seq_batch = []
            # pad_incatetime_seq_batch = []

            for i in range(len(incateitem_seq_batch)):
                pads = incateitem_seq_batch[i]
                pads += [[0]*maxlen_incateseq for j in range(maxlen_cateset-len(pads))]
                for j in range(len(pads)):
                    pads[j] = [0] *(maxlen_incateseq - len(pads[j])) + pads[j]
                pad_incateitem_seq_batch.append(pads)
                

                # padt = incatetime_seq_batch[i]
                # padt += [[self.timespan]*maxlen_incateseq for j in range(maxlen_cateset-len(padt))]
                # for j in range(len(padt)):
                #     padt[j] = [self.timespan] *(maxlen_incateseq - len(padt[j])) + padt[j]
                # pad_incatetime_seq_batch.append(padt)
            self.itemseqtensors.append(torch.LongTensor(item_seq_batch))
            self.itemytensors.append(torch.LongTensor(item_y_batch))
            self.cateseqtensors.append(torch.LongTensor(cate_seq_batch))
            self.cateytensors.append(torch.LongTensor(cate_y_batch))
            self.itemyseqtensors.append(torch.LongTensor(itemy_seq_batch))
            self.cateyseqtensors.append(torch.LongTensor(catey_seq_batch))
            # self.behaseqtensors.append(torch.LongTensor(beha_seq_batch))
            # self.behaytensors.append(torch.LongTensor(beha_y_batch))
            # self.timeseqtensors.append(torch.LongTensor(time_seq_batch))
            self.padincateitemtensors.append(torch.LongTensor(pad_incateitem_seq_batch))
            # self.padincatetimetensors.append(torch.LongTensor(pad_incatetime_seq_batch))
            self.padcatesettensors.append(torch.LongTensor(pad_cateset_seq_batch))
            self.usetensors.append(user_seq_batch)
            self.seqlentensors.append(seqlen_batch)


        self.dataset = list(zip(self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors))



    def __iter__(self):

        self.seqlength = len(self.usetensors)
        self.batchnum = self.seqlength
        for i in range(self.batchnum):
            yield  self.usetensors[i], self.itemytensors[i], self.cateytensors[i], self.itemseqtensors[i], self.cateseqtensors[i], self.padincateitemtensors[i], self.padcatesettensors[i], self.itemyseqtensors[i], self.cateyseqtensors[i]      
    @classmethod
    def load(self, sstemp, args):

        self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors= zip(* sstemp)
        self.dataset = zip(self.usetensors, self.itemytensors, self.cateytensors,  self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors)

        self.batchsize = args.batch_size
        self.timespan = args.time_span


class lastslow_newevalDataloader(object):
    def __init__(self):
        pass

    def init(self, item_sequence_all, cate_sequence_all, item_sequence_eval, usernum, batchsize, threshstart, maxlen, incatemaxlen):
        time_seq = []
        user_seq = []
        cate_seq = []
        item_seq = []
        beha_seq = []

        cateset_seq = []
        incateitem_seq = []
        incatetime_seq = []
        catey_seq = []
        itemy_seq = []


        user_seq = []
        cate_y = []
        item_y = []
        beha_y = []
        seqlen = []
        threshstart = threshstart
        argsmaxlen = maxlen
        incatemaxlen = incatemaxlen
        # self.timespan = timespan

        for user in tqdm(range(1, usernum+1)):
            item_user =np.array(item_sequence_all[user])
            if(len(item_user)< threshstart):
                continue
            cate_user =np.array(cate_sequence_all[user])
            # beha_user =np.array(beha_sequence_all[user])
            
            # time_user = relation_matrix_all[user]
            cate_set = list(set(cate_user[:]))

            item_eval_user = np.array(item_sequence_eval[user])
            maxlen = len(item_user)-1 if argsmaxlen>len(item_user)  else argsmaxlen-1

            for i in range(1,len(item_eval_user)+1):
    

                new_itemseq = item_user[-i-maxlen:-i].tolist()
                if(len(new_itemseq) < 5):
                    continue
                new_cateseq = cate_user[-i-maxlen:-i].tolist()
                if i ==1:
                    new_cateyseq = cate_user[-i-maxlen+1:].tolist()
                    new_itemyseq = item_user[-i-maxlen+1:].tolist()
                else:
                    new_cateyseq = cate_user[-i-maxlen+1:-i+1].tolist()
                    new_itemyseq = item_user[-i-maxlen+1:-i+1].tolist()

                # new_behaseq = beha_user[-i-maxlen:-i].tolist()
                # new_timeseq = time_user[-i-maxlen:-i].tolist()

                new_incateitemlist = []
                # new_incatetimelist = []


                for cateexhaustion in cate_set:
                    
                    incate_index = np.array(np.nonzero(cate_user[:-i] ==cateexhaustion)).reshape(-1)
                    if len(incate_index)<incatemaxlen:
                        new_incateitemseq = np.squeeze(item_user[incate_index]).reshape(-1).tolist()
                        # new_incatetimeseq = time_user[incate_index].reshape(-1).tolist()

                    else:
                        new_incateitemseq = np.squeeze(item_user[incate_index]).reshape(-1).tolist()
                        # new_incatetimeseq = time_user[incate_index].reshape(-1).tolist()
                        new_incateitemseq = new_incateitemseq[-incatemaxlen:]
                        # new_incatetimeseq = new_incatetimeseq[-incatemaxlen:]
                    new_incateitemlist.append(new_incateitemseq)
                    # new_incatetimelist.append(new_incatetimeseq)
                    
                seqlen.append(len(new_itemseq))
                item_seq.append(new_itemseq)
                cate_seq.append(new_cateseq)
                catey_seq.append(new_cateyseq)
                itemy_seq.append(new_itemyseq)
                # beha_seq.append(new_behaseq)
                # time_seq.append(new_timeseq)

                cateset_seq.append(cate_set)
                incateitem_seq.append(new_incateitemlist)
                # incatetime_seq.append(new_incatetimelist)


                cate_y.append(cate_user[-i])
                item_y.append(item_user[-i])
                # beha_y.append(beha_user[-i])
                user_seq.append(user)
        self.seqlength = len(user_seq)
        self.batchnum = self.seqlength//batchsize
        self.batchsize = batchsize

        self.temp = sorted(list(zip(seqlen, user_seq, item_y, cate_y, item_seq, cate_seq, incateitem_seq, cateset_seq, itemy_seq, catey_seq )), reverse=True)
        self.seqlen, self.user_seq, self.item_y, self.cate_y, self.item_seq, self.cate_seq, self.incateitem_seq, self.cateset_seq, self.itemy_seq, self.catey_seq = zip(*self.temp)



        
        batchsize = self.batchsize
        seqlen, user_seq, item_y, cate_y, item_seq, cate_seq, incateitem_seq, cateset_seq, itemy_seq, catey_seq =self.seqlen, self.user_seq, self.item_y, self.cate_y, self.item_seq, self.cate_seq, self.incateitem_seq, self.cateset_seq, self.itemy_seq, self.catey_seq

        self.itemseqtensors = []
        self.itemytensors = []
        self.cateseqtensors = []
        self.cateyseqtensors = []
        self.itemyseqtensors = []
        self.cateytensors = []
        # self.behaseqtensors = []
        # self.behaytensors = []
        # self.timeseqtensors = []
        self.padincateitemtensors = []
        # self.padincatetimetensors = []
        self.padcatesettensors = []
        self.seqlentensors = []
        self.usetensors = []

        for batchidx in range(self.batchnum):
            seqlen_batch = seqlen[batchidx*batchsize:(batchidx+1)*batchsize]
            user_seq_batch = user_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            item_y_batch = item_y[batchidx*batchsize:(batchidx+1)*batchsize]
            cate_y_batch = cate_y[batchidx*batchsize:(batchidx+1)*batchsize]
            # beha_y_batch = beha_y[batchidx*batchsize:(batchidx+1)*batchsize]
            item_seq_batch = item_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            cate_seq_batch = cate_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            catey_seq_batch = catey_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            itemy_seq_batch = itemy_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # beha_seq_batch = beha_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # time_seq_batch = time_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            incateitem_seq_batch = incateitem_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            # incatetime_seq_batch = incatetime_seq[batchidx*batchsize:(batchidx+1)*batchsize]
            cateset_seq_batch = cateset_seq[batchidx*batchsize:(batchidx+1)*batchsize]


            


            pad_item_seq_batch = [ [0]*(argsmaxlen-len(i)) + i for i in item_seq_batch ]
            pad_cate_seq_batch = [ [0]*(argsmaxlen-len(i)) + i for i in cate_seq_batch ]
            pad_catey_seq_batch = [ [0]*(argsmaxlen-len(i)) + i for i in catey_seq_batch ]
            pad_itemy_seq_batch = [ [0]*(argsmaxlen-len(i)) + i for i in itemy_seq_batch ]
            # pad_beha_seq_batch = [ [0]*(argsmaxlen-len(i)) + i for i in beha_seq_batch ]
            # pad_time_seq_batch = [ [self.timespan]*(argsmaxlen-len(i)) + i for i in time_seq_batch ]
            item_seq_batch = pad_item_seq_batch
            cate_seq_batch = pad_cate_seq_batch
            # beha_seq_batch = pad_beha_seq_batch
            # time_seq_batch = pad_time_seq_batch
            catey_seq_batch = pad_catey_seq_batch
            itemy_seq_batch = pad_itemy_seq_batch
            
            maxlen_cateset = max(len(i) for i in cateset_seq_batch)


            maxlen_incateseq = max([len(j) for i in incateitem_seq_batch for j in i])


            pad_cateset_seq_batch = [subseq + [-1]*(maxlen_cateset-len(subseq)) for subseq in cateset_seq_batch]
            pad_incateitem_seq_batch = []
            pad_incatetime_seq_batch = []
            for i in range(len(incateitem_seq_batch)):
                pads = incateitem_seq_batch[i]
                pads += [[0]*maxlen_incateseq for j in range(maxlen_cateset-len(pads))]
                for j in range(len(pads)):
                    pads[j] = [0] *(maxlen_incateseq - len(pads[j])) + pads[j]
                pad_incateitem_seq_batch.append(pads)



            self.itemseqtensors.append(torch.LongTensor(item_seq_batch))
            self.itemytensors.append(torch.LongTensor(item_y_batch))
            self.cateseqtensors.append(torch.LongTensor(cate_seq_batch))
            self.cateyseqtensors.append(torch.LongTensor(catey_seq_batch))
            self.itemyseqtensors.append(torch.LongTensor(itemy_seq_batch))
            self.cateytensors.append(torch.LongTensor(cate_y_batch))
            # self.behaseqtensors.append(torch.LongTensor(beha_seq_batch))
            # self.behaytensors.append(torch.LongTensor(beha_y_batch))
            # self.timeseqtensors.append(torch.LongTensor(time_seq_batch))
            self.padincateitemtensors.append(torch.LongTensor(pad_incateitem_seq_batch))
            # self.padincatetimetensors.append(torch.LongTensor(pad_incatetime_seq_batch))
            self.padcatesettensors.append(torch.LongTensor(pad_cateset_seq_batch))
            self.usetensors.append(user_seq_batch)
            self.seqlentensors.append(seqlen_batch)

        self.dataset = zip(self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors)


    def __iter__(self):
        self.seqlength = len(self.usetensors)
        self.batchnum = self.seqlength
        for i in range(self.batchnum):
            yield  self.usetensors[i], self.itemytensors[i], self.cateytensors[i], self.itemseqtensors[i], self.cateseqtensors[i], self.padincateitemtensors[i], self.padcatesettensors[i], self.itemyseqtensors[i], self.cateyseqtensors[i]   
    @classmethod
    def load(self, sstemp, args):

        self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors,  self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors = zip(* sstemp)
        self.dataset = zip(self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors, self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors)
        self.batchsize = args.batch_size
        self.timespan = args.time_span


class mydataset(torch.utils.data.Dataset):
    def __init__(self, usetensors, itemytensors, cateytensors, itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors, itemyseqtensors, cateyseqtensors, args):   

        self.usetensors, self.itemytensors, self.cateytensors, self.itemseqtensors, self.cateseqtensors, self.padincateitemtensors,self.padcatesettensors, self.itemyseqtensors, self.cateyseqtensors = usetensors, itemytensors, cateytensors, itemseqtensors, cateseqtensors, padincateitemtensors, padcatesettensors, itemyseqtensors, cateyseqtensors
        self.args = args


    def __len__(self):
        return len(self.usetensors)

    def __getitem__(self, idx):
        return self.usetensors[idx], self.itemytensors[idx], self.cateytensors[idx], self.itemseqtensors[idx], self.cateseqtensors[idx], self.padincateitemtensors[idx], self.padcatesettensors[idx], self.itemyseqtensors[idx], self.cateyseqtensors[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, traindataset, evaldataset, testdataset, args):
        super().__init__()
        self.traindataset = traindataset
        self.evaldataset = evaldataset
        self.testdataset = testdataset
        self.args = args

    def train_dataloader(self):
        return DataLoader(self.traindataset, 1, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.evaldataset, 1, num_workers=0, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.testdataset, 1, num_workers=0, persistent_workers=False)





def catemap(fname):

    f = open('%s'%fname, 'r')


    user_count = defaultdict(int)
    item_count = defaultdict(int)
    cate_count = defaultdict(int)
    beha_set = set()
    imap = defaultdict(int)
    imaps = defaultdict(list)
    for line in f:
        try:
            u, i, c, timestamp, beha = line.rstrip().split(',')
        except:
            u, i, c, timestamp, beha = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)
        if c not in imaps[i]:
            imaps[i].append(c)

        
        imap[i] = c
        b = beha
        user_count[u]+=1
        item_count[i]+=1
        cate_count[c]+=1
        beha_set.add(b)


    f.close()
    return imap, imaps

