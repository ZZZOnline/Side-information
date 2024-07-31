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
        
        self.cate_emb1 = torch.nn.Embedding(catenum+1, args.hidden_units, padding_idx=0)
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
        # seqs = self.item_emb(log_seqs)

        # beha = self.beha_emb(beha_seqs)

        # seqs = iseqs
        Q = item_seqs +fea_seqs


        # Q *= self.item_emb.embedding_dim ** 0.5

        # positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        # seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        Q = self.emb_dropout(Q)



        timeline_mask = mask
        Q *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = Q.shape[1] # time dim len for enforce causality
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

            Q = iseqs + fea_seqs

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
    def __init__(self, itemmodel,  args
        ):
        super().__init__()

        self.args = args
        self.itemmodel = itemmodel


        # self.train_epoch = 0
        # self.dropout = torch.nn.Dropout(p=args.dropout_rate)

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

        interaction = batch

        # item_seq_batch = interaction['item_id_list']
        # cate_seq_batch = self.itemmodel.cateall[item_seq_batch]
        # item_y_batch = interaction['item_id']

        loss = self.itemmodel.calculate_loss(interaction)
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

        

        interaction = batch

        # print(interaction['item_id'], interaction['item_id'].shape)

        scores = self.itemmodel.full_sort_predict(interaction)
    
        valid_labels = torch.ones((scores.shape[0])).long().to(self.args.device)


        metrics = evaluate(scores, interaction['item_id'], torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=self.args.recall)


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

        interaction = batch

        scores = self.itemmodel.full_sort_predict(interaction)
    
        valid_labels = torch.ones((scores.shape[0])).long().to(self.args.device)


        metrics = evaluate(scores, interaction['item_id'], torch.tensor([True]).expand_as(valid_labels).to(self.args.device), recalls=self.args.recall)


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
        # print('trainer learning rate:', self.args.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        return optimizer
