# import numpy as np
# import torch
# import torch.nn
# from torch import nn
# import math

# #based on SASRec official implementation
# class PointWiseFeedForward(torch.nn.Module):
#     def __init__(self, hidden_units, dropout_rate):

#         super(PointWiseFeedForward, self).__init__()

#         self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
#         self.dropout1 = torch.nn.Dropout(p=dropout_rate)
#         self.relu = torch.nn.ReLU()
#         self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
#         self.dropout2 = torch.nn.Dropout(p=dropout_rate)

#     def forward(self, inputs):
#         outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
#         outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
#         outputs += inputs
#         return outputs



# #self-attention module
# class cudaSASRec(torch.nn.Module):
#     def __init__(self,  item_num, args):
#         super(cudaSASRec, self).__init__()

#         self.item_num = item_num
#         self.dev = args.device

#         self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
#         self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
#         self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

#         self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
#         self.attention_layers = torch.nn.ModuleList()
#         self.forward_layernorms = torch.nn.ModuleList()
#         self.forward_layers = torch.nn.ModuleList()

#         self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

#         for _ in range(args.num_layers):
#             new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
#             self.attention_layernorms.append(new_attn_layernorm)

#             new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
#                                                             args.num_heads,
#                                                             args.dropout_rate)
#             self.attention_layers.append(new_attn_layer)

#             new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
#             self.forward_layernorms.append(new_fwd_layernorm)

#             new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
#             self.forward_layers.append(new_fwd_layer)


#     def log2feats(self, log_seqs):
#         seqs = self.item_emb(log_seqs)
#         seqs *= self.item_emb.embedding_dim ** 0.5
#         positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
#         seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
#         seqs = self.emb_dropout(seqs)

#         timeline_mask = (log_seqs == 0)
#         seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

#         tl = seqs.shape[1] # time dim len for enforce causality
#         attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

#         for i in range(len(self.attention_layers)):
#             seqs = torch.transpose(seqs, 0, 1)
#             Q = self.attention_layernorms[i](seqs)
#             mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
#                                             attn_mask=attention_mask)
#                                             # key_padding_mask=timeline_mask
#                                             # need_weights=False) this arg do not work?
#             seqs = Q + mha_outputs
#             seqs = torch.transpose(seqs, 0, 1)

#             seqs = self.forward_layernorms[i](seqs)
#             seqs = self.forward_layers[i](seqs)
#             seqs *=  ~timeline_mask.unsqueeze(-1)

#         log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

#         return log_feats

#     def head(self, logits, indices):
#         if indices is None:
#             indices = torch.arange(0, self.item_num+1).to(self.dev)
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('cd,bd -> bc',item_embs, logits)
#         else:
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('bcd,bd -> bc',item_embs, logits)
#         return logits

#     def trainq(self, log_seqs, seq_y, seq_neg):
#         if seq_neg is None:
#             seq_neg = torch.arange(0,self.item_num+1).to(self.dev)
#             item_embs = self.item_emb(seq_neg)
#             logits = torch.einsum('bnd, cd->bnc',log_seqs, item_embs)
#         else:
#             neg_embs = self.item_emb(seq_neg)
#             seqy_embs = self.item_emb(seq_y)
#             neg_logits = torch.einsum('bnd, cd -> bnc', log_seqs, neg_embs)
#             pos_logits = torch.einsum('bnd, bnd->bn', log_seqs, seqy_embs)
#             logits = torch.cat((pos_logits.unsqueeze(-1), neg_logits), -1)
#         return logits
    
#     def losshead(self, logits, indices):
#         if indices is None:
#             indices = torch.arange(0, self.item_num+1).to(self.dev)
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('cd,bnd -> bnc',item_embs, logits)
#         else:
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('bcd,bnd -> bnc',item_embs, logits)
#         return logits
    

# #self-attention module with time embeddings
# class cudatqSASRec(torch.nn.Module):
#     def __init__(self,  item_num, args):
#         super(cudatqSASRec, self).__init__()

#         self.item_num = item_num
#         self.dev = args.device


#         self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
#         self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
#         self.time_emb = torch.nn.Embedding(args.time_bucket+1, args.hidden_units)
#         self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

#         self.attention_layernorms = torch.nn.ModuleList() 
#         self.attention_layers = torch.nn.ModuleList()
#         self.forward_layernorms = torch.nn.ModuleList()
#         self.forward_layers = torch.nn.ModuleList()

#         self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

#         for _ in range(args.num_layers):
#             new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
#             self.attention_layernorms.append(new_attn_layernorm)

#             new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
#                                                             args.num_heads,
#                                                             args.dropout_rate)
#             self.attention_layers.append(new_attn_layer)

#             new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
#             self.forward_layernorms.append(new_fwd_layernorm)

#             new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
#             self.forward_layers.append(new_fwd_layer)


#     def log2feats(self, log_seqs, time_seqs):
#         seqs = self.item_emb(log_seqs)
#         seqs *= self.item_emb.embedding_dim ** 0.5
#         positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
#         seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
#         time = self.time_emb(time_seqs)
#         seqs += time
#         seqs = self.emb_dropout(seqs)
        



#         timeline_mask = (log_seqs == 0)
#         seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

#         tl = seqs.shape[1] # time dim len for enforce causality
#         #no causality
#         attention_mask = torch.zeros((tl, tl), dtype=torch.bool, device=self.dev)
#         # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

#         for i in range(len(self.attention_layers)):
#             seqs = torch.transpose(seqs, 0, 1)
#             Q = self.attention_layernorms[i](seqs)
#             mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
#                                             attn_mask=attention_mask)
#                                             # key_padding_mask=timeline_mask
#                                             # need_weights=False) this arg do not work?
#             seqs = Q + mha_outputs
#             seqs = torch.transpose(seqs, 0, 1)

#             seqs = self.forward_layernorms[i](seqs)
#             seqs = self.forward_layers[i](seqs)
#             seqs *=  ~timeline_mask.unsqueeze(-1)

#         log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

#         return log_feats


#     def head(self, logits, indices):
#         if indices is None:
#             indices = torch.arange(0, self.item_num+1).to(self.dev)
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('cd,bd -> bc',item_embs, logits)
#         else:
#             item_embs = self.item_emb(indices) # (U, I, C)
#             logits = torch.einsum('bcd,bd -> bc',item_embs, logits)
#         return logits



# #category based preference encoder
# class starcateseq(torch.nn.Module):
#     def __init__(self, embedding_dim, cate_embedding_dim,  dropout_rate,  args):
#         super(starcateseq, self).__init__()
#         self.dropout = torch.nn.Dropout(p=dropout_rate)
#         self.lastk =args.lastk
#         self.norm = torch.nn.LayerNorm(args.hidden_units)
#         self.embedding_dim = embedding_dim
#         self.cate_embedding_dim = cate_embedding_dim

#         self.cate_querynetwork = nn.Linear(embedding_dim, args.hidden_units)
#         self.cate_keynetwork = nn.Linear(embedding_dim, args.hidden_units)
#         self.item_valuenetwork = nn.Linear(embedding_dim, embedding_dim)


        
#         self.feed_forward = PointWiseFeedForward(args.hidden_units, dropout_rate)
#         self.hidden_size = embedding_dim
#         self.head_num = args.num_heads
#         self.layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

#         assert self.hidden_size%self.head_num ==0
#         self.head_size = args.hidden_units // self.head_num

#     def  forward(self, item_seq, cate_seq, cate_y, mask):
#         cate_query = self.cate_querynetwork(cate_y)
#         cate_key = self.cate_keynetwork(cate_seq)
#         item_value = self.item_valuenetwork(item_seq)



#         cate_query_ = torch.cat(torch.split(cate_query, self.head_size, dim=2), dim=0)
#         cate_key_ = torch.cat(torch.split(cate_key, self.head_size, dim=2), dim=0)
#         item_value_ = torch.cat(torch.split(item_value, self.head_size, dim=2), dim=0)



#         att_0 = torch.einsum('bnd, bhd -> bnh', cate_query_, cate_key_)

#         scores = att_0 \
#                 / math.sqrt(self.cate_embedding_dim) 


#         mask = mask.repeat(self.head_num, 1 ).unsqueeze(1)
#         if mask is not None:
#             if scores.dtype == torch.float16:
#                 scores = scores.masked_fill(mask == 0, -65500)
#             else:
#                 scores = scores.masked_fill(mask == 0, -torch.inf)

#         c_attn = self.dropout(nn.functional.softmax(scores, dim=-1))


        
#         att_out = torch.einsum('bnh, bhd -> bnd', c_attn, item_value_)
#         att_out = torch.cat(torch.split(att_out, item_seq.shape[0], dim=0), dim=2) # div batch_size
#         att_value = self.layernorm(self.feed_forward(att_out))
        

#         return att_value

