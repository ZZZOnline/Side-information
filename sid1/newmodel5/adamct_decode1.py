# -*- coding: utf-8 -*-
import torch
import copy
import math
import numpy as np
from torch import nn
import torch.nn.functional as fn

# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.layers import AdaMCTEncoder
from .layers import *

class adamct_decode1(torch.nn.Module):
    def __init__(self, config, dataset, args):
        super(adamct_decode1, self).__init__()

        self.dataset = dataset
        self.args = args
        self.config = config
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num('item_id')
        self.ITEM_SEQ = 'item_id_list'
        self.POS_ITEM_ID = 'item_id'




        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.kernel_size = config["kernel_size"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"], # max sequence length
        self.reduction_ratio = config["reduction_ratio"],
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]


        self.kernel_size = 3
        self.reduction_ratio = 2
        # define layers and loss

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.position_embedding1 = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.adamct_encoder = AdaMCTEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            seq_len = self.seq_len, 
            reduction_ratio = self.reduction_ratio,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.p_heads = args.p_heads
        self.weight_norm = args.weight_norm
        self.beha = config['RATING_FIELD']
        self.n_beha = dataset.num(self.beha)
        self.beha_embedding = nn.Embedding(self.n_beha, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12
        self.attribute_hidden_size = config['attribute_hidden_size']

        self.feat_emb_layers = nn.ModuleList()
        self.feat_all = []
        self.feanum = []

        self.boost1 = torch.nn.Parameter(torch.ones((self.p_heads, 1, 1), requires_grad=True))
        self.feat_boost = nn.ParameterList()
        for i, field_name in enumerate(config['selected_features']):
            
            if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                # self.token_field_names[type].append(field_name)
                # self.token_field_dims[type].append(self.dataset.num(field_name))
                self.feanum.append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.attribute_hidden_size[i], padding_idx=0))
                self.feat_all.append(self.dataset.item_feat[field_name])
            elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                # self.token_seq_field_names[type].append(field_name)
                # self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                self.feanum.append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.attribute_hidden_size[i], padding_idx=0))
                self.feat_all.append(self.dataset.item_feat[field_name])
            self.feat_boost.append(torch.nn.Parameter(torch.ones((self.p_heads, 1, 1), requires_grad=True)))
        



        self.weight_norm = args.weight_norm
        self.inner_size = config['inner_size']
        self.new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.hidden_act = 'gelu'
        self.new_fwd_layer = FeedForward(self.hidden_size, self.inner_size, self.hidden_dropout_prob, hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps)
        self.fea_new_fwd_layer = FeedForward(self.hidden_size, self.inner_size, self.hidden_dropout_prob, hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps)

        self.num_attention_heads = self.n_heads
        self.attention_head_size = int(self.hidden_size / self.n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.p_head_size = int(self.hidden_size / self.p_heads)





        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.query0 = nn.Linear(self.hidden_size, self.all_head_size)
        self.key0 = nn.Linear(self.hidden_size, self.all_head_size)

        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.pos_key_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.pos_query_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.beha_key_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.beha_query_proj = nn.Linear(self.hidden_size, self.all_head_size)

        self.query1 = nn.Linear(self.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(self.hidden_size, self.all_head_size)

        self.query10 = nn.Linear(self.hidden_size, self.all_head_size)
        self.key10 = nn.Linear(self.hidden_size, self.all_head_size)
        
        self.value1 = nn.Linear(self.hidden_size, self.all_head_size)
        self.pos_key_proj1 = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.pos_query_proj1 = nn.Linear(self.hidden_size, self.all_head_size)
        self.beha_key_proj1 = nn.Linear(self.hidden_size, self.all_head_size)
        self.beha_query_proj1 = nn.Linear(self.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(self.attn_dropout_prob)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)


        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    #the last dimension d is transformed into (h, d//h, 1)
    def transpose_for_heads(self, x, heads):
        new_x_shape = x.size()[:-1] + (heads, self.p_head_size, 1)
        x = x.view(*new_x_shape)
        return x
    
    def item(self, item_logits):
        return item_logits
    
    def sum(self, item_logits, fea_logits):

        fea_input = torch.sum(torch.cat(fea_logits, dim=-2), dim=-2)
        return item_logits+fea_input

    def decoder(self, input, value_input, beha_emb, extended_attention_mask):
        position_ids = torch.arange(input.size(1), dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand(input.size()[:-1])
        position_embedding = self.position_embedding1(position_ids)
        pos_emb = position_embedding

        # beha_emb = self.dropout(beha_emb)
        value_input = self.dropout(value_input)
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        
        mixed_query_layer0 = self.query0(input)
        mixed_key_layer0 = self.key0(input)

        mixed_value_layer = self.value(value_input)+value_input


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        query_layer0 = self.transpose_for_scores(mixed_query_layer0)
        key_layer0 = self.transpose_for_scores(mixed_key_layer0)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        pos_key_layer = self.transpose_for_scores(self.pos_key_proj(pos_emb))
        pos_query_layer = self.transpose_for_scores(self.pos_query_proj(pos_emb))

        # beha_key_layer = self.transpose_for_scores(self.beha_key_proj(beha_emb))
        # beha_query_layer = self.transpose_for_scores(self.beha_query_proj(beha_emb))
        # Take the dot product between "query" and "key" to get the raw attention scores.

        # attention_scores = torch.matmul(query_layer+pos_query_layer, (key_layer+pos_key_layer).transpose(-1, -2))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))+torch.matmul(query_layer0, pos_key_layer.transpose(-1, -2))+\
                    torch.matmul(pos_query_layer, key_layer0.transpose(-1, -2))+torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size*4)

        attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = context_layer
        hidden_states = self.out_dropout(hidden_states)

        return self.LayerNorm(hidden_states)
    #no more self.dropout(position_embedding)
    def decoder1(self, input, value_input, beha_emb, extended_attention_mask):
        position_ids = torch.arange(input.size(1), dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand(input.size()[:-1])
        position_embedding = self.position_embedding1(position_ids)
        pos_emb = position_embedding

        # beha_emb = self.dropout(beha_emb)
        value_input = self.dropout(value_input)
        mixed_query_layer = self.query1(input)
        mixed_key_layer = self.key1(input)

        mixed_query_layer0 = self.query10(input)
        mixed_key_layer0 = self.key10(input)

        mixed_value_layer = self.value1(value_input)+value_input

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        query_layer0 = self.transpose_for_scores(mixed_query_layer0)
        key_layer0 = self.transpose_for_scores(mixed_key_layer0)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        pos_key_layer = self.transpose_for_scores(self.pos_key_proj1(pos_emb))
        pos_query_layer = self.transpose_for_scores(self.pos_query_proj1(pos_emb))

        # beha_key_layer = self.transpose_for_scores(self.beha_key_proj1(beha_emb))
        # beha_query_layer = self.transpose_for_scores(self.beha_query_proj1(beha_emb))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))+torch.matmul(query_layer0, pos_key_layer.transpose(-1, -2))+\
                    torch.matmul(pos_query_layer, key_layer0.transpose(-1, -2))+torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size*4)

        attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = context_layer
        hidden_states = self.out_dropout(hidden_states)

        return self.LayerNorm(hidden_states)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

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



    def forward(self, item_seq, item_seq_len, beha_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        adamct_output = self.adamct_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = adamct_output[-1]
        



        args = self.args
        feature_table = []
        for i, name in enumerate(self.selected_features):
            if self.dataset.field2type[name] == FeatureType.TOKEN:
                feature_table.append(self.feat_emb_layers[i](self.feat_all[i][item_seq].to(item_seq.device)).unsqueeze(-2))

            elif self.dataset.field2type[name] == FeatureType.TOKEN_SEQ:
                # self.token_seq_field_names[type].append(field_name)
                if self.args.pooling_mode=='sum':
                    feat_emb = torch.sum(self.feat_emb_layers[i](self.feat_all[i][item_seq].to(item_seq.device)), dim=-2).unsqueeze(-2)
                    # feat_emb = self.transpose_for_scores(feat_emb)
                    feature_table.append(feat_emb)
                #0 padding should not be considered in mean function
                elif self.args.pooling_mode=='mean':
                    token_seq_field = self.feat_all[i][item_seq].to(item_seq.device)

                    mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
                    token_seq_embedding = self.feat_emb_layers[i](token_seq_field)
                    mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
                    masked_token_seq_embedding = token_seq_embedding * mask.float()
                    result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]

                    eps = torch.FloatTensor([1e-8]).to(item_seq.device)
                    result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
                    result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]
                    feature_table.append(result)


        # beha_emb = self.beha_embedding(beha_seq)
        beha_emb = None
        extended_attention_mask = self.get_attention_mask(item_seq)
        hidden_states = self.decoder(output, item_emb.squeeze(), beha_emb, extended_attention_mask)
        itemoutput = hidden_states + output

        origin_itemoutput = self.gather_indexes(itemoutput, item_seq_len - 1)
        itemoutput = self.gather_indexes(itemoutput, item_seq_len - 1).unsqueeze(1)
        itemoutput = self.transpose_for_heads(itemoutput, args.p_heads)
        itemoutput = torch.einsum('bnhdl, hlk->bnhdk', itemoutput, self.boost1)
        new_shape = itemoutput.size()[:2] + (self.hidden_size,)
        itemoutput = itemoutput.view(*new_shape)



        fea_output = []
        for i, name in enumerate(self.selected_features):
            fea_hidden = self.decoder1(output, feature_table[i].squeeze(), beha_emb, extended_attention_mask)
            fea_hidden = self.gather_indexes(fea_hidden, item_seq_len - 1).unsqueeze(1)
            fea_hidden = self.transpose_for_heads(fea_hidden, args.p_heads)
            fea_hidden = torch.einsum('bnhdl, hlk->bnhdk', fea_hidden, self.feat_boost[i])
            fea_hidden = fea_hidden.view(*new_shape)

            fea_output.append(fea_hidden)
        

        output = self.gather_indexes(output, item_seq_len - 1)  
        return itemoutput.squeeze(), fea_output, origin_itemoutput  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        beha_seq = interaction[self.beha+'_list']
        pos_items = interaction['item_id']
        itemoutput, fea_output, original_output = self.forward(item_seq, item_seq_len, beha_seq)
        args = self.args

        if self.args.integra =='add':
            fuse_emb = self.sum(itemoutput, fea_output) 
        if self.args.integra =='item':
            fuse_emb = original_output

        final_output = self.new_fwd_layer(fuse_emb)


        feature_table = []

        for i, name in enumerate(self.selected_features):
            pos_fea = self.feat_all[i][pos_items].to(item_seq.device)
            if self.dataset.field2type[name] == FeatureType.TOKEN:

                fea_result = self.feat_emb_layers[i](self.feat_all[i].to(item_seq.device)).unsqueeze(-2)

            elif self.dataset.field2type[name] == FeatureType.TOKEN_SEQ:

                if self.args.pooling_mode=='sum':

                    fea_result = torch.sum(self.feat_emb_layers[i](self.feat_all[i].to(item_seq.device)), dim=-2).unsqueeze(-2)
                elif self.args.pooling_mode=='mean':
                    
                    token_seq_field = self.feat_all[i].to(item_seq.device)
                    mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
                    token_seq_embedding = self.feat_emb_layers[i](token_seq_field)
                    mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
                    masked_token_seq_embedding = token_seq_embedding * mask.float()
                    fea_result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, embed_dim]
                    eps = torch.FloatTensor([1e-8]).to(item_seq.device)
                    fea_result = torch.div(fea_result, value_cnt + eps).unsqueeze(-2)  # [batch_size,  embed_dim]

            result = self.transpose_for_heads(fea_result, args.p_heads)

            result = torch.einsum('bnhdl, hlk->bnhdk', result, self.feat_boost[i])

            #b1d
            new_shape = result.size()[:2] + (self.hidden_size,)
            result = result.view(*new_shape)
            feature_table.append(result)

        feature_emb = feature_table

        if args.integra =='add':
            catetest_item_emb = self.item_embedding.weight.unsqueeze(1)
            catetest_item_emb = self.transpose_for_heads(catetest_item_emb, args.p_heads)

            catetest_item_emb = torch.einsum('bnhdl, hlk->bnhdk', catetest_item_emb, self.boost1)

            #b1d
            new_shape = result.size()[:2] + (self.hidden_size,)
            catetest_item_emb = catetest_item_emb.view(*new_shape).squeeze()
    
            allinput = self.sum(catetest_item_emb, feature_emb)
    
            logits = torch.matmul(final_output, allinput.transpose(0, 1))
            itemloss = self.loss_fct(logits, pos_items)
            if torch.rand(1)<0.01:
                print(self.boost1, self.feat_boost[0])
            
            boostloss = self.boost1.norm()
            for i in self.feat_boost:
                boostloss += i.norm()
            return itemloss+ self.weight_norm*(boostloss)

        if args.integra =='item':
            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb
            logits = torch.matmul(final_output, allinput.transpose(0, 1))
            itemloss = self.loss_fct(logits, pos_items)
            return itemloss


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        args = self.args
        
        item_seq = interaction['item_id_list']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        beha_seq = interaction[self.beha+'_list']
        itemoutput, fea_output, original_output = self.forward(item_seq, item_seq_len, beha_seq)

        if self.args.integra =='add':
            fuse_emb = self.sum(itemoutput, fea_output) 

        if self.args.integra =='item':
            fuse_function = self.item
            fuse_emb = original_output
        
        final_output = self.new_fwd_layer(fuse_emb)



        feature_table= []
        for i, name in enumerate(self.selected_features):
            if self.dataset.field2type[name] == FeatureType.TOKEN:

                fea_result = self.feat_emb_layers[i](self.feat_all[i].to(item_seq.device)).unsqueeze(-2)

            elif self.dataset.field2type[name] == FeatureType.TOKEN_SEQ:

                if self.args.pooling_mode=='sum':

                    #[c,1,d]
                    fea_result = torch.sum(self.feat_emb_layers[i](self.feat_all[i].to(item_seq.device)), dim=-2).unsqueeze(-2)
                elif self.args.pooling_mode=='mean':
                    
                    token_seq_field = self.feat_all[i].to(item_seq.device)
                    mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
                    mask = mask.float()
                    value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
                    token_seq_embedding = self.feat_emb_layers[i](token_seq_field)
                    mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
                    masked_token_seq_embedding = token_seq_embedding * mask.float()
                    fea_result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, embed_dim]
                    eps = torch.FloatTensor([1e-8]).to(item_seq.device)
                    fea_result = torch.div(fea_result, value_cnt + eps).unsqueeze(-2)  # [batch_size,  embed_dim]


            result = self.transpose_for_heads(fea_result, args.p_heads)
            result = torch.einsum('bnhdl, hlk->bnhdk', result, self.feat_boost[i])


            new_shape = result.size()[:2] + (self.hidden_size,)
            result = result.view(*new_shape)
            feature_table.append(result)

        feature_emb = feature_table

        if args.integra =='add':
            catetest_item_emb = self.item_embedding.weight.unsqueeze(1)
            catetest_item_emb = self.transpose_for_heads(catetest_item_emb, args.p_heads)

            catetest_item_emb = torch.einsum('bnhdl, hlk->bnhdk', catetest_item_emb, self.boost1)

            #b1d
            new_shape = result.size()[:2] + (self.hidden_size,)
            catetest_item_emb = catetest_item_emb.view(*new_shape).squeeze()
    
            allinput = self.sum(catetest_item_emb, feature_emb)
            
        if args.integra =='item':
            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb

        scores = torch.matmul(final_output, allinput.transpose(0, 1))
        return scores


class AdaMCTEncoder(nn.Module):
    r"""One AdaMCTEncoder consists of several AdaMCTLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        kernel_size(num): the size of kernel in convolutional layer. Default: 2
        reduction_ratio(num): the reduction rate in squeeze-excitation attention layer. Default: 2
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in convolutional layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        kernel_size=3,
        seq_len=64, 
        reduction_ratio=2,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(AdaMCTEncoder, self).__init__()
        layer = AdaMCTLayer(
            n_heads,
            hidden_size,
            kernel_size,
            seq_len, 
            reduction_ratio,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class AdaMCTLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        kernel_size,
        seq_len, 
        reduction_ratio,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(AdaMCTLayer, self).__init__()
        self.linear_en = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.local_conv = LocalConv(
            hidden_size,
            kernel_size, 
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.global_seatt = SqueezeExcitationAttention(seq_len, reduction_ratio)
        self.local_seatt = SqueezeExcitationAttention(seq_len, reduction_ratio)

        self.adaptive_mixture_units = AdaptiveMixtureUnits(hidden_size, layer_norm_eps, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        hidden_states_en = self.LayerNorm(self.dropout(self.linear_en(hidden_states)))

        global_output = self.multi_head_attention(hidden_states_en, attention_mask)
        global_output = self.global_seatt(global_output)

        local_output = self.local_conv(hidden_states_en)
        local_output = self.local_seatt(local_output)

        layer_output = self.adaptive_mixture_units(hidden_states, global_output, local_output)
        return layer_output


class AdaptiveMixtureUnits(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super(AdaptiveMixtureUnits, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.adaptive_act_fn = torch.sigmoid
        self.linear_out = nn.Linear(hidden_size, hidden_size)
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

    def forward(self, input_tensor, global_output, local_output):
        input_tensor_avg = torch.mean(input_tensor, dim=1) # [B, D]
        ada_score_alpha = self.adaptive_act_fn(self.linear(input_tensor_avg)).unsqueeze(-1) # [B, 1, 1]
        ada_score_beta = 1 - ada_score_alpha
        
        mixture_output = torch.mul(global_output, ada_score_beta) + torch.mul(local_output, ada_score_alpha) # [B, N, D]

        output = self.LayerNorm(self.dropout(self.linear_out(mixture_output)) + input_tensor) # [B, N, D]
        return output


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, seq_len, reduction_ratio):
        super(SqueezeExcitationAttention, self).__init__()
        # print(seq_len, reduction_ratio)
        self.dense_1 = nn.Linear(seq_len[0], seq_len[0]//reduction_ratio)
        self.squeeze_act_fn = fn.relu

        self.dense_2 = nn.Linear(seq_len[0]//reduction_ratio, seq_len[0])
        self.excitation_act_fn = torch.sigmoid

    def forward(self, input_tensor):
        input_tensor_avg = torch.mean(input_tensor, dim=-1, keepdim=True) # [B, N, 1]
        
        hidden_states = self.dense_1(input_tensor_avg.permute(0, 2, 1)) # [B, 1, N] -> [B, 1, N/r]
        hidden_states = self.squeeze_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states) # [B, 1, N/r] -> [B, 1, N] 
        att_score = self.excitation_act_fn(hidden_states) # sigmoid
        
        # reweight
        input_tensor = torch.mul(input_tensor, att_score.permute(0, 2, 1)) # [B, N, D]
        return input_tensor


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class LocalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(LocalConv, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv_1_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=self.padding)
        self.conv_act_fn = self.get_hidden_act(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.dropout = nn.Dropout(hidden_dropout_prob)

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
        hidden_states = self.conv_1_3(input_tensor.permute(0, 2, 1))
        hidden_states = self.LayerNorm(hidden_states.permute(0, 2, 1))
        hidden_states = self.conv_act_fn(hidden_states)
        return hidden_states