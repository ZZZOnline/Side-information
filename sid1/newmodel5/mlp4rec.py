# import torch
# from torch import nn
from functools import partial
# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
# from recbole.model.loss import BPRLoss
from .layers import *

# Partially based on the implementation of MLPMixer from Lucidrain
# https://github.com/lucidrains/mlp-mixer-pytorch

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x.clone())) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class mlp4rec(torch.nn.Module):

    def __init__(self, config, dataset, args):
        super(mlp4rec, self).__init__()


        self.dataset = dataset
        self.args = args
        self.config = config
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num('item_id')

        # load parameters info
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        expansion_factor = 4
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.num_feature_field = len(config['selected_features'])
        self.layerSize = self.num_feature_field + 1

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)


        self.attribute_hidden_size = config['attribute_hidden_size']
        self.feat_emb_layers = nn.ModuleList()
        self.feat_all = []
        self.feanum = []
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
        
        self.sequenceMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.featureMixer = PreNormResidual(self.hidden_size, FeedForward(self.layerSize, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.layers = nn.ModuleList([])
        for i in range(self.num_feature_field+1):
            self.layers.append(self.sequenceMixer)
            self.layers.append(self.channelMixer)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

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

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        # sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        # sparse_embedding = sparse_embedding['item']
        # dense_embedding = dense_embedding['item']
        # if sparse_embedding is not None:
        #     feature_embeddings = sparse_embedding
        # if dense_embedding is not None:
        #     if sparse_embedding is not None:
        #         feature_embeddings = torch.cat((sparse_embedding,dense_embedding),2)
        #     else:
        #         feature_embeddings = dense_embedding

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


        feature_emb = torch.cat(feature_table, -2)

        item_emb = torch.unsqueeze(item_emb,2)
        # print(item_emb.shape, feature_emb.shape)
        item_emb = torch.cat((item_emb,feature_emb),2)
        mixer_outputs = torch.split(item_emb,[1]*(self.num_feature_field+1),2)
        mixer_outputs = torch.stack(mixer_outputs,0)
        mixer_outputs = torch.squeeze(mixer_outputs)
        for _ in range(self.n_layers):
            for x in range(self.num_feature_field+1):
                mixer_outputs[x] = self.layers[x*2](mixer_outputs[x])
                mixer_outputs[x] = self.layers[(x*2)+1](mixer_outputs[x])
            mixer_outputs = torch.movedim(mixer_outputs,0,2)
            batch_size = mixer_outputs.size()[0]
            mixer_outputs = torch.flatten(mixer_outputs,0,1)
            mixer_outputs = self.featureMixer(mixer_outputs)
            mixer_outputs = torch.reshape(mixer_outputs,(batch_size,self.max_seq_length,self.layerSize,self.hidden_size))
            mixer_outputs = torch.movedim(mixer_outputs,2,0)

        output = self.gather_indexes(mixer_outputs[0], item_seq_len - 1)
        output = self.LayerNorm(output)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction['item_id_list']
        # feat_seq_batch = self.cateall[item_seq_batch].to(self.args.device)

        
        pos_items = interaction['item_id']
    
        # item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


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


    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        # feat_seq_batch = self.cateall[item_seq_batch].to(self.args.device)

        
        pos_items = interaction['item_id']
    
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
