from .layers import *
from recbole.utils import FeatureType


class sasrec_posf(torch.nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset, args):
        super(sasrec_posf, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.args = args

        self.n_items = dataset.num('item_id')
        self.dataset = dataset
        self.config = config
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        # self.n_items = dataset.num('item_id')


        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        # self.layer_norm_eps = config['layer_norm_eps']

        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.attention_layers = torch.nn.ModuleList()
        # self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(self.n_layers):

            new_attn_layer =  novaMultiHeadAttention(self.n_heads, self.hidden_size, self.hidden_dropout_prob, self.attn_dropout_prob, self.layer_norm_eps)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layer = FeedForward(self.hidden_size, self.hidden_size, self.hidden_dropout_prob, self.hidden_act, self.layer_norm_eps)
            self.forward_layers.append(new_fwd_layer)

        self.concat_layer = nn.Linear(self.hidden_size * (1 + self.num_feature_field), self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.feat_emb_layers = nn.ModuleList()
        self.feat_all = []
        for i, field_name in enumerate(config['selected_features']):
            
            if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                # self.token_field_names[type].append(field_name)
                # self.token_field_dims[type].append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.attribute_hidden_size[i], padding_idx=0))
                self.feat_all.append(self.dataset.item_feat[field_name])
            elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                # self.token_seq_field_names[type].append(field_name)
                # self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.attribute_hidden_size[i], padding_idx=0))
                self.feat_all.append(self.dataset.item_feat[field_name])

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
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        # print(gather_index.shape)
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(output.shape, gather_index.shape)
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        # feature_table = []




        # for i, name in enumerate(self.selected_features):
        #     if self.dataset.field2type[name] == FeatureType.TOKEN:
        #         feature_table.append(self.feat_emb_layers[i](self.feat_all[i][item_seq].to(item_seq.device)).unsqueeze(-2))

        #     elif self.dataset.field2type[name] == FeatureType.TOKEN_SEQ:
        #         # self.token_seq_field_names[type].append(field_name)
        #         if self.args.pooling_mode=='sum':
        #             feat_emb = torch.sum(self.feat_emb_layers[i](self.feat_all[i][item_seq].to(item_seq.device)), dim=-2).unsqueeze(-2)
        #             # feat_emb = self.transpose_for_scores(feat_emb)
        #             feature_table.append(feat_emb)
        #         elif self.args.pooling_mode=='mean':
        #             token_seq_field = self.feat_all[i][item_seq].to(item_seq.device)

        #             mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
        #             mask = mask.float()
        #             value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
        #             token_seq_embedding = self.feat_emb_layers[i](token_seq_field)
        #             mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
        #             masked_token_seq_embedding = token_seq_embedding * mask.float()
        #             result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]

        #             eps = torch.FloatTensor([1e-8]).to(item_seq.device)
        #             result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
        #             result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]
        #             feature_table.append(result)



        # if self.args.integra =='gating':
        #     fuse_function = self.gating
        #     fuse_emb = self.gating(item_emb, feature_table) 

        # if self.args.integra =='gating_plus':
        #     fuse_function = self.gating_plus
        #     fuse_emb = self.gating_plus(item_emb, feature_table) 
            
        # if self.args.integra =='add':
        #     fuse_function = self.sum
        #     fuse_emb = self.sum(item_emb, feature_table) 

        # if self.args.integra =='concat':
        #     fuse_function = self.concat
        #     fuse_emb = self.concat(item_emb, feature_table) 

        # if self.args.integra =='item':
        #     fuse_function = self.item
        #     fuse_emb = item_emb

        # cate_emb = feature_table[0].squeeze()

        input_emb = self.dropout(position_embedding+item_emb)
        # input_emb = fuse_emb + position_embedding
        # input_emb = self.LayerNorm(input_emb)
        # input_emb = self.dropout(input_emb)

        # item_input = self.dropout(item_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        for i in range(1):

            item_input = self.attention_layers[i](input_emb, extended_attention_mask, input_emb)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            item_input = self.forward_layers[i](item_input)

            # input_emb = fuse_function(item_input, feature_table) + position_embedding
        
        output = item_input
        seq_output = self.gather_indexes(output, item_seq_len - 1)

        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction['item_id_list']
        # item_seq_len = torch.ones(interaction['item_id'].size()).long().to(item_seq.device)*50
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction['item_id']
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

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = torch.ones(interaction['item_id'].size()).long().to(item_seq.device)*50
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        # item_seq_len = torch.ones(interaction['item_id'].size()).long().to(item_seq.device)*50
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
