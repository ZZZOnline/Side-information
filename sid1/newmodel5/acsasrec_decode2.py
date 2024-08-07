import torch
from torch import nn

from .layers import AttackRTransformerEncoder
# from recbole.model.loss import BPRLoss
from .layers import *

#decode with both attention
class acsasrec_decode2(torch.nn.Module):
    def __init__(self, config, dataset, args):
        super(acsasrec_decode2, self).__init__()


        self.dataset = dataset
        self.args = args
        # self.config = config
        # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        # self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.ITEM_SEQ = 'item_id_list'
        self.POS_ITEM_ID = 'item_id'
        self.n_items = dataset.num('item_id')
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        # self.ITEM_SEQ = config['ITEM_ID_FIELD']

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        # self.layer_norm_eps = config['layer_norm_eps']
        self.layer_norm_eps = 1e-12

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.combine_option = 'gate'
        self.rich_calibrated_combine = 'none'
        self.two_level = 'True'
        self.use_position_embedding = 'False'
        self.use_order = 'True'
        self.use_distance = 'True'
        self.trainable_mask_loss_weight = 'True'


        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.beha = config['RATING_FIELD']
        self.n_beha = dataset.num(self.beha)
        self.beha_embedding = nn.Embedding(self.n_beha, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.hidden_act = 'gelu'
        self.layer_norm_eps = 1e-12

        if self.use_position_embedding:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = AttackRTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            combine_option=self.combine_option,
            use_order=self.use_order,
            use_distance=self.use_distance,
            two_level=self.two_level,
            rich_calibrated_combine=self.rich_calibrated_combine
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.trainable_mask_loss_weight = config['trainable_mask_loss_weight']
        if self.trainable_mask_loss_weight:
            self.mask_loss_weight = nn.Parameter(torch.FloatTensor([0.3]), requires_grad=True)
        else:
            self.mask_loss_weight = 0.03

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.selected_features = config['selected_features']
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.feat_emb_layers = nn.ModuleList()
        self.feat_all = []
        self.feanum = []
        self.p_heads = args.p_heads
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
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
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
        position_embedding = self.position_embedding(position_ids)
        pos_emb = self.dropout(position_embedding)

        beha_emb = self.dropout(beha_emb)
        value_input = self.dropout(value_input)
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(value_input)+value_input


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        pos_key_layer = self.transpose_for_scores(self.pos_key_proj(pos_emb))
        pos_query_layer = self.transpose_for_scores(self.pos_query_proj(pos_emb))

        beha_key_layer = self.transpose_for_scores(self.beha_key_proj(beha_emb))
        beha_query_layer = self.transpose_for_scores(self.beha_query_proj(beha_emb))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))+torch.matmul(query_layer, (beha_key_layer+pos_key_layer).transpose(-1, -2)/2)+\
                    torch.matmul((beha_query_layer+ pos_query_layer)/2, key_layer.transpose(-1, -2))
    
        attention_scores = attention_scores / math.sqrt(self.attention_head_size*3)

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

    def decoder1(self, input, value_input, beha_emb, extended_attention_mask):
        position_ids = torch.arange(input.size(1), dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand(input.size()[:-1])
        position_embedding = self.position_embedding(position_ids)
        pos_emb = self.dropout(position_embedding)

        beha_emb = self.dropout(beha_emb)
        value_input = self.dropout(value_input)
        mixed_query_layer = self.query1(input)
        mixed_key_layer = self.key1(input)
        mixed_value_layer = self.value1(value_input)+value_input

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        pos_key_layer = self.transpose_for_scores(self.pos_key_proj1(pos_emb))
        pos_query_layer = self.transpose_for_scores(self.pos_query_proj1(pos_emb))

        beha_key_layer = self.transpose_for_scores(self.beha_key_proj1(beha_emb))
        beha_query_layer = self.transpose_for_scores(self.beha_query_proj1(beha_emb))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))+torch.matmul(query_layer, (beha_key_layer+pos_key_layer).transpose(-1, -2)/2)+\
                    torch.matmul((beha_query_layer+ pos_query_layer)/2, key_layer.transpose(-1, -2))
    
        attention_scores = attention_scores / math.sqrt(self.attention_head_size*3)

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


    def forward(self, item_seq, item_seq_len, is_train=False):
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        if self.use_position_embedding:
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # print('a', input_emb.shape)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        all_attack_masks = trm_output[1]
        attacked_output, calibrated_output = trm_output[0][-1]
        # calibrated_output = self.gather_indexes(calibrated_output, item_seq_len - 1)
        # attacked_output = self.gather_indexes(attacked_output, item_seq_len - 1)


        
        return attacked_output, calibrated_output, all_attack_masks


    def _cal_loss(self, output, interaction, attack_loss=False):



        args = self.args
        item_seq = interaction['item_id_list']
        # feat_seq_batch = self.cateall[item_seq_batch].to(self.args.device)

        
        pos_items = interaction['item_id']
    
        # item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        beha_seq = interaction[self.beha+'_list']
    
        pos_items = interaction[self.POS_ITEM_ID]
        item_emb = self.item_embedding(item_seq)
        beha_emb = self.beha_embedding(beha_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)
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

        hidden_states = self.decoder(output, item_emb, beha_emb, extended_attention_mask)
        itemoutput = hidden_states + output
        itemoutput = self.gather_indexes(itemoutput, item_seq_len - 1).unsqueeze(1)
        itemoutput = self.transpose_for_heads(itemoutput, args.p_heads)
        itemoutput = torch.einsum('bnhdl, hlk->bnhdk', itemoutput, self.boost1)
        new_shape = itemoutput.size()[:2] + (self.hidden_size,)
        itemoutput = itemoutput.view(*new_shape).squeeze()



        fea_output = []
        for i, name in enumerate(self.selected_features):
            fea_hidden = self.decoder1(output, feature_table[i].squeeze(), beha_emb, extended_attention_mask)
            fea_hidden = self.gather_indexes(fea_hidden, item_seq_len - 1).unsqueeze(1)
            fea_hidden = self.transpose_for_heads(fea_hidden, args.p_heads)
            fea_hidden = torch.einsum('bnhdl, hlk->bnhdk', fea_hidden, self.feat_boost[i])
            fea_hidden = fea_hidden.view(*new_shape)

            fea_output.append(fea_hidden)

        if self.args.integra =='add':
            fuse_emb = self.sum(itemoutput, fea_output) 
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

        if self.args.integra =='add':
            catetest_item_emb = self.item_embedding.weight
            allinput = self.sum(catetest_item_emb, feature_emb)
    
        if self.args.integra =='item':
            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb



        logits = torch.matmul(final_output, allinput.transpose(0, 1))
        itemloss = self.loss_fct(logits, pos_items)
        if torch.rand(1)<0.01:
            print(self.boost1, self.feat_boost)
        
        boostloss = self.boost1.norm()
        for i in self.feat_boost:
            boostloss += i.norm()


        # test_item_emb = self.item_embedding.weight
        # logits = torch.matmul(output, test_item_emb.transpose(0, 1))
        # loss = self.loss_fct(logits, pos_items)
        return itemloss+ self.weight_norm*(boostloss)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        attacked_output, calibrated_output, all_attack_masks = self.forward(item_seq, item_seq_len, is_train=True)

        final_attacked_loss = None
        if attacked_output is not None:
            attacked_loss = -self._cal_loss(attacked_output, interaction, attack_loss=True)

            mask_penalty = []
            for attack_mask in all_attack_masks:
                if attack_mask is None:
                    continue
                mask_penalty.append(torch.norm(1 - attack_mask, p=2))
            assert len(mask_penalty) > 0
            mask_penalty = torch.mean(torch.stack(mask_penalty, dim=0))
            if self.trainable_mask_loss_weight:
                final_attacked_loss = attacked_loss + mask_penalty * self.mask_loss_weight[0]
            else:
                # print(mask_penalty, self.mask_loss_weight)
                final_attacked_loss = attacked_loss + mask_penalty * self.mask_loss_weight
        calibrated_loss = self._cal_loss(calibrated_output, interaction)

        return final_attacked_loss + calibrated_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        attacked_output, calibrated_output, _ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)

        attacked_scores = torch.mul(attacked_output, test_item_emb).sum(dim=1)  # [B]
        scores = torch.mul(calibrated_output, test_item_emb).sum(dim=1)  # [B]
        return attacked_scores, scores

    def full_sort_predict(self, interaction):

        args = self.args
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        attacked_output, output, _ = self.forward(item_seq, item_seq_len)
        beha_seq = interaction[self.beha+'_list']

        item_emb = self.item_embedding(item_seq)
        beha_emb = self.beha_embedding(beha_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)
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

        hidden_states = self.decoder(output, item_emb, beha_emb, extended_attention_mask)
        itemoutput = hidden_states + output
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

        if self.args.integra =='add':
            fuse_emb = self.sum(itemoutput, fea_output) 
        final_output = self.new_fwd_layer(fuse_emb)


        feature_table = []

        for i, name in enumerate(self.selected_features):

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

        if self.args.integra =='add':
            catetest_item_emb = self.item_embedding.weight
            allinput = self.sum(catetest_item_emb, feature_emb)
    
        if self.args.integra =='item':
            catetest_item_emb = self.item_embedding.weight
            allinput = catetest_item_emb

        scores = torch.matmul(final_output, allinput.transpose(0, 1))

        return scores
