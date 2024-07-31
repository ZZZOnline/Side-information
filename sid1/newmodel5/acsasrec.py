import torch
from torch import nn

from .layers import AttackRTransformerEncoder
# from recbole.model.loss import BPRLoss


class acsasrec(torch.nn.Module):
    def __init__(self, config, dataset, args):
        super(acsasrec, self).__init__()


        self.dataset = dataset
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
        calibrated_output = self.gather_indexes(calibrated_output, item_seq_len - 1)
        attacked_output = self.gather_indexes(attacked_output, item_seq_len - 1)
        return attacked_output, calibrated_output, all_attack_masks


    def _cal_loss(self, output, interaction, attack_loss=False):
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

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
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        attacked_output, calibrated_output, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        # attacked_scores = torch.matmul(attacked_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores = torch.matmul(calibrated_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
