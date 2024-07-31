from .layers import *


class difsr(torch.nn.Module):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, dataset, args):
        super(difsr, self).__init__()
        # self.USER_ID = config['USER_ID_FIELD']
        # self.ITEM_ID = config['ITEM_ID_FIELD']
        # self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        # self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        # self.POS_ITEM_ID = self.ITEM_ID
        # self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        # self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        # self.n_items = dataset.num(self.ITEM_ID)
        self.dataset = dataset
        self.config = config
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num('item_id')

        self.n_feat =[]
        self.feat_all = dict()



            # self.cateall = dataset.item_feat['categories']
        # self.cateall = torch.tensor([imap[i] for i in np.arange(0, itemnum)]) 
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        # self.layer_norm_eps = config['layer_norm_eps']
        self.layer_norm_eps = 1e-12

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = 'cuda'
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # self.fusion_type = config['fusion_type']
        self.fusion_type = 'sum'

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']
        # self.attribute_predictor = 'not'

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # self.fea_embedding = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)
        # self.fea_embedding1 = nn.Embedding(self.n_cates, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.selected_features),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

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
        
        self.n_attributes = {}
        for attribute in self.selected_features:
            self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        if self.attribute_predictor == 'MLP':
            self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.hidden_size),
                                             nn.BatchNorm1d(num_features=self.hidden_size),
                                             nn.ReLU(),
                                             # final logits
                                             nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.n_attributes)
                                             )
        elif self.attribute_predictor == 'linear':
            self.ap = nn.ModuleList(
                [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_]))
                 for _ in self.selected_features])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


        self.args = args
        # self.cateall = torch.tensor([imap[i] for i in np.arange(0, itemnum)]) 
        # define the module feature gating need
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.b = nn.Parameter(torch.zeros(args.hidden_units), requires_grad=True).to(self.dev)

        # define the module instance gating need
        self.w3 = nn.Linear(self.hidden_size, 1, bias=False)
        self.w4 = nn.Linear(self.hidden_size, args.maxlen, bias=False)
        self.w5 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.w6 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.gating_network = VanillaAttention(self.hidden_size, self.hidden_size)


        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']


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

    def forward(self, item_seq, item_seq_len, feat_seq):

        # print(feat_seq.shape, item_seq.shape)

        args = self.args
        item_emb = self.item_embedding(item_seq)
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # feature_emb = []
        # for i in range(len(self.config['selected_features'])):

        #     feature_emb = self.fea_embedding(feat_seq)
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


        feature_emb = feature_table
        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # print(feature_emb[0].shape, feature_emb)
        trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq_batch = interaction['item_id_list']
        # feat_seq_batch = self.cateall[item_seq_batch].to(self.args.device)

        
        pos_items = interaction['item_id']
        # pos_cate = self.cateall[pos_items]
        # item_seq_len = torch.ones(pos_items.size()).long().to(item_seq_batch.device)*50
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        args = self.args
        seq_output = self.forward(item_seq_batch, item_seq_len, None)


        allinput = self.item_embedding.weight

        if True:  # self.loss_type = 'CE'
            # test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, allinput.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            if self.attribute_predictor!='' and self.attribute_predictor!='not':
                loss_dic = {'item_loss':loss}
                attribute_loss_sum = 0
                for i, a_predictor in enumerate(self.ap):
                    attribute_logits = a_predictor(seq_output)
                    attribute_labels = self.feat_all[i][pos_items].to(pos_items.device)

                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.feanum[i])
                    # if torch.rand(1)>0.99:
                    #     with open('result.txt','w') as file:
                    #         file.write(str(pos_cate.shape))
                    #         file.write(str(attribute_labels.shape))
                        # print(pos_cate.shape, attribute_labels.shape)
                    if len(attribute_labels.shape) > 2:
                        attribute_labels = attribute_labels.sum(dim=1)
                    attribute_labels = attribute_labels.float()
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
                    attribute_loss = torch.mean(attribute_loss[:, 1:])
                    loss_dic[self.selected_features[i]] = attribute_loss
                if self.num_feature_field == 1:
                    total_loss = loss + self.lamdas[0] * attribute_loss
                    # print('total_loss:{}\titem_loss:{}\tattribute_{}_loss:{}'.format(total_loss, loss,self.selected_features[0],attribute_loss))
                else:
                    for i,attribute in enumerate(self.selected_features):
                        attribute_loss_sum += self.lamdas[i] * loss_dic[attribute]
                    total_loss = loss + attribute_loss_sum
                    loss_dic['total_loss'] = total_loss
                    # s = ''
                    # for key,value in loss_dic.items():
                    #     s += '{}_{:.4f}\t'.format(key,value.item())
                    # print(s)
            else:
                total_loss = loss
            return total_loss


    def full_sort_predict(self, interaction):
        args = self.args
        
        item_seq_batch = interaction['item_id_list']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # feat_seq_batch = self.cateall[item_seq_batch].to(self.args.device)
        # pos_items = interaction['item_id']
        # pos_cate = self.cateall[pos_items]
        # item_seq_len = torch.ones(interaction['item_id'].size()).long().to(item_seq_batch.device)*50

        seq_output = self.forward(item_seq_batch, item_seq_len, None)
        # test_items_emb = self.item_embedding.weight

        allinput = self.item_embedding.weight

        scores = torch.matmul(seq_output, allinput.transpose(0, 1))  # [B, item_num]
        return scores

