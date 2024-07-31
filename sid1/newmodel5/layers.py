# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2020/8/24 14:58, 2020/9/16, 2020/9/21, 2020/10/9
# @Author : Yujie Lu, Xingyu Pan, Zhichao Feng, Hui Wang
# @Email  : yujielu1998@gmail.com, panxy@ruc.edu.cn, fzcbupt@gmail.com, hui.wang@ruc.edu.cn

"""
recbole.model.layers
#############################
Common Layers in recommender system
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_

from recbole.utils import FeatureType, FeatureSource

import torch.nn.functional as fn
class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
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

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights

class VanillaAttention1(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor, boost):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1)+boost, dim=-1)
        if torch.rand(1)<0.01:
            print(energy.squeeze(-1), boost, weights)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        # hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return  weights


class VanillaAttention2(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor, boost):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)+torch.softmax(boost, dim=-1)
        if torch.rand(1)<0.01:
            print(torch.softmax(energy.squeeze(-1), dim=-1)+torch.softmax(boost, dim=-1), weights)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        # hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return  weights



class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def transpose(self, x):
        new_shape = x.size()[:-1]+(size1, size2)
        x = x.view(*new_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DisentangledSelfAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        # self.num_attention_heads = config.num_attention_heads
        # _attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        # self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'c2p').lower().split('|')] # c2p|p2c
        # self.relative_attention = getattr(config, 'relative_attention', False)

        # if self.relative_attention:
        #     self.position_buckets = getattr(config, 'position_buckets', -1)
        #     self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
        #     if self.max_relative_positions <1:
        #         self.max_relative_positions = config.max_position_embeddings
        #     self.pos_ebd_size = self.max_relative_positions
        #     if self.position_buckets>0:
        #         self.pos_ebd_size = self.position_buckets
        #         # For backward compitable

        #     self.pos_dropout = StableDropout(config.hidden_dropout_prob)

        #     if (not self.share_att_key):
        #         if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    
        #         if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads).float()
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads).float()
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = 1/math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)*scale)
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True).values.detach()).to(hidden_states)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

        # bxhxlxd
        _attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(_attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {
            'hidden_states': context_layer,
            'attention_probs': _attention_probs,
            'attention_logits': attention_scores
            }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, \
                max_position = self.max_relative_positions, device=query_layer.device)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2).to(query_layer)*scale)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1).squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_query_layer.size(-1)*scale_factor)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(pos_query_layer.to(key_layer)*scale, key_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-2, index=c2p_pos)
            score += p2c_att

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
          v1_proj = state_dict[prefix+'in_proj.weight']
          v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
          q,k,v=v1_proj.chunk(3, dim=1)
          state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
          state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
          v1_query_bias = state_dict[prefix + 'q_bias']
          state_dict[prefix + 'query_proj.bias'] = v1_query_bias
          v1_value_bias = state_dict[prefix +'v_bias']
          state_dict[prefix + 'value_proj.bias'] = v1_value_bias

          v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
          state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
          v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
          state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
          v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
          state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
          state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']

          del state_dict[prefix + 'in_proj.weight']
          del state_dict[prefix + 'q_bias']
          del state_dict[prefix + 'v_bias']
          del state_dict[prefix + 'pos_proj.weight']
          del state_dict[prefix + 'pos_q_proj.weight']
          del state_dict[prefix + 'pos_q_proj.bias']

class novaMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(novaMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask, value_input):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(value_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + value_input)

        return hidden_states


class novaMultiHeadAttention1(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(novaMultiHeadAttention1, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, key_tensor, attention_mask, value_input):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + value_input)

        return hidden_states
    


class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len*(2+self.feat_num), self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table[i].squeeze(-2)
            # print(attribute_tensor.shape)
            # print(attribute_query(attribute_tensor).shape)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )

        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

        # all_encoder_layers = []
        # mid_states = hidden_states
        # for layer_module in self.layer:
        #     mid_states = layer_module(mid_states, attention_mask, hidden_states)
        #     if output_all_encoded_layers:
        #         all_encoder_layers.append(mid_states)
        # if not output_all_encoded_layers:
        #     all_encoder_layers.append(mid_states)
        # return all_encoder_layers


class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=[64],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Decoder(nn.Module):
        def __init__(self, hidden_size, n_heads, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
            super(Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.n_heads = n_heads
            
            self.num_attention_heads = self.n_heads
            self.attention_head_size = int(self.hidden_size / self.n_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(self.hidden_size, self.all_head_size)
            self.key = nn.Linear(self.hidden_size, self.all_head_size)
            self.value = nn.Linear(self.hidden_size, self.all_head_size)

            self.attn_dropout = nn.Dropout(attn_dropout_prob)
            self.LayerNorm = nn.LayerNorm(self.hidden_size, layer_norm_eps)
            self.out_dropout = nn.Dropout(hidden_dropout_prob)
        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        def forward(self, input, value_input, extended_attention_mask):
            mixed_query_layer = self.query(input)
            mixed_key_layer = self.key(input)
            mixed_value_layer = self.attn_dropout(value_input)


            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

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




class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.token_embedding_table = {}
        self.float_embedding_table = {}
        self.token_seq_embedding_table = {}

        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field.

        """
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}

        for type in self.types:
            for field_name in self.field_names[type]:
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features.

        """
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array((0, *np.cumsum(self.token_field_dims[type])[:-1]),
                                                          dtype=np.long)
                self.token_embedding_table[type] = FMEmbedding(
                    self.token_field_dims[type], self.token_field_offsets[type], self.embedding_size
                ).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                self.float_embedding_table[type] = nn.Embedding(
                    np.sum(self.float_field_dims[type], dtype=np.int32), self.embedding_size
                ).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(self.device)
                    )

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[-1]
        # [batch_size, max_item_length, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, max_item_length, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table[type](index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(-1))

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == 'item':
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, max_item_length, seq_len, embed_dim]
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9
                result = torch.max(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
                result = result.values
            elif self.pooling_mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
            elif self.pooling_mode == 'raw':
                return token_seq_embedding, mask
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    def embed_input_fields(self, user_idx, item_idx):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {'user': self.user_feat, 'item': self.item_feat}
        user_item_idx = {'user': user_idx, 'item': item_idx}
        float_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(feature if len(feature.shape) == (2 + (type == 'item')) else feature.unsqueeze(-1))
            if len(float_fields) > 0:
                float_fields = torch.cat(float_fields, dim=-1)  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            # [batch_size, max_item_length, num_float_field]
            # or [batch_size, max_item_length, num_float_field, embed_dim] or None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(token_fields, dim=-1)  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(token_seq_fields, type)

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat([token_fields_embedding[type], token_seq_fields_embedding[type]],
                                                       dim=-2)
            dense_embedding[type] = float_fields_embedding[type]

        # sparse_embedding[type]
        # shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type]
        # shape: [batch_size, max_item_length, num_float_field]
        #     or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx):
        return self.embed_input_fields(user_idx, item_idx)


class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            'user': list(self.user_feat.interaction.keys()),
            'item': list(self.item_feat.interaction.keys())
        }

        self.types = ['user', 'item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super(FeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {'item': selected_features}

        self.types = ['item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum', 'raw']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()



from recbole.utils import FeatureType, FeatureSource
from torch import nn
class newFeatureSeqEmbLayer(nn.Module):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super(newFeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names =  selected_features
        # self.types = ['item']
        self.feat_emb_layers = []
        self.num_feature_field = 0
        for i, field_name in enumerate(self.field_names):
            
            if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                # self.token_field_names[type].append(field_name)
                # self.token_field_dims[type].append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.embedding_size[i], padding_idx=0).to(self.device))
            elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                # self.token_seq_field_names[type].append(field_name)
                # self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                self.feat_emb_layers.append(nn.Embedding(self.dataset.num(field_name), self.embedding_size[i], padding_idx=0).to(self.device))
            else:
                raise NotImplementedError

            self.num_feature_field += 1
        
        # self.token_embedding = 
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum', 'raw']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
    def forward(self, item_seq):
        # dataset = self.dataset
        feature_emb= []
        for i in range(len(self.field_names)):
            if self.dataset.field2type[self.field_names[i]] == FeatureType.TOKEN:
                feature_emb.append(self.feat_emb_layers[i](self.item_feat[self.field_names[i]][item_seq]))

            elif self.dataset.field2type[self.field_names[i]] == FeatureType.TOKEN_SEQ:
                feature_emb.append(torch.sum(self.feat_emb_layers[i](self.item_feat[self.field_names[i]][item_seq]), dim=-2))
        return feature_emb




class AttackRMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
                 use_order, use_distance):
        super(AttackRMultiHeadAttention, self).__init__()
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

        self.use_order = use_order
        self.use_distance = use_distance
        if self.use_order:
            self.order_affine = nn.Linear(2 * self.attention_head_size,1)
            self.activation = nn.Sigmoid()
        if self.use_distance:
            self.distance_affine = nn.Linear(2 * self.attention_head_size,1)
            self.scalar = nn.Parameter(torch.randn(1))

        self.attack_query_transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.attack_key_transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print(x.shape)
        return x

    def cal_attack_mask(self, query, key, attention_mask):
        mixed_query_layer = self.attack_query_transform(query)
        mixed_key_layer = self.attack_key_transform(key)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size

        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        return attention_probs

    def cal_adjusted_outputs(self, attention_probs, input_tensor, value_layer):
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def cal_origin_qkv(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        # print(input_tensor.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)

        origin_attention_scores = attention_scores
        max_seq_len = input_tensor.shape[-2]
        batch_size = input_tensor.shape[0]
        '''
            yqc modify here 0725
            add rich attention
        '''
        # generate concatenation
        key_layer_ = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 1, 3)
        q_vec = query_layer.unsqueeze(3).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        k_vec = key_layer_.unsqueeze(2).expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len, -1)
        q_k = torch.cat((q_vec, k_vec), dim=-1)

        error_order = torch.zeros(attention_scores.shape).to(attention_scores.device)
        error_distance = torch.zeros(attention_scores.shape).to(attention_scores.device)

        if self.use_order:
            # Generate order ground truth
            gd_order = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).unsqueeze(0).unsqueeze(0)
            gd_order = gd_order.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_order = self.activation(self.order_affine(q_k).squeeze(-1))
            error_order = torch.log(pr_order + 1e-24) * gd_order + torch.log(1 - pr_order + 1e-24) * (1 - gd_order)
        if self.use_distance:
            row_index = torch.arange(0, max_seq_len, 1).unsqueeze(0).expand((max_seq_len, max_seq_len))
            col_index = torch.arange(0, max_seq_len, 1).unsqueeze(1).expand((max_seq_len, max_seq_len))
            gd_distance = torch.log(torch.abs(row_index - col_index) + 1).unsqueeze(0).unsqueeze(0)
            gd_distance = gd_distance.expand(batch_size, self.num_attention_heads, max_seq_len, max_seq_len).to(
                input_tensor.device)
            pr_distance = self.distance_affine(q_k).squeeze(-1)
            error_distance = -torch.square(gd_distance - pr_distance) * torch.square(self.scalar) / 2

        attention_scores = attention_scores + error_order + error_distance

        def _func(_scores):
            _scores = _scores / self.sqrt_attention_head_size

            _scores = _scores + attention_mask
            _prob = self.softmax(_scores)
            _prob = self.attn_dropout(_prob)
            return _prob

        attention_probs = _func(attention_scores)
        origin_attention_probs = _func(origin_attention_scores)

        return mixed_query_layer, mixed_key_layer, value_layer, attention_probs, origin_attention_probs



class AttackRTransformerLayer(nn.Module):
    def __init__(
            self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps, combine_option='fixed', use_order=True, use_distance=True, two_level=True,
            rich_calibrated_combine='fixed', seq_length=50
    ):
        super(AttackRTransformerLayer, self).__init__()
        self.hidden_size = hidden_size

        self.attack_attention = AttackRMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
            use_order=use_order, use_distance=use_distance
        )
        self.two_level = two_level
        self.rich_calibrated_combine = rich_calibrated_combine
        if self.rich_calibrated_combine == 'trainable':
            self.rich_calibrated_combine_ratio = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.combine_option = combine_option
        if self.combine_option == 'gate':
            self.gate = torch.nn.Linear(hidden_size, seq_length)
        self.combine_ratio = 0.5
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)
        self.anneal_step = 0

    def combine_attention(self, origin_attention, calibrated_attention, query):
        if self.combine_option == 'fixed':
            combined_attention = torch.softmax(origin_attention + self.combine_ratio * calibrated_attention, dim=-1)
        elif self.combine_option == 'gate':
            combine_gate = torch.sigmoid(self.gate(query)).unsqueeze(1)
            combined_attention = combine_gate * origin_attention + (1 - combine_gate) * calibrated_attention
        elif self.combine_option == 'annealing':
            adjust_rate = np.exp(-self.anneal_step / 100000)
            self.anneal_step += 1
            combined_attention = adjust_rate * origin_attention + (1 - adjust_rate) * calibrated_attention

        else:
            raise KeyError
        return combined_attention

    def forward(self, hidden_states, attention_mask, return_attention_prob=False, return_all_attention_prob=False):
        all_attention_prob = {
            'before_spatial': None,
            'after_spatial': None,
            'perturbed_mask': None,
            'perturbed_attention': None,
            'calibrated_attention': None
        }
        mixed_query, mixed_key, value, after_rich_attn_probs, before_rich_attention_probs = \
            self.attack_attention.cal_origin_qkv(hidden_states, attention_mask)

        all_attention_prob['before_spatial'] = before_rich_attention_probs
        all_attention_prob['after_spatial'] = after_rich_attn_probs
        if self.two_level:
            origin_attn_probs = after_rich_attn_probs
        else:
            origin_attn_probs = before_rich_attention_probs
        attack_mask = self.attack_attention.cal_attack_mask(mixed_query, mixed_key, attention_mask)
        all_attention_prob['perturbed_mask'] = attack_mask
        noise = torch.randn(attack_mask.shape).to(attack_mask.device)
        attacked_attention_prob = origin_attn_probs * attack_mask + noise * (1 - attack_mask)
        attacked_attention_prob = torch.softmax(attacked_attention_prob + attention_mask, dim=-1)
        calibrated_attention_prob = origin_attn_probs * torch.exp(1 - attack_mask)
        calibrated_attention_prob = torch.softmax(calibrated_attention_prob + attention_mask, dim=-1)
        combined_attention_prob = self.combine_attention(origin_attn_probs,
                                                         calibrated_attention_prob,
                                                         mixed_query)
        combined_attention_prob = torch.softmax(combined_attention_prob + attention_mask, dim=-1)
        all_attention_prob['perturbed_attention'] = attacked_attention_prob
        all_attention_prob['calibrated_attention'] = combined_attention_prob

        if not self.two_level:
            if self.rich_calibrated_combine == 'fixed':
                combined_attention_prob = (combined_attention_prob + after_rich_attn_probs) / 2
            elif self.rich_calibrated_combine == 'trainable':
                combined_attention_prob = self.rich_calibrated_combine_ratio * combined_attention_prob + \
                                          (1 - self.rich_calibrated_combine_ratio) * after_rich_attn_probs
            else:
                raise KeyError

        attacked_attention_output = self.attack_attention.cal_adjusted_outputs(attacked_attention_prob,
                                                                               hidden_states,
                                                                               value)

        calibrated_attention_output = self.attack_attention.cal_adjusted_outputs(combined_attention_prob,
                                                                                 hidden_states,
                                                                                 value)

        attacked_feedforward_output = self.feed_forward(attacked_attention_output)
        calibrated_feedforward_output = self.feed_forward(calibrated_attention_output)
        # if return_attention_prob:
        if return_all_attention_prob:
            return attacked_feedforward_output, calibrated_feedforward_output, attack_mask, combined_attention_prob, all_attention_prob
        return attacked_feedforward_output, calibrated_feedforward_output, attack_mask, combined_attention_prob

        # return attacked_feedforward_output, calibrated_feedforward_output, attack_mask



class AttackRTransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            combine_option='fixed',
            use_order=True,
            use_distance=True,
            two_level=True,
            rich_calibrated_combine='fixed',
            seq_length=50
    ):

        super(AttackRTransformerEncoder, self).__init__()
        layer = AttackRTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            combine_option, use_order=use_order, use_distance=use_distance, two_level=two_level,
            rich_calibrated_combine=rich_calibrated_combine, seq_length=seq_length
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask,
                output_all_encoded_layers=True, return_attention_prob=False, return_all_attention_prob=False):
        all_encoder_layers = []
        attacked_hidden_states = None
        calibrated_hidden_states = None
        all_attack_masks = []
        all_attention_prob = [] if return_attention_prob else None
        all_probs = [] if return_all_attention_prob else None
        for layer_idx, layer_module in enumerate(self.layer):
            if return_all_attention_prob:
                attacked_hidden_states, calibrated_hidden_states, attack_mask, combined_attention_prob, all_prob = \
                    layer_module(hidden_states, attention_mask, return_attention_prob, return_all_attention_prob)
            else:
                attacked_hidden_states, calibrated_hidden_states, attack_mask, combined_attention_prob = \
                    layer_module(hidden_states, attention_mask, return_attention_prob)
            hidden_states = calibrated_hidden_states
            # if return_attention_prob:
            # else:
            #     attacked_hidden_states, calibrated_hidden_states, attack_mask = \
            #         layer_module(hidden_states, attention_mask, return_attention_prob)
            # print(attack_mask.shape)
            all_attack_masks.append(attack_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
            if return_attention_prob:
                all_attention_prob.append(combined_attention_prob)
            if return_all_attention_prob:
                all_probs.append(all_prob)
        if not output_all_encoded_layers:
            all_encoder_layers.append((attacked_hidden_states, calibrated_hidden_states))
        if return_all_attention_prob:
            return all_encoder_layers, all_attack_masks, all_probs
        if return_attention_prob:
            return all_encoder_layers, all_attack_masks, all_attention_prob
        return all_encoder_layers, all_attack_masks

