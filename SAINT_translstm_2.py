"""
Based on Annotated Transformer from Harvard NLP:
https://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ARGS

import numpy as np
import torch
import torch.nn as nn
import copy

PAD_INDEX = 0
MAX_SEQ = 100
device = 'cuda'

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # key와 query를 곱한거에 masking 해주기!!
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # masking된 weight과 value 곱해주기.
    return torch.matmul(p_attn, value), p_attn
'''

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
             
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.2):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4) # Q, K, V, last
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder block of SAINT, 2-sublayer
    """
    def __init__(self, hidden_dim, num_head, dropout=0.2):
        super().__init__()
        # hidden_dim = hidden_dim * ARGS.num_features
        self._self_attn = MultiHeadedAttention(num_head, hidden_dim, dropout)
        self._ffn = PositionwiseFeedForward(hidden_dim, ARGS.dff_dim, dropout)
        self._layernorms = clones(nn.LayerNorm(hidden_dim, eps=1e-6), 2)

        self.sublayer = clones(SublayerConnection(hidden_dim, dropout), 2)

    def forward(self, x, kv, mask=None):
        
        ## 1st solution
        # x = self.sublayer[0](x, lambda x: self._self_attn(x[:, -1:, :], x, x, mask)) # (q, k, v), x = (b, s, d)
        # print('x: ', x.shape, kv.shape)
        
        x = self.sublayer[0](x, lambda x: self._self_attn(x, kv, kv, mask)) # (q, k, v), x=(b, s, d), (b, s, d*3)
        return self.sublayer[1](x, self._ffn)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class DecoderLayer(nn.Module):
    """
    Single Decoder block of SAINT, 3-sublayer
    """
    def __init__(self, hidden_dim, num_head, dropout):
        super().__init__()
        self._self_attn = MultiHeadedAttention(num_head, hidden_dim*3, dropout)
        self._src_attn =  MultiHeadedAttention(num_head, hidden_dim*3, dropout)
        self._ffn = PositionwiseFeedForward(hidden_dim*3, hidden_dim*3, dropout)
        self._layernorms = clones(nn.LayerNorm(hidden_dim*3, eps=1e-6), 3)

        self.sublayer = clones(SublayerConnection(hidden_dim*3, dropout), 3)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):

        m = memory
        x = self.sublayer[0](x, lambda x: self._self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self._src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self._ffn)


class SAINT(nn.Module):
    """
    Transformer-based
    all hidden dimensions (d_k, d_v, ...) are the same as hidden_dim
    """
    def __init__(self, hidden_dim, question_num, num_layers, num_head, dropout=0.2):
        super().__init__()

        self._question_num = question_num

        # ENCODER/DECODER
        self._encoder_layers = clones(EncoderLayer(hidden_dim, num_head, dropout), num_layers)
        self._decoder_layers = clones(DecoderLayer(hidden_dim, num_head, dropout), num_layers)
        
        self._hidden_dim = hidden_dim * ARGS.num_features
        self.layer_normal = nn.LayerNorm(ARGS.emb_hidden_dim) 
        # self.layer_normal = nn.LayerNorm(ARGS.LSTM_hidden_dim) 
        self._ffn = PositionwiseFeedForward(ARGS.emb_hidden_dim, ARGS.emb_hidden_dim, dropout)
        # self._ffn = PositionwiseFeedForward(ARGS.LSTM_hidden_dim, ARGS.LSTM_hidden_dim, dropout)

        # 1st solution
        # self._LSTM = nn.LSTM(ARGS.emb_hidden_dim, ARGS.LSTM_hidden_dim, num_layers=1, batch_first=True) 
        self.hidden = self.init_hidden()

        self.hidden2tag = nn.Linear(ARGS.emb_hidden_dim, 1) # DNN
        # self.hidden2tag = nn.Linear(self._hidden_dim, 1) # DNN

    def init_hidden(self):  
        # return (torch.zeros(1, ARGS.train_batch, ARGS.lstm_hidden_dim).cuda(),
        #         torch.zeros(1, ARGS.train_batch, ARGS.lstm_hidden_dim).cuda())
        return (torch.zeros(1, ARGS.train_batch, self._hidden_dim*3).cuda(),
                torch.zeros(1, ARGS.train_batch, self._hidden_dim*3).cuda())

    def forward(self, E, D, q_id): # E: current information, D: previous information
        # ENCODER 
        PAD_INDEX = 0
        
        src_mask = get_subsequent_mask(q_id) # lookahead mask
        look_ahead_mask = future_mask(q_id.size(1)).unsqueeze(0) # (1, s_l, s_l)

        padding_mask =  get_pad_mask(q_id, PAD_INDEX) # padding
        # print(padding_mask[0, :], padding_mask[-1, :])

        tgt_mask = get_pad_mask(q_id[:, 1:], PAD_INDEX) & get_subsequent_mask(q_id[:, 1:]) # lookahead mask + padding mask

        for layer in self._encoder_layers:
            E = layer(x=D, kv=E, mask=tgt_mask)
        
        att_output = E.permute(1, 0, 2)
        # print(att_output.shape)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        
        x = self._ffn(att_output)
        x = self.layer_normal(x + att_output)
        
        output = self.hidden2tag(E) # output: [batch_size, s_len]
        
        return output
        
def future_mask(seq_length):
    future_mask = (1 - np.triu(np.ones((seq_length, seq_length)), k=1)).astype('bool')

    return torch.from_numpy(future_mask).to(device)

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2) # 원래는 answer_id에 pad_index랑 똑같은거있었다. 그럼 맞는 정보만 썼다는 말? 

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask