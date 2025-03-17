
import pandas as pd
import random
import gc
import torch

import psutil
import random
import logging

import numpy as np
import gc
import pandas as pd
import time

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch.nn.utils import vector_to_parameters,parameters_to_vector

import torch
import torch.nn as nn

# from last_query_model import last_query_model
import pickle
from config import ARGS
from SAINT_translstm_2 import SAINT
# from SAINT_translstm_concat import SAINT

DROPOUT = 0.1
TIME_CAT_FLAG = True

device = 'cuda'

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('double')

    return torch.from_numpy(future_mask)

def gaussian_noise(input, is_training, stddev=0.01, mean=0):
    '''ARGS
    input : numpy array of input embeddings
    '''
    if is_training:  
        noise = torch.FloatTensor(np.random.normal(loc=mean, scale=stddev, size=np.shape(input))).to(device)
        return input + noise
    else:
        return input

class SAINTModel(nn.Module):
    
    def __init__(self, n_skill, n_part, max_seq=ARGS.seq_size, embed_dim= 128, time_cat_flag = True):
        
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part
        self.time_cat_flag = time_cat_flag

        self.e_embedding = nn.Embedding(self.n_skill+1, embed_dim) ## exercise
        self.c_embedding = nn.Embedding(self.n_cat+1, embed_dim) ## category
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim) ## position

        self.res_embedding = nn.Embedding(2+1, embed_dim) ## response

        # tag_embed_dim = 512
        self.tag_embedding = nn.Embedding(293+1, ARGS.hidden_dim)

        print('time_cat_flag: ', time_cat_flag)
        if self.time_cat_flag == True:
            self.elapsed_time_embedding = nn.Embedding(300+1, embed_dim) ## elapsed time (the maximum elasped time is 300)
            self.lag_embedding1 = nn.Embedding(300+1, embed_dim) ## lag time1 for 300 seconds
            self.lag_embedding2 = nn.Embedding(1440+1, embed_dim) ## lag time2 for 1440 minutes
            self.lag_embedding3 = nn.Embedding(365+1, embed_dim) ## lag time3 for 365 days

        else:
            self.elapsed_time_embedding = nn.Linear(1, embed_dim, bias=False) ## elapsed time
            self.ts_embedding = nn.Linear(1, embed_dim, bias=False)
            self.lag_embedding = nn.Linear(1, embed_dim, bias=False) ## lag time
            self.diff_embedding = nn.Linear(1, ARGS.hidden_dim, bias=False) ## difficulty
            self.popl_embedding = nn.Linear(1, ARGS.hidden_dim, bias=False) ## popularity

        self.exp_embedding = nn.Embedding(2+1, embed_dim) ## user had explain

        ## NEW
        self.u_answ = nn.Embedding(4+1, ARGS.hidden_dim) ## user_answer

        # self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= ARGS.num_layers, num_decoder_layers= ARGS.num_layers, dropout = DROPOUT)
        
        self.saint = SAINT(embed_dim, 13532, ARGS.num_layers, 8, ARGS.dropout)

        self.dropout = nn.Dropout(DROPOUT)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        self.max_seq = max_seq

        # self.tag_bag_embedding = nn.EmbeddingBag(293+1, embed_dim, max_norm=1., mode='sum')

        enc_hidden = ARGS.hidden_dim * 4
        dec_hidden = ARGS.hidden_dim * 4
        self.emb_dense_layer1 = nn.Linear(enc_hidden, ARGS.emb_hidden_dim, bias=True)
        self.emb_dense_layer2 = nn.Linear(dec_hidden, ARGS.emb_hidden_dim, bias=True)

        hidden = ARGS.hidden_dim * 6
        self.before_encoder_layer = nn.Linear(hidden, ARGS.hidden_dim, bias=True)
        self.before_decoder_layer = nn.Linear(hidden, ARGS.hidden_dim, bias=True)

        tag_embed_dim = 640
        self.tag_layer = nn.Linear(tag_embed_dim, ARGS.hidden_dim, bias=True)
        
    def forward(self, is_training, question, part, response, elapsed_time, lag_time, exp, timestamp, diff, popl, tags):
        device = question.device  
        q_id = question

        ## <Tag embedding>
        ### 1. Making mask
        tags_mask = (tags[:, 1:, :] != 0).unsqueeze(-1).float()

        ### 2. Embedding & Masking 
        tags = self.tag_embedding(tags[:, 1:, :]) * tags_mask

        ### 3. dim=2 기준으로 sum 해서 차원 없애주기 (b_s, m_s, dim)
        tags = torch.sum(tags, 2) # Sum 하는 방식

        ### 4. nn.Linear에 한번 통과시켜준다. => 성능향상 없음.
        # tags = self.tag_layer(tags)
        
        ### Categorical embedding layer
        question = self.e_embedding(question[:, 1:])
        part = self.c_embedding(part[:, 1:])

        pos_id = np.array([np.arange(self.max_seq-1) for k in range(ARGS.train_batch)])
        pos_id = torch.tensor(pos_id).to(device)
        pos_id = self.pos_embedding(pos_id)

        res = self.res_embedding(response[:, :-1])
        exp = self.exp_embedding(exp[:, 1:])
        
        ## Continuous features Embedding
        # print(elapsed_time[:, :-1].shape)
        elapsed_time = elapsed_time[:, :-1].contiguous().view(-1, 1)
        elapsed_time = self.elapsed_time_embedding(elapsed_time)
        elapsed_time = elapsed_time.view(-1, ARGS.seq_size-1, self.embed_dim)
        
        # lag_time = p_lag_time.view(-1, 1)
        c_lag_time = self.lag_embedding(lag_time[:, 1:].contiguous().view(-1, 1))
        c_lag_time = c_lag_time.view(-1, ARGS.seq_size-1, self.embed_dim)

        p_lag_time = self.lag_embedding(lag_time[:, :-1].contiguous().view(-1, 1))
        p_lag_time = p_lag_time.view(-1, ARGS.seq_size-1, self.embed_dim)

        ## Difficulty
        diff = self.diff_embedding(diff[:, 1:].contiguous().view(-1, 1))
        diff = diff.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        # e_diff = self.diff_embedding(diff[:, 1:].contiguous().view(-1, 1))
        # e_diff = e_diff.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        # d_diff = self.diff_embedding(diff[:, :-1].contiguous().view(-1, 1))
        # d_diff = d_diff.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        
        ## Popularity
        popl = self.popl_embedding(popl[:, 1:].contiguous().view(-1, 1))
        popl = popl.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        # e_popl = self.popl_embedding(popl[:, 1:].contiguous().view(-1, 1))
        # e_popl = e_popl.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        # d_popl = self.popl_embedding(popl[:, :-1].contiguous().view(-1, 1))
        # d_popl = d_popl.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)

        ## Timestamp
        p_timestamp = self.ts_embedding(timestamp[:, :-1].contiguous().view(-1, 1))
        p_timestamp = p_timestamp.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)

        c_timestamp = self.ts_embedding(timestamp[:, 1:].contiguous().view(-1, 1))
        c_timestamp = c_timestamp.view(-1, ARGS.seq_size-1, ARGS.hidden_dim)
        
        '''
        ## Original
        # enc = torch.cat([question, part, res, elapsed_time, c_lag_time, pos_id], dim=2) # (batch, sequence*3, embedding_dim) #

        enc = torch.cat([question, part, c_lag_time, elapsed_time, res], dim=-1)
        dec = torch.cat([question, part, c_lag_time, elapsed_time, res], dim=-1)
        '''

        ### v1 diff, popl, layer추가하기
        # question_info = torch.cat([question, part, pos_id, diff, popl], dim=2)
        # question_info = self.before_encoder_layer(question_info)

        # enc = torch.cat([question_info, pos_id, c_lag_time], dim=2)
        # dec = torch.cat([res, elapsed_time, p_lag_time, pos_id], dim=2) 

        ### v2 timestamp 추가하기.
        # enc = torch.cat([question, part, pos_id, c_lag_time, c_timestamp], dim=2)
        # dec = torch.cat([res, elapsed_time, p_lag_time, pos_id, p_timestamp], dim=2) 

        ### v1+v2 timestamp + 2개의 question feature 추가.
        # question_info = torch.cat([question, part, pos_id, diff, popl], dim=2)
        # question_info = self.before_encoder_layer(question_info)

        # enc = torch.cat([question_info, pos_id, c_lag_time, c_timestamp], dim=2)
        # dec = torch.cat([res, elapsed_time, p_lag_time, pos_id, p_timestamp], dim=2) 

        ### v3 question_info에 [tags], diff, popl 추가.
        question_info = torch.cat([question, part, pos_id, diff, popl, tags], dim=2)
        # question_info = torch.cat([question, part, pos_id, diff, popl], dim=2)
        question_info = self.before_encoder_layer(question_info)
        # e_question_info = torch.cat([e_question, e_part, pos_id, e_diff, e_popl, e_tags], dim=2)
        # e_question_info = self.before_encoder_layer(e_question_info)
        # d_question_info = torch.cat([d_question, d_part, pos_id, d_diff, d_popl, d_tags], dim=2)
        # d_question_info = self.before_decoder_layer(d_question_info)


        # enc = torch.cat([question_info, pos_id, c_lag_time, tags], dim=2) exp
        enc = torch.cat([question_info, pos_id, c_lag_time, exp], dim=2)
        # enc = torch.cat([question_info, pos_id, c_lag_time], dim=2)
        dec = torch.cat([res, elapsed_time, p_lag_time, pos_id], dim=2) 
        # enc = torch.cat([e_question_info, pos_id, c_lag_time], dim=2)
        # dec = torch.cat([res, elapsed_time, p_lag_time, pos_id, d_question_info], dim=2) 

        ### v4 very original
        # enc = torch.cat([question_info, pos_id, c_lag_time], dim=2)
        # # enc = torch.cat([question, part, pos_id, c_lag_time], dim=2)
        # dec = torch.cat([res, elapsed_time, p_lag_time, pos_id], dim=2) 

        ### Adding Gaussian noise to input of my model(Before)------------
        # enc = gaussian_noise(enc, is_training, stddev=0.01, mean=0)
        # dec = gaussian_noise(dec, is_training, stddev=0.01, mean=0)

        enc = self.emb_dense_layer1(enc)
        dec = self.emb_dense_layer2(dec)
        
        ### Adding Gaussian noise to input of my model(After)------------
        enc = gaussian_noise(enc, is_training, stddev=ARGS.stddev, mean=0)
        dec = gaussian_noise(dec, is_training, stddev=ARGS.stddev, mean=0)
        # print("After" , enc[0, 0, :])
        # 4 my transformer model
        
        
        
        # ### Adding Gaussian noise to weights of my model------------
        # # parameters_to_vector(x.parameters())
        # param_vector = parameters_to_vector(self.saint.parameters()).cuda()
        # # print(param_vector.shape)

        # n_params = len(param_vector)
        # noise = torch.distributions.Normal(0, ARGS.stddev).sample_n(n_params).cuda()

        # param_vector.add_(noise)
        # # vector_to_parameters(param_vector, self.saint.parameters())
        # ### ---------------------------------------------------------
        

        att_output = self.saint(enc, dec, q_id) # [bs, s_len, embed]

        '''
        # 5 new last query open source model
        outs = self.lq_model(enc)
       
        return outs    
        '''
        return att_output.squeeze(-1)
