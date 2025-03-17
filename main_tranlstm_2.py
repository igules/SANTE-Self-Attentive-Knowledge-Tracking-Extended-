# https://www.kaggle.com/choiseulgi/saint-is-all-you-need-training-private-0-f05021/edit
import pandas as pd
import random
import gc
import torch
from model_translstm_2 import SAINTModel
from opts import ScheduledOptim

# import neptune
import psutil
import joblib
import random
import logging
from tqdm import tqdm

import numpy as np
import gc
import pandas as pd
import time

from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score
from sklearn.preprocessing import QuantileTransformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from config import ARGS

import sys
sys.path.append("/home/skchoi/kaggle_KT/data/file")
from preprocess_data import count_trainable_parameters

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    print('cuda is available')

# global variables
global MAX_TIME_STAMP

global BATCH_SIZE 
global DROPOUT
global TIME_CAT_FLAG
# n_skill, n_part = 0, 0

def rand_time(max_time_stamp):
    global MAX_TIME_STAMP

    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0, interval)
    return rand_time_stamp

def feature_time_lag(df, time_dict):

    tt = np.zeros(len(df), dtype=np.int64)
    
    for ind, row in enumerate(df[['user_id','timestamp','task_container_id']].values):

        if row[0] in time_dict.keys():
            if row[2]-time_dict[row[0]][1] == 0:

                tt[ind] = time_dict[row[0]][2]

            else:
                t_last = time_dict[row[0]][0]
                task_ind_last = time_dict[row[0]][1]
                tt[ind] = row[1] - t_last
                time_dict[row[0]] = (row[1], row[2], tt[ind])
        else:
            # time_dict : timestamp, task_container_id, lag_time
            time_dict[row[0]] = (row[1], row[2], -1)
            tt[ind] =  0

    df["time_lag"] = tt
    return df

# train을 userid-taskcontainerid-priorquestion_elapsed_time 단위로 시간을 
def get_pqet(train):
    new_pqet = []

    print("start getting new column")
    train.prior_question_elapsed_time = train.prior_question_elapsed_time.fillna(0)
    sort_train = train[['user_id', 'timestamp', 'task_container_id', 'prior_question_elapsed_time']]
    sort_train = sort_train.groupby(['user_id', 'timestamp', 'task_container_id'])

    print(len(train))
    count = 0
    c = 0

    time = np.zeros(len(train), dtype=np.int64)

    for i, (name, user_time_data) in enumerate(sort_train):
        if i==0:
            count = len(user_time_data)
        else:
            time[c:(c + count)] = user_time_data['prior_question_elapsed_time'].unique()
            c += count
            count = len(user_time_data)

    train['current_question_elapsed_time'] = time
    return train

def cal_acc(outputs, y, mask):
    tot_acc = 0

    outputs = torch.sigmoid(outputs)

    result = []
    for o in outputs:
        if o>=0.5:
            result.append(1)
        else:
            result.append(0)
    sent_leng = len(result)
    result = torch.FloatTensor(result).to(device)
    num_corrects = torch.sum(result == y)

    acc = num_corrects.item()/sent_leng
    # print(acc)
    return acc

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
def train_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, criterion, device="cuda", time_cat_flag = True):
    
    model.train()

    train_loss = []
    train_acc = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    start_time = time.time()
    total_train_loss = 0
    ## training
    for i, item in enumerate(train_dataloader):

        # return q, qa, pri_elap, lag, pri_exp, timestamp, tags, diff, popl, part, label

        exercise = item[0].to(device).long()
        response = item[1].to(device).long()
        part = item[8].to(device).long()

        elapsed_time = item[2].to(device).float()
        lag_time = item[3].to(device).float()

        exp = item[4].long().to(device).long()
        timestamp = item[5].float().to(device).float()
        tags = item[10].int().to(device).long()

        diff = item[6].to(device).float()
        popl = item[7].to(device).float()

        label = item[9].float().to(device).float()
        
        target_mask = (exercise != 0)

        optimizer.zero_grad()
        output = model(1, exercise, part, response, elapsed_time, lag_time, exp, timestamp, diff, popl, tags)

        # output = output.squeeze()
        loss = criterion(output, label)
        # loss = criterion(output.squeeze(), label[:, -1])

        loss.backward()
        optimizer.step()
        
        total_train_loss += loss
        train_loss.append(loss.item())

        # acc = cal_acc(output, label[:, -1], target_mask)
        acc = cal_acc(output[:, -1], label[:, -1], target_mask)
        train_acc.append(acc)
        # print(acc)
        if i > 0 and i%100 == 0:
            print('[Epoch {} - {}] {} {}'.format(epoch+1, i, total_train_loss/100, acc))

            # if ARGS.is_neptune == 1:
            #     neptune.log_metric('train_print_loss', total_train_loss/100)
            total_train_loss = 0

        # mask the output for calculating AUC
        # output_mask = torch.masked_select(output, target_mask)
        # label_mask = torch.masked_select(label, target_mask)

        labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        outs.extend(output[:, -1].view(-1).data.cpu().numpy())
        # labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        # outs.extend(output.view(-1).data.cpu().numpy())
        
    train_auc = roc_auc_score(labels, outs)
    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)

    ## roc graph
    # fpr, tpr, thresholds = roc_curve(labels, outs)
    
    # plot_roc_curve(fpr, tpr)
    # plt.savefig("plot_rocauc.png")
    ##

    labels = []
    outs = []
    val_loss = []
    val_acc = []

    # validation
    model.eval()
    for item in val_dataloader:
        exercise = item[0].to(device).long()
        response = item[1].to(device).long()
        part = item[8].to(device).long()

        elapsed_time = item[2].to(device).float()
        lag_time = item[3].to(device).float()

        exp = item[4].long().to(device).long()
        timestamp = item[5].float().to(device).float()
        tags = item[10].long().to(device).long()

        diff = item[6].to(device).float()
        popl = item[7].to(device).float()

        label = item[9].float().to(device).float()
        
        target_mask = (exercise != 0)

        optimizer.zero_grad()
        output = model(0, exercise, part, response, elapsed_time, lag_time, exp, timestamp, diff, popl, tags)

        ## mask the output
        # output_mask = torch.masked_select(output, target_mask)
        # label_mask = torch.masked_select(label, target_mask)

        acc = cal_acc(output[:, -1], label[:, -1], target_mask)
        # acc = cal_acc(output, label[:, -1], target_mask)
        val_acc.append(acc)

        loss = criterion(output.squeeze(), label)
        # loss = criterion(output.squeeze(), label[:, -1])
        val_loss.append(loss.item())

        labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        outs.extend(output[:, -1].view(-1).data.cpu().numpy())
        # labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        # outs.extend(output.view(-1).data.cpu().numpy())

    val_auc = roc_auc_score(labels, outs)
    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)

    elapsed_time = time.time() - start_time 

    return train_loss, train_auc, val_loss, val_auc, elapsed_time, val_acc, train_acc

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)

class SAINTDataset(Dataset):

    def __init__(self, group, n_skill, max_seq=ARGS.seq_size):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        
        for user_id in group.index:
            # q, qa, part, pri_elap, lag, pri_exp = group[user_id]
            q, qa, pri_elap, pri_exp, lag, timestamp = group[user_id]
            if len(q) < 2:
                continue
            
            # Credit to https://www.kaggle.com/manikanthr5/riiid-sakt-model-training-public
            if len(q) > self.max_seq:
                
                total_questions = len(q)
                initial = total_questions % self.max_seq
                
                if initial >= 2:
                    self.user_ids.append("{user_id}_0".format(user_id=user_id))
                    self.samples["{u}_0".format(u=user_id)] = (q[:initial], qa[:initial], pri_elap[:initial], pri_exp[:initial], 
                                                            lag[:initial], timestamp[:initial])
                    
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append("{user_id}_{seq}".format(user_id=user_id, seq=seq+1))
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples["{user_id}_{seq}".format(user_id=user_id, seq=seq+1)] = (q[start:end], qa[start:end], pri_elap[start:end], pri_exp[start:end], 
                                                                                        lag[start:end], timestamp[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa, pri_elap, pri_exp, lag, timestamp)

        # question dict 불러와서 변수에 저장하기.
        with open('../data/question_info_dict.pkl', 'rb') as f:
            self.qi_dict = pickle.load(f)

        # self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    # 데이터 불러올때마다 
    def __getitem__(self, index):
        user_id = self.user_ids[index]

        ### 1. Load the data
        q_, qa_, pri_elap_, pri_exp_, lag_, timestamp_ = self.samples[user_id]
        seq_len = len(q_)
        res_ = qa_

        ## q_에 맞게 dictionary
        tags_ = np.array([self.qi_dict[q_id]['tags'] for q_id in q_])
        tags_ = np.array([list(map(int, tag.split())) if tag != 'nan' else [0] for tag in tags_])
        # print('tag shape: ', tags_)
        tags_ = get_numpy_from_nonfixed_2d_array(tags_, fixed_length=ARGS.num_tag, padding_value=0)
        # print('TAG2: ', tags_.shape)

        diff_ = np.array([self.qi_dict[q_id]['difficulty'] for q_id in q_])
        popl_ = np.array([self.qi_dict[q_id]['popularity'] for q_id in q_])
        part_ = np.array([self.qi_dict[q_id]['part'] for q_id in q_])

        ## 2. Zero padding
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        res = np.zeros(self.max_seq, dtype=int)
        pri_elap = np.zeros(self.max_seq, dtype=float)
        pri_exp = np.zeros(self.max_seq, dtype=int)
        lag = np.zeros(self.max_seq, dtype=float)
        timestamp = np.zeros(self.max_seq, dtype=float)

        # tags = np.zeros(self.max_seq, dtype=int)
        tags = np.zeros((self.max_seq, ARGS.num_tag), dtype=int)
        diff = np.zeros(self.max_seq, dtype=float)
        popl = np.zeros(self.max_seq, dtype=float)
        part = np.zeros(self.max_seq, dtype=int)

        if seq_len == self.max_seq:

            q[:] = q_
            qa[:] = qa_
            res[:] = res_
            pri_elap[:] = pri_elap_
            pri_exp[:] = pri_exp_
            lag[:] = lag_
            timestamp[:] = timestamp_

            tags[:] = tags_
            diff[:] = diff_
            popl[:] = popl_
            part[:] = part_
            
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            res[-seq_len:] = res_
            pri_elap[-seq_len:] = pri_elap_
            lag[-seq_len:] = lag_
            pri_exp[-seq_len:] = pri_exp_
            timestamp[-seq_len:] = timestamp_

            tags[-seq_len:] = tags_
            diff[-seq_len:] = diff_
            popl[-seq_len:] = popl_
            part[-seq_len:] = part_
    
        pa = torch.Tensor(part)
        # tg = torch.Tensor(tags)
        dif = torch.Tensor(diff)
        pop = torch.Tensor(popl)

        ## for labeling
        label = qa[1:]

        ## shape of feature : (, max_seq)
        return q, qa, pri_elap, lag, pri_exp, timestamp, dif, pop, pa, label, tags



if __name__=='__main__':
    global D_MODEL
    global N_LAYER 
    global TIME_CAT_FLAG

    D_MODEL = 256 
    N_LAYER = 2
    BATCH_SIZE = ARGS.train_batch
    DROPOUT = 0.2
    TIME_CAT_FLAG = False # New step!!
    
    '''
    # ## preprocess
    train_group, val_group, test_group = preprocess_data()

    print('Write data')
    with open('train_group_D_tag.pkl', 'wb') as f:
        pickle.dump(train_group, f)
    with open('val_group_D_tag.pkl', 'wb') as f:
        pickle.dump(val_group, f)
    with open('test_group_D_tag.pkl', 'wb') as f:
        pickle.dump(test_group, f)
    '''
    # with open('../train_group_D.pkl', 'rb') as f:
    #     train_group = pickle.load(f)
    # with open('../val_group_D.pkl', 'rb') as f:
    #     val_group = pickle.load(f)
    # with open('../test_group_D.pkl', 'rb') as f:
    #     test_group = pickle.load(f)
    
    with open('../data/train_group_v3.pkl', 'rb') as f:
        train_group = pickle.load(f)
    with open('../data/val_group_v3.pkl', 'rb') as f:
        val_group = pickle.load(f)
    # with open('../data/test_group_v3.pkl', 'rb') as f:
    #     test_group = pickle.load(f)
    print("Finish loading data.\n")
    
    n_skill = 13523 #train_group["exercise"].max() + 1
    n_part = 7 #len(train_group["part"].unique())
    print('n_skil: ', n_skill, 'n_part: ', n_part)

    ## Data
    train_dataset = SAINTDataset(train_group, n_skill)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    print('length of train_dataset, train_dataloader', len(train_dataset), len(train_dataloader))

    val_dataset = SAINTDataset(val_group, n_skill)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True, drop_last=True)
    print('length of val_dataset, val_dataloader', len(val_dataset), len(val_dataloader))

    # test_dataset = SAINTDataset(test_group, n_skill)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True, drop_last=True)
    # print('length of test_dataset, test_dataloader', len(test_dataset), len(test_dataloader))

    ## Model
    model = SAINTModel(n_skill, n_part, max_seq=ARGS.seq_size, embed_dim= ARGS.hidden_dim, time_cat_flag = TIME_CAT_FLAG)
    print("model's number of parameters: {}".format(count_trainable_parameters(model)))
    """
    # resume_model = 'best_model/add_exp_tag6_AUC80.pt'#'gaussian_weight_resume.pt'#
    # if ARGS.resume == 1:
    #     print('Resuming the model {}'.format(resume_model))
    #     model.load_state_dict(torch.load(resume_model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ## Optimizer Scheduling
    # optim_adam = torch.optim.Adam(model.parameters())
    start = 1000
    # optimizer = ScheduledOptim(optim_adam, 0.1, ARGS.hidden_dim, start)
    optimizer = ScheduledOptim(optimizer, 0.1, ARGS.hidden_dim, start)


    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    epochs = 100

    # if ARGS.is_neptune == 1:
    #     neptune.init(project_qualified_name='timothy/sandbox'), 
    #          api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMjk4YTg1Yi0yYTU5LTRlZTMtOTAyZS05NThjZjViZGFiNzYifQ==',
    #         )
    #     neptune.create_experiment(name='SAINT+model_test_N_lookaheadmask')

    for epoch in range(epochs):
        train_loss, train_auc, val_loss, val_auc, elapsed_time, val_acc, train_acc = train_epoch(epoch, model, train_dataloader, val_dataloader, optimizer, criterion, device, time_cat_flag = TIME_CAT_FLAG)

        # if ARGS.is_neptune == 1:
        #     neptune.log_metric('train_auc', train_auc)
        #     neptune.log_metric('train_acc', train_acc)

        #     neptune.log_metric('val_auc', val_auc)
        #     neptune.log_metric('val_acc', val_acc)

        with open("{}.csv".format(ARGS.file_name), "a") as f:
            f.write("{:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(train_loss, train_auc, val_loss, val_auc))

        print("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f}  val_acc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, val_acc, elapsed_time))

        torch.save(model.state_dict(), "./{}_{}_{}_{}.pt".format(ARGS.file_name, ARGS.num_layers, ARGS.dropout, epoch))
    #     logging.info("epoch - {} train_loss - {:.4f} train_auc - {:.4f} val_loss - {:.4f} val_auc - {:.4f} time={:.2f}s".format(epoch, train_loss, train_auc, val_loss, val_auc, elapsed_time))

    """