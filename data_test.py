import pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from config import ARGS

from model_translstm_2 import SAINTModel
from main_tranlstm_2 import SAINTDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')

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
'''   
def cal_acc(outputs, y, mask):
    tot_acc = 0
    # print('before: ', outputs[-20:])
    outputs = torch.sigmoid(outputs)
    # print(outputs[-20:])
    # print('result:', y[-20:])

    for i, (Os, Ys) in enumerate(zip(outputs, y)):
        result = np.array([])

        for o in Os:
            if o >= 0.5:
                result = np.append(1, result)
            else:
                result = np.append(0, result)
        result = torch.FloatTensor(result).to(device)

        num_corrects = torch.sum(result == Ys)
        sent_leng = len(Ys)

        A = num_corrects.item()/sent_leng
        tot_acc += num_corrects.item()/sent_leng

    return tot_acc/256
'''
n_skill = 13523 #train_group["exercise"].max() + 1
n_part = 7 #len(train_group["part"].unique())
device = 'cuda'

DROPOUT = 0.0
TIME_CAT_FLAG = False

## load data
with open('../data/val_group_v3.pkl', 'rb') as f:
    val_group = pickle.load(f)
# with open('val_group_data.pkl', 'rb') as f:
#     val_group = pickle.load(f)
with open('../data/test_group_v3.pkl', 'rb') as f:
    test_group = pickle.load(f)

val_dataset = SAINTDataset(val_group, n_skill)
val_dataloader = DataLoader(val_dataset, batch_size=ARGS.train_batch, shuffle=False, pin_memory=True, drop_last=True)

test_dataset = SAINTDataset(test_group, n_skill)
test_dataloader = DataLoader(test_dataset, batch_size=ARGS.train_batch, shuffle=False, pin_memory=True, drop_last=True)
print('length of val_dataset, val_dataloader', len(test_dataset), len(test_dataloader))
'''
for i, user_id in enumerate(val_group2.index):
    if i<3:
        q, qa, part, pri_elap, lag, pri_exp = val_group2[user_id]
        print(user_id, '\n', pri_elap)
        
    else:
        break

for i, user_id in enumerate(val_group.index):
    if i<3:
        q, qa, part, pri_elap, lag, pri_exp = val_group[user_id]
        print(user_id, '\n', pri_elap)
        
    else:
        break
'''
'''
## check same group
tr_g = set()
va_g = set()
te_g = set()

for i, key in enumerate(val_group.index):
    va_g.add(key)
for i, key in enumerate(train_group.index):
    te_g.add(key)

c = te_g.intersection(va_g)
print(len(va_g), len(te_g))
print('length of intersection of validate and test data: ', len(c))
'''

## load model
model = SAINTModel(n_skill, n_part, max_seq=ARGS.seq_size, embed_dim= ARGS.hidden_dim, time_cat_flag = TIME_CAT_FLAG).to(device)
model.load_state_dict(torch.load('add_taglinear_4_0.0_21.pt'))

model.eval()
criterion = nn.BCEWithLogitsLoss()

## test
dataset = {'test' : test_dataloader, 'val' : val_dataloader}

for k in dataset.keys():
    labels = []
    outs = []
    test_loss = []
    test_acc = []
    sig_outs = []

    for e, item in enumerate(dataset[k]):
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

        output = model(0, exercise, part, response, elapsed_time, lag_time, exp, timestamp, diff, popl, tags)

        ## mask the output
        # output_mask = torch.masked_select(output, target_mask)
        # label_mask = torch.masked_select(label, target_mask)

        ## ACC
        # outputs = torch.sigmoid(output) # new
        # labelss = label # new
        outputs = torch.sigmoid(output[:, -1]) # new
        labelss = label[:, -1] # new

        result = []
        # O = outputs[:, -1]
        # print(O.shape)
        for o in outputs:
            if o>=0.5:
                result.append(1)
            else:
                result.append(0)
        sent_leng = len(result)
        result = torch.FloatTensor(result).to(device)
        num_corrects = torch.sum(result == labelss)

        acc = num_corrects.item()/sent_leng

        # acc = cal_acc(output, label, target_mask)
        test_acc.append(acc)

        ## loss
        loss = criterion(output.squeeze(), label)
        test_loss.append(loss.item())

        ## f1_score
        # output_mask = torch.sigmoid(output_mask)
        # sig_outs.extend(1 if oo>=0.5 else 0 for oo in output_mask)

        ## AUC
        # labels.extend(label_mask.view(-1).data.cpu().numpy())
        # outs.extend(output_mask.view(-1).data.cpu().numpy())
        labels.extend(labelss.view(-1).data.cpu().numpy())
        # outs.extend(output[:, -1].view(-1).data.cpu().numpy())
        outs.extend(output[:, -1].view(-1).data.cpu().numpy())

        if (e != 0) and (e % 100 == 0):
        #     result = []
            # label_mask = torch.sigmoid(label_mask)
            # print(outs[-10:], labels[-10:])
            print('ACC:', acc, ' loss:', loss.item())#, 'f1_score:', test_f1score)
        #     o = torch.sigmoid(output_mask[-10:])

        #     for oo in o:
        #         result.append(1 if oo>=0.5 else 0)
        #         # if oo>=0.5:
        #         #     result.append(1 if oo>=0.5 else 0)
        #         # else:
        #         #     result.append(0)
        #     print(result, 'label:', label_mask[-10:])
    
    # labels = torch.sigmoid(labels)
    test_auc = roc_auc_score(labels, outs)
    # test_f1score = f1_score(labels, sig_outs)
    test_loss = np.mean(test_loss)
    test_acc = np.mean(test_acc)

    print('{} loss: {:.4f}, acc: {:.4f}, auc: {:.4f}'.format(k, test_loss, test_acc, test_auc))

    # roc graph
    fpr, tpr, thresholds = roc_curve(labels, outs)

    plot_roc_curve(fpr, tpr)
    plt.savefig("plot_{}_rocauc.png".format(k))

    print('\n')
