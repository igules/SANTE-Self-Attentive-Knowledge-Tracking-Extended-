import argparse
import random
import torch
import sys
import os

parser = argparse.ArgumentParser()


def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)

    return run_script


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    params = parser.parse_args()
    params.run_script = get_run_script()

    if params.gpu != 'none':
        print('gpu: ', params.gpu)
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    # random_seed
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)

    if torch.cuda.is_available():
        params.device = 'cuda'
        params.gpu = list(range(len(params.gpu.split(','))))
        if params.gpu is not None:
            torch.cuda.set_device(params.gpu[0])

    params.weight_path = 'weight/{}/'.format(params.name)
    os.makedirs(params.weight_path, exist_ok=True)

    return params


def print_args(params):
    info = '\n[args]________________________________________________\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += '|─ {} ({})\n'.format(sub_args.title, size_sub)
        for i, arg in enumerate(sub_args._group_actions):
            prefix = 'L__' if i == size_sub-1 else '|─'
            info += '│     {} {}: {}\n'.format(prefix, arg.dest, getattr(params, arg.dest))
    info += '└─────────────────────────────────────────────────────\n'
    print(info)


dataset_list = ['ASSISTments2009', 'ASSISTments2012', 'ASSISTments2015', 'ASSISTmentsChall',
                'STATICS', 'KDDCup', 'Junyi', 'EdNet-KT1']

base_args = parser.add_argument_group('Base args')
base_args.add_argument('--name', type=str, default="name2")
base_args.add_argument('--device', type=str, default='cuda')
base_args.add_argument('--gpu', type=str, default='none')
base_args.add_argument('--num_workers', type=int, default=4)
base_args.add_argument('--resume', type=int, default=1)
# base_args.add_argument('--base_path', type=str, default='/shared/benchmarks/')
# /home/skchoi/SAINT/data/EdNet_KT1/new_processed/train
base_args.add_argument('--base_path', type=str, default='/home/skchoi/SAINT/data/EdNet_KT1/new_processed/processed_KT1/')
base_args.add_argument('--weight_path', type=str)
base_args.add_argument('--dataset_name', type=str, default='ASSISTments2009', choices=dataset_list)

model_list = ['DKT', 'DKVMN', 'NPA', 'SAKT', 'SAINT']

model_args = parser.add_argument_group('Model args')
model_args.add_argument('--model', type=str, default='DKT', choices=model_list)

# DKT, NPA, SAKT
model_args.add_argument('--num_layers', type=int, default=4)
model_args.add_argument('--hidden_dim', type=int, default=128)
model_args.add_argument('--LSTM_hidden_dim', type=int, default=128)
model_args.add_argument('--num_features', type=int, default=4)
model_args.add_argument('--input_dim', type=int, default=100)
model_args.add_argument('--dropout', type=float, default=0.0)
model_args.add_argument('--emb_hidden_dim', type=int, default=128)#512)

# NPA
model_args.add_argument('--attention_dim', type=int, default=128)
model_args.add_argument('--dff_dim', type=int, default=512)#2048)#

# SAKT
model_args.add_argument('--num_head', type=int, default=5)

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--model_name', type=str, default='basemodel')
train_args.add_argument('--random_seed', type=int, default=1)
train_args.add_argument('--num_epochs', type=int, default=10)
train_args.add_argument('--train_batch', type=int, default=256)#32)
train_args.add_argument('--lr', type=float, default=0.0005)#0.001)
train_args.add_argument('--seq_size', type=int, default=100)
train_args.add_argument('--is_neptune', type=int, default=1)
train_args.add_argument('--stddev', type=int, default=0.01)#0.05)#0.01)
train_args.add_argument('--file_name', type=str, default='step_1000_AdamW')#'add_ts_qinfo')#'add_timestamp')#'add_qinfo')#'vector_to_paramters_add')#'after_Gaussian_0.02')# 'weight_Gaussian_results')#
train_args.add_argument('--num_tag', type=int, default=6)

ARGS = get_args()


if __name__ == '__main__':
    ARGS = get_args()
    print_args(ARGS)


# 0 : 500, 1 : 1700
