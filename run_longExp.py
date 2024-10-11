import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model_id', type=str,  default='test', help='model id')
parser.add_argument('--model', type=str, default='FreqMoE',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--in_dataset_augmentation', type=bool , default=False)

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# TDLinear
parser.add_argument('--t_dim', type=int, default=4, help='number of time features')

#FreqMoE
parser.add_argument('--freq_num_blocks', type=int, default=3, help='number of prediction block')
parser.add_argument('--expert_num', type=int, default=3, help='number of expert in Frequency decomposition block')
parser.add_argument('--dropout_freq', type=int, default=0.3, help='Dropout rate between two complex linear layer')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

#Film
parser.add_argument('--ab', type=int, default=2, help='ablation version')

# SCINet
parser.add_argument('--hidden_size', default=1, type=float, help='hidden channel of module')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# FLinear
parser.add_argument('--train_mode', type=int,default=1)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=2)

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Augmentation
parser.add_argument('--aug_method', type=str, default='f_mask', help='f_mask: Frequency Masking, f_mix: Frequency Mixing')
parser.add_argument('--aug_rate', type=float, default=0.5, help='mask/mix rate')
parser.add_argument('--in_batch_augmentation', action='store_true', help='Augmentation in Batch (save memory cost)', default=False)
parser.add_argument('--in_datast_augmentation', action='store_true', help='Augmentation in Dataset', default=False)
parser.add_argument('--closer_data_aug_more', action='store_true', help='Augment times increase for data closer to test set', default=False)
parser.add_argument('--data_size', type=float, default=1, help='size of dataset, i.e, 0.01 represents uses 1 persent samples in the dataset')
parser.add_argument('--aug_data_size', type=int, default=1, help='size of augmented data, i.e, 1 means double the size of dataset')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--wo_original_set', action='store_true', help='without original train set')
parser.add_argument('--test_time_train', type=bool, default=False, help='Affect data division')

# output
parser.add_argument('--save_gating_weights', type=str, help='save gating weights')

# debuggin
parser.add_argument('--show_num_parameters_only', action='store_true', help='show number of parameters only', default=False)

args = parser.parse_args()

args.label_len = args.seq_len

if args.cut_freq == 0:
    args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '专家数测试{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_{}_{}_{}_sd{}_hd{}'.format(
            args.expert_num,
            args.model_id,
            args.model,
            args.data,
            args.data_path,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii, args.batch_size, args.t_dim, args.aug_method, args.aug_rate, args.learning_rate, args.seed,
            args.head_dropout,)

        exp = Exp(args)  # set experiments
        if args.show_num_parameters_only:
            print(f"Number of parameters: {sum(p.numel() for p in exp.model.parameters())}")
            exit()
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        if args.train_mode == 0:
            exp.train(setting, ft=False) # train on xy
        elif args.train_mode == 1:
            exp.train(setting, ft=True) # train on y
        elif args.train_mode == 2:
            exp.train(setting, ft=False)
            print('>>>>>>>start finetuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting, ft=True) # finetune
        
        print('>>>>>>>testing for training losses : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, flag='train')     
        
        print('>>>>>>>testing for vali losses : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, flag='val')        

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_hd{}_fr{}_tsr{}_rr{}_rbr{}_ew{}_bh{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, 
                                                                                                  ii, 
                                                                                                  args.head_dropout, 
                                                                                                  args.use_fixed_routing,
                                                                                                  args.use_time_series_routing,
                                                                                                  args.use_random_routing,
                                                                                                  args.use_random_bypass_routing,
                                                                                                  args.use_equal_weighting_in_testing_time,
                                                                                                  args.use_best_head_in_testing_time)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
