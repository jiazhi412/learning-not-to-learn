# -*- coding: utf-8 -*-
import torch
from torch.backends import cudnn
import os
import random
from trainer_Diabetes import Trainer
from utils import save_option_Diabetes
import argparse
from dataloader.Diabetes import DiabetesDataset 
from dataloader.Diabetes_I import DiabetesDataset_I
from dataloader.Diabetes_II import DiabetesDataset_II

def backend_setting(option):
    if option.bias_type == "I":
        log_dir = os.path.join(option.save_dir, option.exp_name, option.minority, str(option.minority_size))
    elif option.bias_type == "II":
        log_dir = os.path.join(option.save_dir, option.exp_name, option.Diabetes_train_mode, option.Diabetes_test_mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    if not torch.cuda.is_available() and option.cuda:
        option.cuda = False

    if option.cuda:
        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark

    if option.train_baseline:
        option.is_train = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_name',   required=True,              help='experiment name')
    parser.add_argument('--load_path', default='/nas/vista-ssd01/users/jiazli/datasets/Diabetes/Diabetes_newData.csv')
    parser.add_argument('--n_class',          default=1,     type=int,   help='number of classes')
    parser.add_argument('--n_class_bias', default=1, type=int, help='number of bias classes')
    parser.add_argument('--input_size',       default=28,     type=int,   help='input size')
    parser.add_argument('--batch_size',       default=32,    type=int,   help='mini-batch size')
    parser.add_argument('--momentum',         default=0.9,    type=float, help='sgd momentum')
    parser.add_argument('--lr',               default=0.001,   type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate',    default=0.1,    type=float, help='lr decay rate')
    parser.add_argument('--lr_decay_period',  default=40,     type=int,   help='lr decay period')
    parser.add_argument('--weight_decay',     default=0.0005, type=float, help='sgd optimizer weight decay')
    parser.add_argument('--max_step',         default=100,    type=int,   help='maximum step for training')
    parser.add_argument('--depth',            default=20,     type=int,   help='depth of network')
    parser.add_argument('--color_var',        default=0.03,   type=float, help='variance for color distribution')
    parser.add_argument('--seed',             default=2,      type=int,   help='seed index')

    parser.add_argument('--checkpoint',       default=None,               help='checkpoint to resume')
    parser.add_argument('--log_step',         default=50,     type=int,   help='step for logging in iteration')
    parser.add_argument('--save_step',        default=10,     type=int,   help='step for saving in epoch')
    parser.add_argument('--data_dir', default='/nas/vista-ssd01/users/jiazli/datasets/CMNIST/generated_uniform', help='data directory')
    parser.add_argument('--save_dir', default='./results', help='save directory for checkpoint')
    parser.add_argument('--use_pretrain',     action='store_true',        help='whether it use pre-trained parameters if exists')
    parser.add_argument('--train_baseline',   action='store_true',        help='whether it train baseline or unlearning')
    parser.add_argument('--random_seed',                      type=int,   help='random seed')
    parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
    parser.add_argument('--cudnn_benchmark',  default=True,   type=bool,  help='cuDNN benchmark')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('-d', '--debug',      action='store_true',        help='debug mode')
    parser.add_argument('--is_train', default=1, type=int, help='whether it is training')

    ## Census
    parser.add_argument("--bias_attr", type=str, default='age', choices=['sex', 'race', 'age'])
    parser.add_argument("--bias_type", type=str, default='I', choices=['I', 'II', 'General'])

    # Type I Bias
    parser.add_argument("--minority", type=str, default='young')
    parser.add_argument("--minority_size", type=int, default=100)

    # Type II Bias
    parser.add_argument("--Diabetes_train_mode", type=str, default='eb1')
    parser.add_argument("--Diabetes_test_mode", type=str, default='eb2')
    

    option = parser.parse_args()
    opt = vars(option)
    print(option)
    backend_setting(option)
    trainer = Trainer(option)

    print('Note', 'YoungP', 'YoungN', 'OldP', 'OldN')
    # imbalance 
    if opt['bias_type'] == 'I':
        # split 80% for train and 20% for test
        print("Train")
        train_set = DiabetesDataset_I(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    minority=opt['minority'], minority_size=opt['minority_size'], mode='train')
        print("Test")
        test_set = DiabetesDataset_I(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    minority=None, mode='test', balance=True, idx=train_set.get_idx())
        dev_set = test_set

    # association
    if opt['bias_type'] == 'II':
        print("Train")
        train_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_train_mode'])
        print("Val")
        dev_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_test_mode'], idx=train_set.get_idx())
        print("Test")
        test_set = DiabetesDataset_II(path=opt['load_path'], quick_load=True, bias_attr=opt['bias_attr'], middle_age=0, 
                    mode=opt['Diabetes_test_mode'], idx=train_set.get_idx())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=option.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=option.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dev_set, batch_size=option.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if option.is_train:
        save_option_Diabetes(option)
        trainer.train(train_loader, dev_loader)
    else:
        trainer._validate(test_loader)

if __name__ == '__main__': main()
