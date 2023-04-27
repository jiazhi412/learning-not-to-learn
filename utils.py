# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import pandas as pd
import torch
from itertools import product

def compute_subAcc_withlogits_binary(logits, target, a):
    # output is logits and predict_prob is probability
    assert logits.shape == target.shape, f"Acc, output {logits.shape} and target {target.shape} are not matched!"
    predict_prob = torch.sigmoid(logits)
    predict_prob = predict_prob.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    # Young
    tmp = a <= 0
    predict_prob_n = predict_prob[tmp.nonzero()]
    target_n = target[tmp.nonzero()]
    Acc_n = (predict_prob_n.round() == target_n).mean()
    tmp = a > 0
    predict_prob_p = predict_prob[tmp.nonzero()]
    target_p = target[tmp.nonzero()]
    Acc_p = (predict_prob_p.round() == target_p).mean()
    return Acc_p, Acc_n


def run(command_template, qos, gpu, *args):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # if not os.path.exists('errors'):
    #     os.makedirs('errors')

    l = len(args)
    job_name_template = '{}'
    for _ in range(l-1):
        job_name_template += '-{}'
    for a in product(*args):
        command = command_template.format(*a)
        job_name = job_name_template.format(*a)
        bash_file = '{}.sh'.format(job_name)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('#!/bin/bash\n')
            OUT.write('#SBATCH --job-name={} \n'.format(job_name))
            OUT.write('#SBATCH --ntasks=1 \n')
            OUT.write('#SBATCH --account=other \n')
            OUT.write(f'#SBATCH --qos={qos} \n')
            OUT.write('#SBATCH --partition=ALL \n')
            OUT.write('#SBATCH --cpus-per-task=4 \n')
            OUT.write(f'#SBATCH --gres=gpu:{gpu} \n')
            OUT.write('#SBATCH --mem={}G \n'.format(16 * gpu))
            OUT.write('#SBATCH --time=5-00:00:00 \n')
            OUT.write('#SBATCH --exclude=vista[03] \n')
            OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
            OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
            OUT.write('source ~/.bashrc\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
            OUT.write('conda activate pytorch\n')
            OUT.write(command)
        qsub_command = 'sbatch {}'.format(bash_file)
        os.system( qsub_command )
        os.system('rm -f {}'.format(bash_file))
        print( qsub_command )
        print( 'Submitted' )

def append_data_to_csv(data,csv_name):
    df = pd.DataFrame(data)
    if os.path.exists(csv_name):
        df.to_csv(csv_name,mode='a',index=False,header=False)
    else:
        df.to_csv(csv_name,index=False)

def save_option_IMDB(option):
    option_path = os.path.join(option.save_dir, option.exp_name, option.IMDB_train_mode, option.IMDB_test_mode, "option.json")
    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_Diabetes(option):
    if option.bias_type == "I":
        option_path = os.path.join(option.save_dir, option.exp_name, option.minority, str(option.minority_size), "option.json")
    elif option.bias_type == "II":
        option_path = os.path.join(option.save_dir, option.exp_name, option.Diabetes_train_mode, option.Diabetes_test_mode, "option.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option(option):
    option_path = os.path.join(option.save_dir, option.exp_name, str(option.color_var), "options.json")
    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def logger_setting(exp_name, color_var, save_dir, debug):
    logger = logging.getLogger(exp_name)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')

    log_out = os.path.join(save_dir, exp_name, color_var, 'train.log')
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

class Timer(object):
    def __init__(self, logger, max_step, last_step=0):
        self.logger = logger
        self.max_step = max_step
        self.step = last_step

        curr_time = time.time()
        self.start = curr_time
        self.last = curr_time

    def __call__(self):
        curr_time = time.time()
        self.step += 1

        duration = curr_time - self.last
        remaining = (self.max_step - self.step) * (curr_time - self.start) / self.step / 3600
        msg = 'TIMER, duration(s)|remaining(h), %f, %f' % (duration, remaining)

        self.last = curr_time
        
import pickle
def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data

def _num_correct_CelebA(outputs, labels):
    preds = torch.sigmoid(outputs)
    # print(preds.size())
    # print(labels.size())
    # print('djsladjad')
    correct = (preds.round().view(-1) == labels.view(-1)).sum()
    # print(correct)
    return correct