# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import pandas as pd
import torch

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