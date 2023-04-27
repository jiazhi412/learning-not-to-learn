# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models

import time
import os
import math

from utils import logger_setting, Timer
import utils
import datetime


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)



class Trainer(object):
    def __init__(self, option):
        self.option = option
        self._build_model()
        self.opt = vars(self.option)
        self._set_optimizer()
        if option.bias_type == "I":
            self.save_dir = os.path.join(option.save_dir, option.exp_name, option.minority, str(option.minority_size))
        elif option.bias_type == "II":
            self.save_dir = os.path.join(option.save_dir, option.exp_name, option.Diabetes_train_mode, option.Diabetes_test_mode)
        self.exp_dir = os.path.join(option.save_dir, option.exp_name)
        # self.logger = logger_setting(self.exp_dir)

    def _build_model(self):
        # set network
        in_dim = 8
        # hidden_dims = [64]
        # self.encoder = models.MLP(in_dim, [], 64).cuda()
        # self.biasPredictor = models.MLP(64, [], self.option.n_class_bias).cuda()
        # self.classPredictor = models.MLP(64, [], self.option.n_class).cuda()
        hidden_dims = [32, 10]
        self.encoder = models.MLP(in_dim, [hidden_dims[0]], hidden_dims[1], is_logits=False).cuda()
        self.biasPredictor = models.MLP(hidden_dims[1], [], self.option.n_class_bias).cuda()
        self.classPredictor = models.MLP(hidden_dims[1], [], self.option.n_class).cuda()

        self.bias_loss = nn.BCEWithLogitsLoss().cuda()
        self.loss = nn.BCEWithLogitsLoss().cuda()

        if self.option.cuda:
            self.encoder.cuda()
            self.biasPredictor.cuda() 
            self.classPredictor.cuda()
            self.loss.cuda()
            self.bias_loss.cuda()

    def _set_optimizer(self):
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, list(self.encoder.parameters())), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_class = optim.SGD(filter(lambda p: p.requires_grad, list(self.classPredictor.parameters())), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_bias = optim.SGD(filter(lambda p: p.requires_grad, list(self.biasPredictor.parameters())), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)

        #TODO: last_epoch should be the last step of loaded model
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_class = optim.lr_scheduler.LambdaLR(self.optim_class, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_bias = optim.lr_scheduler.LambdaLR(self.optim_bias, lr_lambda=lr_lambda, last_epoch=-1)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _initialization(self):
        # self.net.apply(self._weights_init)

        if self.option.is_train and self.option.use_pretrain:
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("Pre-trained model not provided")

    def _mode_setting(self, is_train=True):
        if is_train:
            self.encoder.train()
            self.biasPredictor.train() 
            self.classPredictor.train()
        else:
            self.encoder.eval()
            self.biasPredictor.eval() 
            self.classPredictor.eval()

    def _train_step(self, data_loader, step):
        _lambda = 0.01

        for i, (images,labels,color_labels) in enumerate(data_loader):
            
            images = self._get_variable(images)
            color_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            self.optim_class.zero_grad()
            self.optim_bias.zero_grad()
            feat_label = self.encoder(images)
            pred_label = self.classPredictor(feat_label)
            bias_label = self.biasPredictor(feat_label)

            # loss for self.net
            # loss_pred = self.loss(pred_label, torch.squeeze(labels))
            loss_pred = self.loss(pred_label, labels.float())

            softmax = nn.Softmax(dim=1)
            bias_label = softmax(bias_label)
            loss_pred_ps_color = torch.mean(torch.sum(bias_label*torch.log(bias_label),1))
            loss = loss_pred + loss_pred_ps_color*_lambda

            loss.backward()
            self.optim.step()
            self.optim_class.step()

            self.optim.zero_grad()
            self.optim_class.zero_grad()
            self.optim_bias.zero_grad()

            # feat_label, pred_label = self.net(images)
            feat_label = self.encoder(images)
            pred_label = self.classPredictor(feat_label)

            feat_bias = grad_reverse(feat_label)
            pred_bias = self.biasPredictor(feat_bias)

            # loss for rgb predictors
            loss_pred_color = self.bias_loss(pred_bias, color_labels.float())

            loss_pred_color.backward()
            self.optim.step()
            self.optim_class.step()
            self.optim_bias.step()

            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f, bias : %.6f, MI : %.6f  (epoch %d.%02d)" \
                       % (loss_pred,loss_pred_color,loss_pred_ps_color,step,int(100*i/data_loader.__len__()))
                # self.logger.info(msg)
        return loss_pred, loss_pred_color, loss_pred_ps_color


    def _train_step_baseline(self, data_loader, step):
        for i, (images,labels,color_labels) in enumerate(data_loader):
            
            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            feat_label = self.encoder(images)
            pred_label = self.classPredictor(feat_label)

            # loss for self.net
            # loss_pred = self.loss(pred_label, torch.squeeze(labels))
            loss_pred = self.loss(pred_label, labels.float())
            loss_pred.backward()
            self.optim.step()
            self.optim_class.step()

            # TODO: print elapsed time for iteration
            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f (epoch %d.%02d)" \
                       % (loss_pred,step,int(100*i/data_loader.__len__()))

    def _validate(self, data_loader):
        self._mode_setting(is_train=False)
        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        output_list = []
        target_list = []
        a_list = []
        for i, (images,labels,color_labels) in enumerate(data_loader):
            
            start_time = time.time()
            images = self._get_variable(images)
            colro_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            feat_label = self.encoder(images)
            pred_label = self.classPredictor(feat_label)
            loss = self.loss(pred_label, labels.float())
            
            batch_size = images.shape[0]
            total_num_correct += utils._num_correct_CelebA(pred_label, labels).item()
            total_loss += loss.item()*batch_size
            total_num_test += batch_size
            output_list.append(pred_label)
            target_list.append(labels)
            a_list.append(color_labels)
        test_output, test_target, test_a = torch.cat(output_list), torch.cat(target_list), torch.cat(a_list)
        test_acc_p, test_acc_n = utils.compute_subAcc_withlogits_binary(test_output, test_target, test_a)
        D = test_acc_p - test_acc_n
        avg_loss = total_loss/total_num_test
        avg_acc = total_num_correct/total_num_test
        msg = "EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d)" % \
                        (avg_loss,avg_acc,int(total_num_correct),total_num_test)
        # self.logger.info(msg)
        return avg_acc, test_acc_p, test_acc_n, D

    def _num_correct(self,outputs,labels,topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct

    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step):
        if self.option.bias_type == "I":
            filename = os.path.join(self.option.save_dir, self.option.exp_name, self.option.minority, str(self.option.minority_size), 'checkpoint_step_%04d.pth' % step)
        elif self.option.bias_type == "II":
            filename = os.path.join(self.option.save_dir, self.option.exp_name, self.option.Diabetes_train_mode, self.option.Diabetes_test_mode, 'checkpoint_step_%04d.pth' % step)
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'optim_class_state_dict': self.optim_class.state_dict(),
            'optim_bias_state_dict': self.optim_bias.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'biasPredictor_state_dict': self.biasPredictor.state_dict(),
            'classPredictor_state_dict': self.classPredictor.state_dict()
        }, filename)
        print('checkpoint saved. step : %d'%step)

    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.biasPredictor.load_state_dict(ckpt['biasPredictor_state_dict'])
        self.classPredictor.load_state_dict(ckpt['classPredictor_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        self.optim_class.load_state_dict(ckpt['optim_class_state_dict'])
        self.optim_bias.load_state_dict(ckpt['optim_bias_state_dict'])

    def train(self, train_loader, val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        # timer = Timer(self.logger, self.option.max_step)
        start_epoch = 0
        final_acc = 0
        for step in range(start_epoch, self.option.max_step):
            self._mode_setting(is_train=True)
            if self.option.train_baseline:
                self._train_step_baseline(train_loader, step)
            else:
                loss_pred, loss_pred_color, loss_pred_ps_color = self._train_step(train_loader,step)
            self.scheduler.step()
            self.scheduler_class.step()
            self.scheduler_bias.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step-1):
                if val_loader is not None:
                    # acc = self._validate(val_loader)
                    acc, test_acc_p, test_acc_n, D = self._validate(val_loader)
                    if not math.isnan(loss_pred_color) and not math.isnan(loss_pred_ps_color):
                        final_acc = acc
                self._save_model(step)

       # Output the mean AP for the best model on dev and test set
        if self.option.bias_type == "I":
            data = {
                'Time': [datetime.datetime.now()],
                'Bias': [self.opt['bias_attr']],
                'Minority': [self.opt['minority']],
                'Minority_size': [self.opt['minority_size']],
                'Test_acc_old': [test_acc_p*100],
                'Test_acc_young': [test_acc_n*100],
                'D': [D*100],
                }
            utils.append_data_to_csv(data, os.path.join(self.exp_dir, 'Diabetes_LNL_I_trials.csv'))
        elif self.option.bias_type == "II":
            data = {
                'Time': [datetime.datetime.now()],
                'Bias': [self.opt['bias_attr']],
                'Train': [self.opt['Diabetes_train_mode']],
                'Test': [self.opt['Diabetes_test_mode']],
                'Test Acc': [final_acc * 100]
                }
            utils.append_data_to_csv(data, os.path.join(self.exp_dir, 'Diabetes_LNL_II_trials.csv'))

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)
