# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


import torch.utils.data as data
import torchvision.datasets as datasets


class WholeDataLoader(Dataset):
    def __init__(self,option, istrain=True):
        # self.data_split = option.data_split
        self.is_train = istrain
        # data_dic = np.load(os.path.join(option.data_dir,'mnist_10color_jitter_var_%.03f.npy'%option.color_var),encoding='latin1').item()
        data_dic = np.load(os.path.join(option.data_dir,f'mnist_10color_jitter_var_{option.color_var}.npy'),encoding='latin1', allow_pickle=True).item()
        if self.is_train == 1:
            self.image = data_dic['train_image'].astype(np.uint8)
            self.label = data_dic['train_label'].astype(np.uint8)
        else:
            self.image = data_dic['test_image'].astype(np.uint8)
            self.label = data_dic['test_label'].astype(np.uint8)

        color_var = option.color_var
        self.color_std = color_var**0.5

        self.T = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914,0.4822,0.4465),
                                                   (0.2023,0.1994,0.2010)),
                                    ])

        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])


    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        image = self.ToPIL(image)
        label_image = image.resize((14,14), Image.NEAREST) 
        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()

        # image = self.ToPIL(image)
        # label_image = image.resize((14,14), Image.NEAREST) 
        # label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)).copy())
        # # mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        # label_image = torch.div(label_image,32)
        # label_image = label_image.long()
        # colorlabel = label_image.view(3,-1).max(1)[0].long()


        
        return self.T(image), label_image,  label.astype(np.long)

        

    def __len__(self):
        return self.image.shape[0]



