import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch.nn.functional as F
import random
import torchvision

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, is_logits=True):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)
        self.is_logits = is_logits
        if not is_logits:
            self.bn = nn.BatchNorm1d(out_dim)
            self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.out(x)
        if not self.is_logits:
            x = self.bn(self.relu(x))
        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class ResNet18(nn.Module):    
    def __init__(self, n_classes, pretrained, hidden_size=128, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(512, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # print(x.size())
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))
        return features

class sexClassifier(nn.Module):
    def __init__(self, numclass=8):
        super(sexClassifier, self).__init__()
        self.fc1r = nn.Linear(128,64)
        self.relur = nn.LeakyReLU(inplace=True)
        self.fc2r = nn.Linear(64, numclass)

        # self.fc1g = nn.Linear(128,64)
        # self.relug = nn.LeakyReLU(inplace=True)
        # self.fc2g = nn.Linear(64, numclass)

        # self.fc1b = nn.Linear(128,64)
        # self.relub = nn.LeakyReLU(inplace=True)
        # self.fc2b = nn.Linear(64, numclass)

        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x):

        xr = self.fc1r(x)
        xr = self.relur(xr)
        xr = self.fc2r(xr)

        # xg = self.fc1g(x)
        # xg = self.relug(xg)
        # xg = self.fc2g(xg)

        # xb = self.fc1b(x)
        # xb = self.relub(xb)
        # xb = self.fc2b(xb)
        return xr

class classifier(nn.Module):
    def __init__(self, numclass=10):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(128,64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(64, numclass)
        for m in self.children():
            weights_init_kaiming(m)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class convnet(nn.Module):
    def __init__(self,num_classes=10):
        super(convnet,self).__init__()
        self.bn0     = nn.BatchNorm2d(3)
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        feat_out = x  
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7

        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0),-1)
        y_low = self.fc(feat_low)

        return feat_out, y_low


class Predictor(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm2d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)

        return x,px
