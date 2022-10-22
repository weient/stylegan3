import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


from .resnet import *


def loss_per(img, real):
    sum_loss = 0
    for i in range(0,5):
        sub = torch.sub(real[i], img[i])
        num = torch.numel(sub)
        out = torch.sum(sub.abs())/num
        sum_loss += out
    return sum_loss/4

    

#def loss_text(img, real):

#def loss_emb(img, real):

def call_type(in_img, in_real):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet34()
    net = net.to(device)

    checkpoint = torch.load('/content/drive/Shareddrives/styleGAN3/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    train_epoch = checkpoint['epoch']
    print("model acc: ", best_acc)
    print("model #iter: ", train_epoch)

    in_img.to(device)
    in_real.to(device)
    outputs, maps = net(in_img)
    outputs, maps_real = net(in_real)
    per = loss_per(maps, maps_real)
    return per
    

