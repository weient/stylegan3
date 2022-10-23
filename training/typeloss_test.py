import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


from .resnet import *

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G

def loss_per(img, real):
    sum_loss = 0
    for i in range(0,5):
        sub = torch.sub(real[i], img[i])
        num = torch.numel(sub)
        out = torch.sum(sub.abs())/num
        sum_loss += (out/4) #除以batch_size
    return sum_loss/5 #除以feature map數量

def loss_text(img, real):
    sum_loss = 0
    for i in range(0,5):
        sub = gram_matrix(real[i]) - gram_matrix(img[i])
        out = torch.sum(sub.abs())
        sum_loss += (out/4)
    return sum_loss/5

def loss_emb(img, real):
    sub = real[5]-img[5]
    out = torch.sum(sub.abs())
    return out/4

def call_type(in_img, in_real):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet34()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    checkpoint = torch.load('/content/drive/Shareddrives/styleGAN3/ckpt.pth')
    net.load_state_dict(checkpoint['net'], False)
    best_acc = checkpoint['acc']
    train_epoch = checkpoint['epoch']
    #print("model acc: ", best_acc)
    #print("model #iter: ", train_epoch)

    in_img.to(device)
    in_real.to(device)
    outputs, maps = net(in_img)
    outputs, maps_real = net(in_real)
    per = loss_per(maps, maps_real)
    tex = loss_text(maps, maps_real)
    emb = loss_emb(maps, maps_real)
    #print("per: ", per)
    #print("tex: ", tex)
    #print("emb: ", emb)
    return per + 500*tex + emb
    

