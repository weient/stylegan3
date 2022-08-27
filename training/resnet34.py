import torch
from torch import nn
from torch.nn import functional as F
from .basic_module import *
from torchvision.ops import roi_align

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class content_encoder(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self):
        super(content_encoder, self).__init__()
        self.model_name = 'resnet34'

        self.pre = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2))
        
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 128, 1)
        self.sub1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = self._make_layer( 128, 256, 2)
        self.sub2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = self._make_layer( 256, 512, 5)
        self.sub3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = self._make_layer( 512, 512, 3)
        self.sub4 = nn.Conv2d(512, 512, 3, 1, 1)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.sub1(x)
        x = self.layer2(x)
        x = self.sub2(x)
        x = self.layer3(x)
        x = self.sub3(x)
        x = self.layer4(x)
        x = self.sub4(x)
        #avg = nn.AvgPool2d(4)  
        #x = avg(x)
        return x

class style_encoder(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self):
        super(style_encoder, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2))
        
        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer( 64, 128, 1)
        self.sub1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = self._make_layer( 128, 256, 2)
        self.sub2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = self._make_layer( 256, 512, 5)
        self.sub3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = self._make_layer( 512, 512, 3)
        self.sub4 = nn.Conv2d(512, 512, 3, 1, 1)

        #分类用的全连接
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x, bounding_box):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.sub1(x)
        x = self.layer2(x)
        x = self.sub2(x)
        x = self.layer3(x)
        x = self.sub3(x)
        x = self.layer4(x)
        x = self.sub4(x)
        tmp = []
        tmp.append(bounding_box)
        bounding_box = tmp
        #print(bounding_box)
        device = torch.device('cuda')
        bounding_box = torch.Tensor(bounding_box).to(device)
        x = roi_align(x, [bounding_box], output_size=1, spatial_scale=0.0625, aligned=True)
        print("shape after roi: ", x.size())
        #avg = nn.AvgPool2d(16)  
        #x = avg(x)  # reduce dimension to [1, 512, 1, 1]
        x = x.view(x.size(0), -1) # flatten tensor to [1, 512]
        print("shape after view: ", x)
        return x
 