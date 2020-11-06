"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        super().__init__()
        
        self.block1 = nn.Sequential()
        self.block1.add_module('conv0', nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.block1.add_module('PreAct1', PreActResNetBlock(64))
        self.block1.add_module('conv1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0))
        self.block1.add_module('maxp1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block2 = nn.Sequential()
        self.block2.add_module('PreAct2a', PreActResNetBlock(128))
        self.block2.add_module('PreAct2b', PreActResNetBlock(128))
        self.block2.add_module('conv2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0))
        self.block2.add_module('maxp2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block3 = nn.Sequential()
        self.block3.add_module('PreAct3a', PreActResNetBlock(256))
        self.block3.add_module('PreAct3b', PreActResNetBlock(256))
        self.block3.add_module('conv3', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0))
        self.block3.add_module('maxp3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block4 = nn.Sequential()
        self.block4.add_module('PreAct4a', PreActResNetBlock(512))
        self.block4.add_module('PreAct4b', PreActResNetBlock(512))
        self.block4.add_module('maxp4', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.block5 = nn.Sequential()
        self.block5.add_module('PreAct5a', PreActResNetBlock(512))
        self.block5.add_module('PreAct5b', PreActResNetBlock(512))
        self.block5.add_module('maxp5', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.classifier = nn.Linear(in_features=512, out_features=n_classes)
        
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        out = out.view(-1, out.shape[1])
        out = self.classifier(out)
        
        return out
    
class PreActResNetBlock(nn.Module):

    def __init__(self, c):
        """
        Inputs:
            c - Number of input channels
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.net(x)
        out = x + z
        return out
