import torch
from torch import nn 

import cifar10_utils
import convnet_pytorch

cifar10 = cifar10_utils.get_cifar10()

X, y = cifar10['train'].next_batch(200)
X, y = torch.tensor(X), torch.tensor(y)
t = torch.nonzero(y)[:,1]

VGG_simple = convnet_pytorch.ConvNet(3, 10)
output = VGG_simple(X)
loss_function = nn.CrossEntropyLoss()

loss = loss_function(output, t)
loss.backward()