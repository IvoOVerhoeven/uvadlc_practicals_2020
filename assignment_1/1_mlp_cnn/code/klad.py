import numpy as np
from custom_layernorm import *
import torch
from torch import nn

gamma = nn.Parameter(torch.ones(100, dtype = torch.float32))
beta = nn.Parameter(torch.ones(100, dtype = torch.float32))
eps = 1e-8

X = 2 * torch.rand((20, 100)) + 10
dldy = torch.rand((20, 100))

mu     = torch.mean(X, dim = 1, keepdim=True)
sigma2 = torch.var(X, dim = 1, keepdim=True)
Xhat = (X - mu) / torch.sqrt(sigma2 + eps)

dldgamma = torch.sum((Xhat * dldy), dim = 1, keepdim=True)
dldbeta  = torch.sum(dldy, dim = 1, keepdim=True)

n_batch = 8
n_neurons = 128
# create random tensor with variance 2 and mean 3
x = 2 * torch.randn(n_batch, n_neurons, requires_grad=True) + 10

input = x.double()
gamma = torch.sqrt(10 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True))
beta = 100 * torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
bn_manual_fct = CustomLayerNormManualFunction(n_neurons)
y_manual_fct = bn_manual_fct.apply(input, gamma, beta)

torch.mean((y_manual_fct - beta)/gamma)

import torchvision
from torchvision import models
import cifar10_utils

model = models.densenet121(pretrained=True, memory_efficient=True)

model.classifier.reset_parameters()

model.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)

for module in model.modules():
    module.requires_grad = False

model.classifier.requires_grad = True
model = model.to(device)

cifar10 = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
X, y = cifar10['train'].next_batch(64)
X = torch.tensor(X)

X[0]

torch.mean(X, dim = (0,2,3))
