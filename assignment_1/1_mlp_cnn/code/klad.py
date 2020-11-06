import torch
from torch import nn

import torchvision
from torchvision import models

densenet = models.densenet121(pretrained=True, memory_efficient=True)
densenet.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)

for module in densenet.modules():
    module.requires_grad = False
    
densenet.classifier.requires_grad = True

X, y = cifar10['train'].next_batch(FLAGS.batch_size)
X = torch.tensor(X)

densenet.forward(X)
