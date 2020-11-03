"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        super(MLP, self).__init__()       
        
        dims = []
        dims.append(n_inputs)
        dims.extend(n_hidden)
        dims.append(n_classes)
        
        self.layers = nn.ModuleList()
        for l, neurons in enumerate(dims):
            if l == 0:
                continue
            
            self.layers.append(nn.Linear(dims[l-1], neurons))
            
            # Same init as the numpy implementation for similar eval scores
            nn.init.normal_(self.layers[-1].weight, 
                            mean = 0.0, std =  0.0001)
            nn.init.constant_(self.layers[-1].bias, 0)
            
            # Add activations. Last layer is special
            if l == len(dims)-1: 
                # DO NOT ADD SOFTMAX, loss function does this already...
                #self.layers.append(nn.Softmax(dim=0))
                continue
            else: 
                self.layers.append(nn.ELU())
           
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        return x
