"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
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

        dims = [n_inputs] + n_hidden + [n_classes]

        self.modules = []
        for l, neurons in enumerate(dims):
            if l == 0:
                continue
    
            self.modules.append(LinearModule(dims[l-1], neurons))
            if l < len(dims) - 1:
                self.modules.append(ELUModule())
            else:
                self.modules.append(SoftMaxModule())

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        out = x.copy()
        for module in self.modules:
            out = module.forward(out)

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        """
        
        dy = dout.copy()
        
        for module in reversed(self.modules):
            dy = module.backward(dy)
                
        return
