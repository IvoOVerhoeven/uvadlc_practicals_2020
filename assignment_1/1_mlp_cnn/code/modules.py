"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
        """
        
        self.in_features = in_features
        self.out_features = out_features
        
        w = np.random.normal(loc = 0, scale = 0.0001, 
                             size = (self.out_features, self.in_features))
        b = np.zeros((self.out_features, 1))
        
        self.params = dict()
        self.params['weight'] = w
        self.params['bias'] = b
        
        self.grads = dict()
        self.grads['weight'] = np.zeros_like(w)
        self.grads['bias'] = np.zeros_like(b)
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        
        self.x = x
        
        B = np.repeat(self.params['bias'].T, x.shape[0], axis = 0)
        
        Y = x @ self.params['weight'].T + B
        
        return Y
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        self.grads['weight'] = dout.T @ self.x
        self.grads['bias'] = dout.T @ np.ones((self.x.shape[0], 1))
        
        dx = dout @ self.params['weight']
        
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        
        def _exp_normalize(x):
            # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            b = x.max()
            y = np.exp(x - b)
            return y / y.sum()

        # Assuming the first dimension is the batch size, applies softmax 
        # over the columns
        self.softmax_vals = np.apply_along_axis(_exp_normalize, 1, x)

        return self.softmax_vals
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        
        dx = np.zeros_like(dout)
        for sample, softmax in enumerate(self.softmax_vals):
            dydx = np.diag(softmax) - np.outer(softmax, softmax)
            dx[sample] = dydx @ dout[sample]
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """
                
        self.loss = np.mean(-np.sum(y * np.log(x), 1))
        
        return self.loss
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        """
        
        dx = - (1/x.shape[0]) * (y/x)
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
    
        self.out = x.copy()
        self.out[self.out <= 0] = np.exp(self.out[self.out <= 0])-1
        
        return self.out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        
        dydx = np.where(self.out > 0, 1, self.out+1)
        dx = np.multiply(dydx, dout)
        
        return dx
