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
        b = np.zeros((1, self.out_features))
        
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
        
        B = np.repeat(self.params['bias'], x.shape[0], axis = 0)
        
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
        
        dx = dout @ self.grads['weight']
        self.grads['weight'] = dout.T @ self.x
        self.grads['bias'] = dout.T @ np.ones((self.x.shape[0], 1))
        
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
        
        self.dx = np.zeros_like(self.softmax_vals)
        for node in range(self.softmax_vals.shape[1]):
            # Multivariate version of the kronecker delta
            delta = np.zeros_like(self.softmax_vals)
            delta[:, node] = np.ones(self.softmax_vals.shape[0])
            
            # Intermediate result. Can be optimised if it is known that the
            # CE-loss module follows.
            dydx = self.softmax_vals * (delta-self.softmax_vals)
            self.dx[:, node] = np.sum( dydx * dout, 1)
        
        return self.dx


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
        
        # Given that we don' need to specify values for each of the input
        # we're free to be smart about how to retrieve the loss.
        
        #rel_vals = x[np.arange(0, y.shape[0]), y]
        rel_vals =  np.sum(np.multiply(y, x), 1)
        
        self.loss = -np.mean(np.log(rel_vals))
        
        return self.loss
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        rel_vals = np.sum(np.multiply(y, np.log(x)), 1)
        dx = - np.divide(np.divide(1, rel_vals), x.shape[0])
        
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
        
        self.activation = np.where(x >= 0, x, np.exp(x))
        
        return self.activation
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """
        
        dydx = np.where(self.x >= 0, np.ones_like(self.x), np.exp(self.x))
        dx = np.multiply(dydx, dout)
        
        return dx
