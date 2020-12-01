"""
This module implements a bidirectional LSTM in PyTorch.
You should fill in code into indicated sections.
Date: 2020-11-09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class biLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(biLSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.seq_length = seq_length-1
        self.embedding_dims = input_dim
        
        self.Embedding = nn.Embedding(num_embeddings=seq_length, 
                                      embedding_dim=self.embedding_dims)
        for layer in self.Embedding.parameters(): layer.requires_grad = False  
        
        self.LSTM_forward = LSTMCell(self.embedding_dims, hidden_dim, num_classes,
                                     batch_size, device)
        
        self.LSTM_backward = LSTMCell(self.embedding_dims, hidden_dim, num_classes,
                                      batch_size, device)
        
        self.params = nn.ParameterDict({
                # Parameters for output (h->p->y)
                'predict_h': nn.Parameter(torch.empty((2*hidden_dim, num_classes),
                                                      dtype=torch.float, 
                                                      device=device)),
                'predict_bias': nn.Parameter(torch.empty((num_classes), 
                                                         dtype=torch.float, 
                                                         device=device))
            })
        
        for _, weight_matrix in self.params.items():
            if len(weight_matrix.size()) > 1:
                # Weights get proper init call
                nn.init.kaiming_normal_(weight_matrix)
            else:
                # Bias gets filled with 0 
                nn.init.constant_(weight_matrix, 0)
        
        self.to(device)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        embedded_x = self.Embedding(x.long())
        
        for t in range(self.seq_length):
            t_r = (self.seq_length) - t - 1
            if t == 0:
                self.LSTM_forward(embedded_x[:,t], start = True)
                self.LSTM_backward(embedded_x[:,t_r], start = True)
            else:
                self.LSTM_forward(embedded_x[:,t])
                self.LSTM_backward(embedded_x[:,t_r])
                
        H = torch.cat((self.LSTM_forward.h, self.LSTM_backward.h), 1)
        
        predict  = H @ self.params['predict_h']
        predict += self.params['predict_bias']

        return predict
    
        ########################
        # END OF YOUR CODE    #
        #######################


class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTMCell, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
                
        self.params = nn.ParameterDict({
            #  Parameters for input modulation gate
            'modulation_x': nn.Parameter(torch.empty((input_dim, hidden_dim), dtype=torch.float, device=device)),
            'modulation_h': nn.Parameter(torch.empty((hidden_dim,hidden_dim), dtype=torch.float, device=device)),
            'modulation_bias': nn.Parameter(torch.empty((hidden_dim), dtype=torch.float, device=device)),
            
            # Parameters for input gate
            'input_x': nn.Parameter(torch.empty((input_dim, hidden_dim), dtype=torch.float, device=device)),
            'input_h': nn.Parameter(torch.empty((hidden_dim,hidden_dim), dtype=torch.float, device=device)),
            'input_bias': nn.Parameter(torch.empty((hidden_dim), dtype=torch.float, device=device)),
            
            # Parameters for forget gate
            'forget_x': nn.Parameter(torch.empty((input_dim, hidden_dim), dtype=torch.float, device=device)),
            'forget_h': nn.Parameter(torch.empty((hidden_dim,hidden_dim), dtype=torch.float, device=device)),
            'forget_bias': nn.Parameter(torch.empty((hidden_dim), dtype=torch.float, device=device)),
            
            # Parameters for output gate (new state)
            'output_x': nn.Parameter(torch.empty((input_dim, hidden_dim), dtype=torch.float, device=device)),
            'output_h': nn.Parameter(torch.empty((hidden_dim,hidden_dim), dtype=torch.float, device=device)),
            'output_bias': nn.Parameter(torch.empty((hidden_dim), dtype=torch.float, device=device)),
        })

        for _, weight_matrix in self.params.items():
            if len(weight_matrix.size()) > 1:
                # Weights get proper init call
                nn.init.kaiming_normal_(weight_matrix)
            else:
                # Bias gets filled with 0 
                nn.init.constant_(weight_matrix, 0)
        
        self.h = torch.zeros((batch_size, hidden_dim), dtype=torch.float, device = device)
        self.c = torch.zeros((batch_size, hidden_dim), dtype=torch.float, device = device)
        
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x, start = False):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        if start:
            # If start of the sequence, purge the existing h and c values
            self.h = torch.zeros_like(self.h)
            self.c = torch.zeros_like(self.h)
        
        g_input  = x @ self.params['modulation_x']
        g_input += self.h @ self.params['modulation_h'] 
        g_input += self.params['modulation_bias'][None,:]
        g_out = torch.tanh(g_input)
        
        i_input  = x @ self.params['input_x']
        i_input += self.h @ self.params['input_h'] 
        i_input += self.params['input_bias'][None,:]
        i_out = torch.sigmoid(i_input)
        
        f_input  = x @ self.params['forget_x']
        f_input += self.h @ self.params['forget_h'] 
        f_input += self.params['forget_bias'][None,:]
        f_out = torch.sigmoid(i_input)
        
        o_input  = x @ self.params['output_x']
        o_input += self.h @ self.params['output_h'] 
        o_input += self.params['output_bias'][None,:]
        o_out = torch.sigmoid(i_input)
        
        self.c = g_out * i_out + f_out * self.c
        self.h = torch.tanh(self.c) * o_out
        
        ########################
        # END OF YOUR CODE    #
        ########################
