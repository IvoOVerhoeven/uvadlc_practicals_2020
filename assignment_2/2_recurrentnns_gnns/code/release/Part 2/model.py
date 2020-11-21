# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


import numpy as np

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, 
                 lstm_num_hidden=256, lstm_num_layers=2, 
                 device='cuda:0', dropout=0):

        super(TextGenerationModel, self).__init__()
        
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_dim = vocabulary_size

        self.Embedding = nn.Embedding(num_embeddings=vocabulary_size, 
                                      embedding_dim=self.embedding_dim)
        for layer in self.Embedding.parameters(): layer.requires_grad = False        
        
        self.LSTM = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            dropout=dropout)
        
        self.linear = nn.Linear(lstm_num_hidden, 
                                vocabulary_size)
        
        self.to(self.device)
        
    def forward(self, x, hidden=None):

        x_embed = self.Embedding(x)        
        
        LSTM_out, (h, c) = self.LSTM(x_embed, hidden)
        
        Classfier_out = self.linear(LSTM_out)
                
        return Classfier_out, (h,c)
    
    def sample(self, N_samples, string_conversion=None, tau=None):
        """ 
        Args:
            - N_samples: int, number of strings to return
            - string_conversion: callable, method for converting idx to string.
              Applied to entire row
            - tau: float, if None, uses greedy sampling. Else, choice is made
              randomly using temperature scaled softmax
        """
    
        SoftMax = nn.LogSoftmax(dim=2)

        text_sample = []

        start_character = torch.randint(0, self.vocabulary_size, 
                                        size=(1, N_samples))
        text_sample.append(start_character)
        
        hidden = None
        for t in range(0, self.seq_length):
            
            input = text_sample[t].to(self.device)
            out, hidden = self.forward(input, hidden)
            
            if tau is None:
                out_p = SoftMax(out)
                text_sample.append(torch.argmax(out_p, dim=2).to('cpu'))
            else:
                # Take the argmax or model a distribution with temp. scaling?
                #out_p = SoftMax(tau * out)
                distribution = Categorical(probs = out_p)
                text_sample.append(torch.argmax(out_p, dim=2).to('cpu'))
        
        
        text_sample = torch.squeeze(torch.stack(text_sample).permute(2,0,1))
        text_sample = text_sample.numpy()
        
        if  string_conversion == None:
            return text_sample
        else:
            return np.apply_along_axis(string_conversion, 
                                       axis=1, arr=text_sample)
