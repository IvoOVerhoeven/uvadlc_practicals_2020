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
from torch.distributions.categorical import Categorical


import numpy as np

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size, 
                 lstm_num_hidden=256, lstm_num_layers=2, 
                 device='cuda:0', dropout=0):

        super(TextGenerationModel, self).__init__()
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        
        self.embedding_dim = vocabulary_size

        self.Embedding = nn.Embedding(num_embeddings=vocabulary_size, 
                                      embedding_dim=self.embedding_dim)
        for layer in self.Embedding.parameters(): layer.requires_grad = False        
        
        self.LSTM = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            dropout=dropout)
        
        self.linear = nn.Linear(self.lstm_num_hidden, 
                                vocabulary_size)
        
        self.to(self.device)
        
    def hidden_init(self, N_samples=None):
        if N_samples is None:
            size = (self.lstm_num_layers, self.batch_size, self.lstm_num_hidden)
        else:
            size = (self.lstm_num_layers, N_samples, self.lstm_num_hidden)
        
        h = torch.zeros(size=size).to(self.device)
        c = torch.zeros(size=size).to(self.device)
        hidden = (h, c)
        
        return hidden
        
    def forward(self, x, h_in=None):

        x_embed = self.Embedding(x)        
        
        LSTM_out, h_out = self.LSTM(x_embed, h_in)
        
        Classfier_out = self.linear(LSTM_out)
                
        return Classfier_out, h_out
    
    def sample(self, N_samples, string_conversion=None, tau=None, length=None):
        """ 
        Samples some strings.

        Parameters
        ----------
        N_samples: int, number of strings to return
        string_conversion: callable, method for converting idx to string.
              Applied to entire row
        tau: float, if None, uses greedy sampling. Else, choice is made
              randomly using temperature scaled softmax
        length: int, the length of the samples, can be set to be longer than
            30. Will use self.seq_length as default

        Returns
        -------
        output_string : numpy array, samples, hopefully converted to chars.

        """
        
        self.eval()
        
        if length == None:
            length = self.seq_length
    
        SoftMax = nn.Softmax(dim=2)

        text_sample = []

        start_character = torch.randint(0, self.vocabulary_size, 
                                        size=(1, N_samples))
        text_sample.append(start_character)
        
        for t in range(0, length):
            
            input = text_sample[t].to(self.device)

            if t == 0:
                out, hidden = self.forward(input, self.hidden_init(N_samples))
            else:
                out, hidden = self.forward(input, hidden)
                        
            if tau is None:
                out_p = SoftMax(out)
                text_sample.append(torch.argmax(out_p, dim=2).to('cpu'))
            else:
                # Take the argmax or model a distribution with temp. scaling?
                out_p = SoftMax(tau * out)
                distribution = Categorical(probs = out_p)
                text_sample.append(distribution.sample().to('cpu'))
        
        
        text_sample = torch.squeeze(torch.stack(text_sample).permute(2,0,1))
        text_sample = text_sample.numpy()
        
        self.train()
        
        if  string_conversion == None:
            return text_sample
        else:
            return np.apply_along_axis(string_conversion, 
                                       axis=1, arr=text_sample)
        
        
    def complete(self, input_string, dataset, length = None):
        """
        Completes an input string up to 30 characters.

        Parameters
        ----------
        input_string : str
            The input string. Works best when ending with space.
        dataset : Dataset object
            Dataset object generated.

        Returns
        -------
        output_string : str
            Completed string.

        """
        
        self.eval()
        
        if length == None:
            length = self.seq_length
        
        SoftMax = nn.Softmax(dim=2)

        input_length = len(input_string)
        
        input_tensor = [dataset._char_to_ix[char] for char in input_string]
        input_tensor = torch.tensor(input_tensor)[:, None].to(self.device)
        
        out, hidden = self.forward(input_tensor, self.hidden_init(1))
        
        for t in range(length - input_length):
            out, hidden = self.forward(input_tensor[-1:], hidden)
            predict = torch.argmax(SoftMax(out), dim=2)
            input_tensor = torch.cat((input_tensor, predict), 0)
            
        input_tensor = input_tensor.view(-1).to('cpu').numpy()    
        
        output_string = ''
        for t in range(length):
            output_string += dataset._ix_to_char[input_tensor[t]]
            
        self.train()
            
        return output_string
