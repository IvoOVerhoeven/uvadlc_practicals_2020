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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################

def hidden_init(config):
        h = torch.zeros(size=(config.lstm_num_layers, 
                              config.batch_size, 
                              config.lstm_num_hidden)
                        ).to(config.device)
                
        c = torch.zeros_like(h).to(config.device)
        hidden = (h, c)
            
        return hidden

def train(config):
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print('Using device:', device)
    if device.type == 'cuda': print(torch.cuda.get_device_name(0))

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, 
                          seq_length = config.seq_length)  
    data_loader = DataLoader(dataset, config.batch_size, drop_last=True)
    data_loader = iter(data_loader)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, 
                                config.seq_length,
                                dataset._vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden, 
                                lstm_num_layers=config.lstm_num_layers,
                                dropout=1-config.dropout_keep_prob)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), config.learning_rate)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config.learning_rate_step,
                                          gamma=config.learning_rate_decay)
    
    best_acc = 0
    losses = []
    accurs = []
    
    for step in range(1,int(config.train_steps)+1):
        
        # Only for time measurement of step through network
        t1 = time.time()
        
        model.train()
        
        # If data runs out, reinitiate the dataloader
        try:
            data = next(data_loader)
        except StopIteration:
            dataloader = iter(DataLoader(dataset, config.batch_size, 
                                         drop_last=True))
            data = next(dataloader)
        
        # Process the data
        (X, y) = data
        X, y = torch.stack(X), torch.stack(y)
        X, y = X.to(config.device), y.to(config.device)
    
        acc  = 0
        loss = 0
        
        # This train method is slow, but it ensures teacher forcing as opposed
        # to implicit via feeding all data at once
        for t in range(config.seq_length):  
            if t == 0:
                hidden = model.hidden_init()
            
            out, hidden = model(X[t:t+1], hidden)
            loss += criterion(out[0], y[t])
            
            predictions = torch.argmax(out, dim=2)
            acc += torch.sum((predictions == y[t])).item() / config.batch_size
        
        # Calculate the gradients, etc.
        loss.backward()
    
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        
        batch_loss, batch_acc = loss.item()/config.seq_length, \
            acc/config.seq_length
        
        losses.append(batch_loss)
        accurs.append(batch_acc)
        
        optimizer.step() 
        scheduler.step()
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1+1e-8)

        if step == 1 or step % config.print_every == 0 or step == config.train_steps:

            print("[{}] Train Step {:04d}/{:04d}, Learning rate = {:.2e}, Examples/Sec = {:>6.2f}, "
                  "Accuracy = {:05.2f}%, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), scheduler.get_last_lr()[0], 
                    examples_per_second, batch_acc*100, batch_loss
                    ))
        
        if  batch_acc > best_acc:
            path  = config.save_path + '/'
            path += config.txt_file.rsplit('/')[-1][:-4]
            path += '_LSTM'
            
            torch.save({
            'batch': step+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config':config
            }, path)
            best_acc = batch_acc
            
        if step % config.sample_every == 0:
            model.eval()
            # Generate some sentences by sampling from the model
            for length in [10,30,100]:
                text = model.sample(N_samples=100, 
                                    string_conversion=dataset.convert_to_string,
                                    length=length)
                path  = config.save_path + '/'
                path += config.txt_file.rsplit('/')[-1][:-4]
                path += '_samples_'+str(step)+'_T-'+str(length)+'.txt'
                print(path)
                np.savetxt(path, text, delimiter="", fmt='|%s|', newline='\n') 
            for tau in [0.5,1,2]:
                text = model.sample(N_samples=100, 
                                    string_conversion=dataset.convert_to_string,
                                    tau=tau)
                path  = config.save_path + '/'
                path += config.txt_file.rsplit('/')[-1][:-4]
                path += '_samples_'+str(step)+'_tau-'+str(tau)+'.txt'
                print(path)
                np.savetxt(path, text, delimiter="", fmt='|%s|', newline='\n') 


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            path  = config.save_path + '/'
            path += config.txt_file.rsplit('/')[-1][:-4]
            path += '_Train_Losses'
            
            np.savez(path, loss=losses, accuracy=accurs)
        
    print(step)
    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, 
                        default='./assets/BartlebytheScrivener.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=1024,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.955,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=75,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=7.5e3,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=75,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=2500,
                        help='How often to sample from the model')
    

    # If needed/wanted, feel free to add more arguments
    # This is in the train loop but not in the config file??
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on')
    parser.add_argument('--save_path', type=str, default='./models/',
                        help='Where to save the model to')
    parser.add_argument('--seed', type=int, default=610,
                        help='Random seed.')

    config = parser.parse_args()

    # Train the model
    train(config)           