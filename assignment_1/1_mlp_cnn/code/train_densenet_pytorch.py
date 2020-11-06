"""
This module implements training and evaluation of Densent in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import time
import torch
import torch.nn as nn

import torchvision
from torchvision import models

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

# Directory in which the models are saved
MODEL_DIR_DEFAULT = './cifar10/models' 

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    """
    
    predictions_label = torch.argmax(predictions, dim = 1)
    
    accuracy = torch.mean((predictions_label == targets).double()).detach().item()
    
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.
      
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ##########################################################################
    ### Private functions
    ##########################################################################
    def _data_transform(X, y, device):
        # Push data to device
        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)
        
        # One-hot -> labels
        t = torch.nonzero(y)[:,1]
        
        return X, y, t
    
    def _eval(X, y, device):
        
        X, y, t = _data_transform(X, y, device)
        
        output = model(X)
        loss = loss_function(output, t).detach().item()
        acc = accuracy(output, t)
        
        return loss, acc
    
    ##########################################################################
    ### Training Code
    ##########################################################################
    
    # Find the device and let Pytorch know
    # From: https://stackoverflow.com/a/53374933/11692721
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda': print(torch.cuda.get_device_name(0))
    
    # Model definition
    model = models.densenet121(pretrained=True, memory_efficient=True)
    model.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)

    for module in model.modules():
        module.requires_grad = False
    
    model.classifier.requires_grad = True
    model = model.to(device)
    
    # Data sets
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    # Loss function and optimizer definition
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = FLAGS.learning_rate, 
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[2500, 3750], 
                                                     gamma=0.1)
    
    # List for loss curve
    train_loss = []
    test_loss  = []
    test_ACC   = []
    best_test_ACC = 0.0
    
    # Pretty print for inspecting training progress
    t0 = time.time()
    print('{:^62s}'.format('Performance on Test-set'))
    print('-'*62)
    print('{:^11s}|{:^11s}|{:^11s}|{:^11s}|{:^14s}'.format(
        'Batch','Minute','Acc','Loss','dLoss' ))
    print('-'*62)
    
    for batch in range(FLAGS.max_steps):
        # Remove stored gradients
        model.train()
        optimizer.zero_grad()
        
        # Load a batch of data
        X, y = cifar10['train'].next_batch(FLAGS.batch_size)
        X, y, t = _data_transform(X, y, device)
        
        # Compute gradients and perform backprop/SGD step
        train_loss_batch = loss_function(model(X), t)
        train_loss.append([batch, train_loss_batch.item()])
                
        # Backprop
        train_loss_batch.backward()
        optimizer.step()
        
        # Evaluate on the whole test set every eval_freq and at end of training
        if batch % FLAGS.eval_freq == 0 or batch == FLAGS.max_steps-1:
            model.eval()
            
            batches = int(np.floor(cifar10['test'].num_examples / FLAGS.batch_size))
            eval_loss, eval_ACC = [], []
            for i in range(batches):
                X, y = cifar10['test'].next_batch(FLAGS.batch_size)

                batch_loss, batch_ACC = _eval(X, y, device)
                
                eval_loss.append(batch_loss)
                eval_ACC.append(batch_ACC)

            test_loss.append([batch, np.mean(eval_loss)])
            test_ACC.append([batch, np.mean(eval_ACC)])
                        
            # Pretty progress print for eval
            if batch == 0: past_loss = -np.inf
            print('{:11}|{:>11.2f}|{:>10.2f}%|{:>11.2e}|{:>+11.2e}'.format(
                batch, (time.time()-t0)/60, test_ACC[-1][1] * 100,
                test_loss[-1][1], test_loss[-1][1] - past_loss ))
            past_loss = test_loss[-1][1]
            
            if test_ACC[-1][1] > best_test_ACC:
                torch.save(model.state_dict(), 
                           os.path.join(FLAGS.model_dir + '/Densenet121_pytorch'))
                
        scheduler.step()
        
    with open(os.path.join(FLAGS.model_dir + '/Densenet121_pytorch_losses'), 'wb') as file:
        np.savez(file, train_loss, test_loss, test_ACC)
        file.close()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR_DEFAULT,
                        help='Directory for storing trained models')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
