"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import time

import torch

from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

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

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    
    ##########################################################################
    ### My code
    ##########################################################################
    
    # Find the device and let Pytorch know
    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"  
    device = torch.device(device)
    print('Training on {:s}'.format(device))
    
    # Code definition
    mlp = MLP(32 * 32 * 3, dnn_hidden_units, 10)        
    
    # Data sets
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    # Loss function and optimizer definition
    loss_function = CrossEntropyModule()
    optimizer = torch.optim.SGD(mlp.parameters(), lr = FLAGS.learning_rate)
    
    # List for loss curve
    loss_values = []
    
    # Pretty print for inspecting training progress
    print('{:^11s}|{:^11s}|{:^11s}|{:^14s}'.format('Batch','Minute','E[Loss]','dLoss' ))
    print('-'*50)
    
    for batch in range(FLAGS.max_steps):
        
        # Load a batch of data
        X, y = cifar10['train'].next_batch(FLAGS.batch_size)
        
        # Unravel the data
        X = X.reshape(FLAGS.batch_size, np.prod(X.shape[-3:]))
        
        # Push data to device
        X = torch.Tensor(X).to(device)
        y = torch.Tensor(y).to(device)
        
        # Remove stored gradients
        optimizer.zero_grad()
        
        # Compute gradients and perform backprop/SGD step
        loss = loss_function(mlp(X), y)
        
        if device == 'cpu': loss_values.append([batch, loss.detach().cpu().item()])
        else: loss_values.append([batch, loss.detach().item()])
        
        loss.backward()
        optimizer.step()
        
        # Evaluate on the whole test set
        if batch % FLAGS.eval_freq == 0:
            X_test = cifar10['test'].images
            y_test = cifar10['test'].labels
            
            # Unravel the data
            X = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[-3:]))
        
            # Push data to device
            X_test = torch.Tensor(X_test).to(device)
            y_test = torch.Tensor(y_test).to(device)
            
            if batch == FLAGS.eval_freq and epoch == 0: past_loss = -np.inf
            curr_loss = sum(loss_values[-batch:])/(verbose)
            print('{:11}|{:>11.2f}|{: 7.2e}|{:+11.2e}'.format(batch, (time.time()-t0)/60, curr_loss, curr_loss - past_loss ))
            past_loss = curr_loss

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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
    FLAGS, unparsed = parser.parse_known_args()

    main()
