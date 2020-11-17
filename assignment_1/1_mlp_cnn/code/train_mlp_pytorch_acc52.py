"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import mlp_pytorch
import cifar10_utils

import time

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2500'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 5000
BATCH_SIZE_DEFAULT = 256
EVAL_FREQ_DEFAULT = 100

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
    
    accuracy = torch.mean((predictions_label == targets).double()).item()
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
    
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    ##########################################################################
    ### Private functions
    ##########################################################################
    def _data_transform(X, y, device):
        
        n_features = np.prod(X.shape[-3:])
        X = X.reshape(X.shape[0], n_features)
        
        # Push data to device
        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)
        
        # One-hot -> labels
        t = torch.nonzero(y)[:,1]
        
        return X, y, t
    
    def _eval(X, y, device):
        
        X, y, t = _data_transform(X, y, device)
        
        output = mlp(X)
        loss = loss_function(output, t)
        #print(output.shape, t.shape)
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
    mlp = mlp_pytorch.MLP(32 * 32 * 3, dnn_hidden_units, 10)  
    
    for module in mlp.layers:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif isinstance(module, nn.ELU):
            module = nn.ReLU(True)
    
    mlp = mlp.to(device)
        
    # Data sets
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    # Loss function and optimizer definition
    loss_function = nn.CrossEntropyLoss()
    
    #optimizer = torch.optim.SGD(mlp.parameters(), lr = FLAGS.learning_rate)
    #optimizer = torch.optim.Adam(mlp.parameters(), lr = FLAGS.learning_rate)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr = FLAGS.learning_rate) 
    
    milestones = [int(0.5*FLAGS.max_steps), int(0.75*FLAGS.max_steps)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones= milestones, 
                                                     gamma=0.1)

    # List for loss curve
    train_loss = []
    test_loss = []
    test_ACC = []
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
        mlp.train()
        optimizer.zero_grad()
        
        # Load a batch of data
        X, y = cifar10['train'].next_batch(FLAGS.batch_size)
        X, y, t = _data_transform(X, y, device)
        
        # Compute gradients and perform backprop/SGD step
        train_loss_batch = loss_function(mlp(X), t)
        train_loss.append([batch, train_loss_batch.item()])
                
        # Backprop
        train_loss_batch.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluate on the whole test set every eval_freq and at end of training
        if batch % FLAGS.eval_freq == 0 or batch == FLAGS.max_steps-1:
            mlp.eval()
            X_test, y_test = cifar10['test'].images, cifar10['test'].labels
            
            eval_loss, eval_ACC = _eval(X_test, y_test, device)
            
            test_loss.append([batch, eval_loss.item()])
            test_ACC.append([batch, eval_ACC])
                        
            # Pretty progress print for eval
            if batch == 0: past_loss = -np.inf
            print('{:11}|{:>11.2f}|{:>10.2f}%|{:>11.2e}|{:>+11.2e}'.format(
                batch, (time.time()-t0)/60, eval_ACC * 100,
                eval_loss, eval_loss - past_loss ), end='')
            past_loss = eval_loss
            
            if eval_ACC > best_test_ACC:
                print('    | Max Acc |')
                torch.save(mlp.state_dict(), 
                           os.path.join(FLAGS.model_dir + '/MLP_pytorch_acc52'))
                best_test_ACC = eval_ACC
            else: print()
        
    with open(os.path.join(FLAGS.model_dir + '/MLP_pytorch_losses_acc52'), 'wb') as file:
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
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR_DEFAULT,
                        help='Directory for storing trained models')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
