"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import os
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

    predictions_label = np.argmax(predictions, axis = 1)
    
    accuracy = np.mean(predictions_label == targets)

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
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
    ### Private functions
    ##########################################################################
    def _data_transform(X, y):

        # Collapse last dimensions into vector        
        n_features = np.prod(X.shape[1:])
        X = X.reshape(X.shape[0], n_features)
        
        # One-hot -> labels
        t = np.argmax(y, axis = 1)
        
        return X, y, t
    
    def _eval(X, y):
        
        X, y, t = _data_transform(X, y)
        
        output = model.forward(X)
        loss = loss_function.forward(output, y)
        acc = accuracy(output, t)
        
        return loss, acc
    
    def _sgd_step(model, lr):
        """
        Performs a single step of vanilla SGD.
    
        Parameters
        ----------
        model : MLP object
        lr : learning rate
    
        Returns
        -------
        None.
    
        """
        for module in model.modules:
            if hasattr(module, 'grads'):
                for key in module.params.keys():
                    module.params[key] += -lr * module.grads[key]
                    
    def _model_save(save_loc, model):
        model_params = []
        
        for module in model.modules:
            if hasattr(module, 'grads'):
                for key in module.params.keys():
                    model_params.append((key, module.params[key]))
        
        with open(save_loc, 'wb') as file:
            np.save(file, np.array(model_params, dtype=object))
        file.close()
    
    ##########################################################################
    ### Training Code
    ##########################################################################
    
    # Data import. First batch imported for defining first layer size
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    X, y = cifar10['train'].next_batch(200)
    X, y, t = _data_transform(X, y)
    
    # Model definition
    model = MLP(X.shape[1], dnn_hidden_units, 10)
    
    # Loss function definition
    loss_function = CrossEntropyModule()
    
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
        if batch != 0:        
            # Load a batch of data
            X, y = cifar10['train'].next_batch(FLAGS.batch_size)
            X, y, t = _data_transform(X, y)
        
        # Compute gradients and perform backprop/SGD step
        output = model.forward(X)
        
        train_loss_batch = loss_function.forward(output, y)
        train_loss.append([batch, train_loss_batch])
                
        # Backprop
        model.backward(loss_function.backward(output, y))
        _sgd_step(model, FLAGS.learning_rate)
        
        # Evaluate on the whole test set every eval_freq and at end of training
        if batch % FLAGS.eval_freq == 0 or batch == FLAGS.max_steps-1:        
            batches = int(np.floor(cifar10['test'].num_examples / 
                                   FLAGS.batch_size))
            
            eval_loss, eval_ACC = [], []
            for i in range(batches):
                X, y = cifar10['test'].next_batch(FLAGS.batch_size)

                batch_loss, batch_ACC = _eval(X, y)
                
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
                _model_save(os.path.join(FLAGS.model_dir + '/MLP_Numpy'), model)
        
    with open(os.path.join(FLAGS.model_dir + '/MLP_Numpy_losses'), 'wb') as file:
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
