"""Generating plots."""
import matplotlib.pyplot as plt
import numpy as np
import os

def loss_plots(loss_loc = '.', title = ''):

    def movingaverage(data, window_size):
        """ 
        Function for computing a smoothed, moving average, for the losses.
        Taken from https://stackoverflow.com/a/11352216/11692721
        """
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(data, window, 'valid')
    losses = np.load(loss_loc)
    
    train_losses = losses['arr_0']
    train_losses_smoothed = np.vstack([train_losses[:,0][4:-4], movingaverage(train_losses[:,1], 9)]).T
    
    test_losses = losses['arr_1']
    test_ACC = losses['arr_2']
    
    fig, ax1 = plt.subplots(figsize=(5.50107, 9.00177/3))
    
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('CE-Loss')
    ax1.plot(train_losses[:,0], train_losses[:,1], c= u'#1f77b4', alpha = 0.2)
    ax1.plot(train_losses_smoothed[:,0], train_losses_smoothed[:,1], label = 'Train Loss\n(Smoothed)', c= u'#1f77b4')
    ax1.plot(test_losses[:,0], test_losses[:,1], label = 'Test Loss', c= u'#ff7f0e')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(test_ACC[:,0], test_ACC[:,1], ls = '--', label = 'Test Accuracy', c= u'#ff7f0e')
    ax2.set_ylim([0,1])
    
    fig.suptitle(title)
    fig.legend(bbox_to_anchor=(1, 0.9), loc='upper left')
    
    fig.savefig(os.path.join('../figures/' + title + '.pdf'), 
                bbox_inches='tight')
    
    return fig

##############################################################################
### Pytorch MLP - Default parameters
##############################################################################

fig1 = loss_plots('./cifar10/models/MLP_Numpy_losses', 'Numpy MLP - Default')
fig1.show()

fig2 = loss_plots('./cifar10/models/MLP_pytorch_losses', 'Pytorch MLP - Default')
fig2.show()

fig3 = loss_plots('./cifar10/models/Convnet_pytorch_losses_v3', 'Pytorch Convnet')
fig3.show()

fig4 = loss_plots('./cifar10/models/Densenet121_pytorch_losses',
                  'Densenet121 Finetuned')
fig4.show()
