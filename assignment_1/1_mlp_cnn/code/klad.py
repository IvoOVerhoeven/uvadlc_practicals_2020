import matplotlib.pyplot as plt
import numpy as np

losses = np.load('./cifar10/modelsMLP_pytorch_losses')

train_losses = losses['arr_0']
test_losses = losses['arr_1']
test_ACC = losses['arr_2']

fig, ax1 = plt.subplots(figsize=(5.50107, 9.00177/3))

ax1.set_xlabel('Batch')
ax1.set_ylabel('CE-Loss')
ax1.plot(train_losses[:,0], train_losses[:,1], label = 'Train Loss', c= u'#1f77b4')
ax1.plot(test_losses[:,0], test_losses[:,1], label = 'Test Loss', c= u'#ff7f0e')

ax2 = ax1.twinx()  
ax2.set_ylabel('Accuracy')  
ax2.plot(test_ACC[:,0], test_ACC[:,1], ls = '--', label = 'Test Accuracy', c= u'#ff7f0e')

fig.suptitle('Pytorch Default MLP')
fig.legend(bbox_to_anchor=(1, 0.9), loc='upper left')

fig.savefig('../figures/PytorchDefaultMLP.pdf', bbox_inches='tight')