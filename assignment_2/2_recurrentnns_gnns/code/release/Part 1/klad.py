import numpy as np
import torch
import torch.nn as nn

from datasets import RandomCombinationsDataset
from lstm import LSTM
from bi_lstm import biLSTM
from torch.utils.data import DataLoader

### Variable Parameters
models = ['LSTM', 'biLSTM']
seq_lengths = [5, 10, 15]
seeds = [42, 520, 610]

var_pars = np.array(np.meshgrid(models, seq_lengths, seeds)).T.reshape(-1,3)

n_experiments = var_pars.shape[0]

### Constant Parameters
max_iters = 2000
verbosity = 200

input_dim = 1
hidden_dim = 128
batch_size = 128

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda': print(torch.cuda.get_device_name(0))

for var_par in var_pars:
    experiment_name = var_par[0]+'-'+var_par[1]+'-'+var_par[2]
    print(experiment_name)
    
    model = var_par[0]
    seq_length = int(var_par[1])
    seed = int(var_par[2])
    num_classes = seq_length+1
    
    ##########################################################################
    ### Experiment Loop ######################################################
    ##########################################################################
    torch.manual_seed(seed)

    dataset = RandomCombinationsDataset(seq_length+1)
    dataloader = iter(DataLoader(dataset, 128, drop_last=True))
    
    losses = []
    accurs = []
    
    if model == 'LSTM':
        ANN = LSTM(seq_length, input_dim, hidden_dim, num_classes, 
                   batch_size, device)
    elif model == 'biLSTM':
        ANN = biLSTM(seq_length, input_dim, hidden_dim, num_classes, 
                     batch_size, device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ANN.parameters(), lr = 1e-3)
    
    for batch in range(max_iters):
        # Remove stored gradients
        ANN.train()
        optimizer.zero_grad()
        
        # Load a batch of data
        X, y = next(dataloader)
        X, y = X.to(device), y.to(device)
        
        # Forward
        output = ANN(X)
        
        # Compute gradients and perform backprop/SGD step
        loss = criterion(output, y.long())
        acc = torch.mean((torch.argmax(output, 1) == y).float())
        
        losses.append(loss.item())
        accurs.append( acc.item())
        if batch == 0 or (batch % verbosity) == 0 or batch == max_iters-1:
            print('{:>5d}|Loss: {:5.2e}, Acc: {:>6.2f}'.format(batch, 
                                                               loss.item(), 
                                                               acc.item()*100))
                
        # Backprop
        loss.backward()
        optimizer.step()
    
    np.savez(str('./models/'+experiment_name), [losses, accurs])    