import torch
import torch.nn as nn

from datasets import RandomCombinationsDataset
from lstm import LSTM
from bi_lstm import biLSTM
from torch.utils.data import DataLoader

seq_length = 4
input_dim = 1
hidden_dim = 128
num_classes = 10
batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda': print(torch.cuda.get_device_name(0))

dataset = RandomCombinationsDataset(seq_length+1)
dataloader = iter(DataLoader(dataset, 128, drop_last=True))

#model = LSTM(seq_length, input_dim, hidden_dim, num_classes, batch_size, device)
model = biLSTM(seq_length, input_dim, hidden_dim, num_classes, batch_size, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for batch in range(2000):
    # Remove stored gradients
    model.train()
    optimizer.zero_grad()
    
    # Load a batch of data
    X, y = next(dataloader)
    X, y = X.to(device), y.to(device)
    
    # Forward
    output = model(X)
    
    # Compute gradients and perform backprop/SGD step
    loss = criterion(output, y.long())
    print(batch, loss.item())
            
    # Backprop
    loss.backward()
    optimizer.step()

X, y = next(dataloader)
X, y = X.to(device), y.to(device)
output = model(X)

torch.mean((torch.argmax(output, 1) == y).float())

#W = torch.empty((input_dim, hidden_dim), dtype=torch.float, device=device)
#X[:, 4] @ W
#X[:, 4]
#for t in range(seq_length):
#    t_r = seq_length-t-1
#    print(t, seq_length-t-1, X[:, t_r:t_r+1].size())