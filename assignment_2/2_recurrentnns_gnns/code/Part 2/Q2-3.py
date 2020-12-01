# This preps the Python documentation into a single .txt. file
#import os

#os.chdir('C:/Users/ivoon/Downloads/python-3.9.0-docs-text - 1/python-3.9.0-docs-text/tutorial')

#filenames = os.listdir('.')

#with open('C:/Users/ivoon/Documents/GitHub/uvadlc_practicals_2020/assignment_2/2_recurrentnns_gnns/code/release/Part 2/assets/PyDocTutorials.txt', 'w') as outfile:
#    for fname in filenames:
#        with open(fname) as infile:
#            for line in infile:
#                outfile.write(line)
                
#os.chdir('C:/Users/ivoon/Documents/GitHub/uvadlc_practicals_2020/assignment_2/2_recurrentnns_gnns/code/release/Part 2/')

##############################################################################
# Sampling sutff for part 2
##############################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from model import TextGenerationModel
from dataset import TextDataset

### Playing with the Lisa trained models, first Bartleby

loaded = torch.load('./models/BartlebytheScrivener_LSTM')

config = loaded['config']

dataset = TextDataset(config.txt_file, 
                      seq_length = config.seq_length)  
model = TextGenerationModel(config.batch_size, 
                            config.seq_length,
                            dataset._vocab_size,
                            lstm_num_hidden=config.lstm_num_hidden, 
                            lstm_num_layers=config.lstm_num_layers,
                            dropout=1-config.dropout_keep_prob)
model.load_state_dict(loaded['model_state_dict'])

model.sample(2, string_conversion=dataset.convert_to_string, tau=2)

print(model.complete('"I would prefer not to."', dataset = dataset, length = 100))
print(model.complete('for i in range', dataset = dataset, length = 100))
print(model.complete('Do Re Mi', dataset = dataset, length = 100))

### Playing with the Lisa trained models, now PyDoc

loaded = torch.load('./models/PyDocTutorials_LSTM')

config = loaded['config']

dataset = TextDataset(config.txt_file, 
                      seq_length = config.seq_length)  
model = TextGenerationModel(config.batch_size, 
                            config.seq_length,
                            dataset._vocab_size,
                            lstm_num_hidden=config.lstm_num_hidden, 
                            lstm_num_layers=config.lstm_num_layers,
                            dropout=1-config.dropout_keep_prob)
model.load_state_dict(loaded['model_state_dict'])

model.sample(2, string_conversion=dataset.convert_to_string, tau=0.5)

print(model.complete('"I would prefer not to."', dataset = dataset, length = 100))
print(model.complete('for i in range', dataset = dataset, length = 100))
print(model.complete('Do Re Mi', dataset = dataset, length = 100))


