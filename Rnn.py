import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rnn_utils import ALL_LETTERS , N_LETTERS
from rnn_utils import load_data, letter_to_tensor , line_to_tensor , random_training_example

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size , hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size , output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
    def forward (self, input_tensor , hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    

