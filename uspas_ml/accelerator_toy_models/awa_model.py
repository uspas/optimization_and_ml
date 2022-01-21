# add toy accelerator package location to PATH
import os

# add PyTorch and TorchVision (used for cropping etc.)
import torch

import torch.nn as nn

from . import data_dir

#AWAMODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/AWA_NN.pth')

AWAMODEL = os.path.join(data_dir, 'AWA_NN.pth')


class AWANN(nn.Module):
    def __init__(self):
        super(AWANN, self).__init__()

        hidden_size = 20
        self.linear1 = nn.Linear(6, hidden_size)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.tanh4 = nn.Tanh()
        self.linear5 = nn.Linear(hidden_size, 7)
        self.tanh5 = nn.Tanh()

    def forward(self, x):
        # propagate through model
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.linear3(x)
        x = self.tanh3(x)
        x = self.linear4(x)
        x = self.tanh4(x)
        x = self.linear5(x)
        x = self.tanh5(x)

        return x


class AWAModel:
    def __init__(self):
        self.model = AWANN()
        self.model.load_state_dict(torch.load(AWAMODEL))        
        #path = os.path.realpath(os.path.relpath('../accelerator_toy_models/AWA_NN.pth'))
        
        self.features = ['P0', 'P1', 'G0', 'G1', 'K1', 'K2']
        self.targets = ['rms_x', 'rms_y', 'rms_s', 'emit_x', 'emit_y', 'emit_s', 'dE']

    def predict(self, x):
        return self.model(x)


if __name__ == '__main__':
    AWAModel()
