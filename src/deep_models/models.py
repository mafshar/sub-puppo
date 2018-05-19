import torch
import torch.nn as nn
import torch.nn.functional as F



class MfccNetWeak(nn.Module):
'''
    Input: vector of size 22 (MFCC features)
    Layer1 (fc1): fully-connected layer with 32 hidden units, ReLU activation
    Layer2 (fc2): fully-connected layer with 64 hidden units, ReLU activation
    Layer3 (dp): dropout layer with keep_prob = 0.8
    Output (fc3): genre classification (4 units i.e. logits layer)
'''

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(22, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x

class MfccNet(nn.Module):
'''
    Input: vector of size 22 (MFCC features)
    Layer1: fully-connected layer with 32 hidden units, ReLU activation
    Layer2: batch-normalization
    Layer3: fully-connected layer with 64 hidden units, ReLU activation
    Layer4: dropout layer with keep_prob = 0.8
    Output: genre classification (10 units i.e. logits layer)
'''

    def __init__(self):
        super(Net, self).__init__()


    def forward(self, x):
        pass


class ConvNetWeak(nn.Module):


    def __num_flat_features(self, x):
        pass


class ConvNet(nn.Module):


    def __num_flat_features(self, x):
        pass
