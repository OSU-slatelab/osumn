import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from constants import FEATURE_LENGTH
# 2 layer feedforward baseline
class FeedForward(nn.Module):

    def __init__(self):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(FEATURE_LENGTH * 11, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 40)
        self.batchnorm = nn.BatchNorm1d(FEATURE_LENGTH * 11)

    def forward(self, x, C):
        x = self.batchnorm(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class CrazyFeedForward(nn.Module):

    def __init__(self):
        super(CrazyFeedForward, self).__init__()
        self.layer1 = nn.Linear(FEATURE_LENGTH * 11, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 40)
        self.batchnorm1 = nn.BatchNorm1d(FEATURE_LENGTH * 11)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout()

    def forward(self, x, C):
        x = self.dropout(self.batchnorm1(x))
        x = F.relu(self.layer1(x))
        x = self.dropout(self.batchnorm2(x))
        x = F.relu(self.layer2(x))
        x = self.dropout(self.batchnorm2(x))
        x = self.layer3(x)
        return x

class NotSoCrazyFeedForward(nn.Module):

    def __init__(self):
        super(NotSoCrazyFeedForward, self).__init__()
        self.layer1 = nn.Linear(FEATURE_LENGTH * 11, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 40)
        self.batchnorm1 = nn.BatchNorm1d(FEATURE_LENGTH * 11)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout()

    def forward(self, x, C):
        x = self.dropout(self.batchnorm1(x))
        x = F.relu(self.layer1(x))
        x = self.batchnorm2(x)
        x = F.relu(self.layer2(x))
        x = self.batchnorm2(x)
        x = self.layer3(x)
        return x

class SingleFrameFeedForward(nn.Module):

    def __init__(self):
        super(SingleFrameFeedForward, self).__init__()
        self.layer1 = nn.Linear(FEATURE_LENGTH, 2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 40)

    def forward(self, x, C):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x