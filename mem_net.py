import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from constants import FEATURE_LENGTH

# the memory net
class MemNet(nn.Module):

    def __init__(self, num_layers=1, inner_emb_size = 50):
        super(MemNet, self).__init__()
        self.inner_emb_size = inner_emb_size
        self.in_mem_encoder = nn.ModuleList([])
        self.out_mem_encoder = nn.ModuleList([])
        self.central_emb = nn.Linear(FEATURE_LENGTH, self.inner_emb_size, bias=False)
        self.top_layer = nn.Linear(inner_emb_size, 40)
        self.num_layers = num_layers
        self.batchnorm = nn.BatchNorm1d(FEATURE_LENGTH)
        self._init_mem_encoders()

    def _init_mem_encoders(self): # adjacent weight tying
        for i in range(self.num_layers):
            encoder = MemEncoder(self.inner_emb_size)
            if i >= 0 and i < self.num_layers - 1:
                self.in_mem_encoder.append(encoder)
            if i > 0 and i <= self.num_layers - 1:
                self.out_mem_encoder.append(encoder)

    def forward(self, x, C):
        x = self.batchnorm(x)
        C = self.batchnorm(C)
        encoded_input = self.central_emb(x) # (N, 1, dims)
        for index, enc in enumerate(self.in_mem_encoder):
            in_mem = self.in_mem_encoder[index](C) # (N, context_frames, dims)
            out_mem = self.out_mem_encoder[index](C) # (N, context_frames, dims)
            weights = torch.bmm(in_mem, encoded_input.transpose(1, 2)) # (N, frames, 1)
            o = torch.bmm(weights.transpose(1,2), out_mem) #(N, 1, dims)
            encoded_input = encoded_input + o # (N, 1, dims)
        encoded_input = F.relu(encoded_input)
        y_hat = self.top_layer(encoded_input) # (N, 1, fbanks)
        return y_hat

# simple encoder
class MemEncoder(nn.Module):

    def __init__(self, emb_size=50):
        super(MemEncoder, self).__init__()
        self.emb_matrix = nn.Linear(FEATURE_LENGTH, emb_size, bias=False)

    def forward(self, x):
        # x is (batchsize, frames, feats)
        return F.tanh(self.emb_matrix(x)) # (N, frames, dims)