from mem_net import MemNet
from data_io import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from feedforward import *
from constants import FEATURE_LENGTH, MODEL_CHOICE, CUDA, INNER_EMB_SIZE, NUM_EPOCHS

train_loader = DataLoader(
    base_dir        = "/fs/project/PAS1315/group1_chime2_data",
    in_frame_file   = "data-spectrogram/train_si84_delta_noisy/feats.scp",
    out_frame_file  = "data-fbank/train_si84_clean/feats.scp",
    batch_size      = 128,
    buffer_size     = 10,
    context         = 5,
    out_frame_count = 1,
    shuffle         = True)

dev_loader = DataLoader(
    base_dir        = "/fs/project/PAS1315/group1_chime2_data",
    in_frame_file   = "data-spectrogram/dev_dt_05_delta_noisy/feats.scp.mod",
    out_frame_file  = "data-fbank/dev_dt_05_clean/feats.scp.mod",
    batch_size      = 128,
    buffer_size     = 10,
    context         = 5,
    out_frame_count = 1,
    shuffle         = True)

if MODEL_CHOICE == 'mem':
    model = MemNet(inner_emb_size=INNER_EMB_SIZE)
elif MODEL_CHOICE == 'ff':
    model = FeedForward()
elif MODEL_CHOICE == 'cff':
    model = CrazyFeedForward()
elif MODEL_CHOICE == 'nscff':
    model = NotSoCrazyFeedForward()
elif MODEL_CHOICE == 'sfff':
    model = SingleFrameFeedForward()
loss = nn.MSELoss()
optimizer = Adam(model.parameters())
context_indices = list(range(5)) + list(range(6, 11))
central_index = 5

if CUDA:
    model.cuda()

# training
for i in range(NUM_EPOCHS):
    model.train()
    for in_frame_batch, out_frame_batch in train_loader.batchify():
        if MODEL_CHOICE == 'mem':

            C = Variable(torch.from_numpy(in_frame_batch[:, context_indices, :]))
            x = Variable(torch.from_numpy(in_frame_batch[:, central_index, :]).unsqueeze(1))
        elif MODEL_CHOICE == 'sfff':
            x = Variable(torch.from_numpy(in_frame_batch[:, central_index, :]).unsqueeze(1))
            C = Variable(torch.Tensor([0]))  # a dummy variable for FF
        elif 'ff' in MODEL_CHOICE:
            x = Variable(torch.from_numpy(in_frame_batch).view(-1, FEATURE_LENGTH * 11).contiguous())
            C = Variable(torch.Tensor([0])) # a dummy variable for FF
        y = Variable(torch.from_numpy(out_frame_batch))
        # print(C.size(), x.size(), y.size())
        if CUDA:
            C = C.cuda()
            x = x.cuda()
            y = y.cuda()
        y_hat = model(x, C)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
# dev
    model.eval()
    total_loss = 0
    for in_frame_batch, out_frame_batch in dev_loader.batchify():
        if MODEL_CHOICE == 'mem':

            C = Variable(torch.from_numpy(in_frame_batch[:, context_indices, :]))
            x = Variable(torch.from_numpy(in_frame_batch[:, central_index, :]).unsqueeze(1))
        elif MODEL_CHOICE == 'sfff':
            x = Variable(torch.from_numpy(in_frame_batch[:, central_index, :]).unsqueeze(1))
            C = Variable(torch.Tensor([0])) # a dummy variable for FF
        elif 'ff' in MODEL_CHOICE:
            x = Variable(torch.from_numpy(in_frame_batch).view(-1, FEATURE_LENGTH * 11))
            C = Variable(torch.Tensor([0])) # a dummy variable for FF
        x.volatile = True
        y = Variable(torch.from_numpy(out_frame_batch))
        if CUDA:
            C = C.cuda()
            x = x.cuda()
            y = y.cuda()
        y_hat = model(x, C)
        l = loss(y_hat, y)
        total_loss += l.data[0]
    print("Epoch {}; total loss {}".format(i, total_loss))