from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, PReLU, Input
from keras.optimizers import Adam

from data_io import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
args = parser.parse_args()


train_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-spectrogram/train_si84_clean_global_normalized/feats.scp.mod",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = True,
)

test_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-spectrogram/dev_dt_05_clean_global_normalized/feats.scp.mod",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = False,
)

inputs = Input(shape=(11,771))

fc1 = Flatten()(inputs)
fc1 = Dropout(0.3)(fc1)
fc1 = Dense(2048)(fc1)
fc1 = BatchNormalization(momentum=0.999)(fc1)
fc1 = PReLU()(fc1)
fc1 = Dropout(0.3)(fc1)

fc2 = Dense(2048)(fc1)
fc2 = BatchNormalization(momentum=0.999)(fc2)
fc2 = PReLU()(fc2)
fc2 = Dropout(0.3)(fc2)

out = Dense(257)(fc2)
model = Model(inputs=inputs, outputs=out)

adam = Adam(lr=0.0001, decay=1e-8)
model.compile(optimizer=adam,loss='mse')


for epoch in range(100):
    train_loss = model.fit_generator(train_loader.batchify(), 5299)
    test_loss = model.evaluate_generator(test_loader.batchify(), 1600)

    print("Epoch", epoch)
    print("Test loss:", test_loss)
