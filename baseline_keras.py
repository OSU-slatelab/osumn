from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam

from data_io import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
args = parser.parse_args()


train_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp",
    out_frame_file = "data-fbank/train_si84_clean_global_normalized/feats.scp",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = True,
)

test_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-fbank/dev_dt_05_clean_global_normalized/feats.scp.mod",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = False,
)

model = Sequential([
    Flatten(input_shape=(11,771)),
    Dropout(0.5),
    Dense(2048),
    Activation('relu'),
    Dropout(0.5),
    Dense(2048),
    Activation('relu'),
    Dropout(0.5),
    Dense(40),
])

adam = Adam(lr=0.0001)
model.compile(optimizer=adam,loss='mse')


for epoch in range(100):
    train_loss = model.fit_generator(train_loader.batchify(), 5309, workers=3, use_multiprocessing=True)
    test_loss = model.evaluate_generator(test_loader.batchify(), 1612, workers=3, use_multiprocessing=True)

    print("Epoch", epoch)
    print("Test loss:", test_loss)
