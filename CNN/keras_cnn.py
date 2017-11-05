from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.initializers import RandomNormal
from keras.optimizers import Adam

# Add parent directory to current path to import data loader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
args = parser.parse_args()


train_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-spectrogram/train_si84_clean_global_normalized/feats.scp.mod",
    batch_size = 256,
    buffer_size = 100,
    context = 5,
    out_frame_count = 1,
    shuffle = True,
)

test_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-spectrogram/dev_dt_05_clean_global_normalized/feats.scp.mod",
    batch_size = 256,
    buffer_size = 100,
    context = 5,
    out_frame_count = 1,
    shuffle = False,
)

model = Sequential([
    Reshape(target_shape=(11,257,1), input_shape=(11,257)),
    Conv2D(64, 7, padding='same', activation='relu'),
    MaxPooling2D((2,3), padding='same'),
    Conv2D(96, 3, padding='same', activation='relu'),
    MaxPooling2D((2,3), padding='same'),
    Conv2D(128, 3, padding='same', activation='relu'),
    Flatten(),
    Dense(2048, activation='relu'),
    Dropout(0.3),
    Dense(2048, activation='relu'),
    Dropout(0.3),
    Dense(257),
])

adam = Adam(lr=0.0001)
model.compile(optimizer=adam,loss='mse')


for epoch in range(100):
    model.fit_generator(train_loader.batchify(), 21200)
    test_loss = model.evaluate_generator(test_loader.batchify(), 6410)

    print("Epoch", epoch)
    print("Test loss:", test_loss)
