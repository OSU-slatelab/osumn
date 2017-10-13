from data_io import DataLoader
import os

# Create loader for train data
train_loader = DataLoader(
            base_dir    = os.getcwd(),
            frame_file  = "data-spectrogram/train_si84_noisy/feats.scp",
            senone_file = "clean_labels_train.txt",
            batch_size  = 128,
            buffer_size = 128*10,
            context     = 5,
            out_frames  = 1,
            shuffle     = True)

print("Frames in train data:", train_loader.frame_count)

for frame_batch, senone_batch in train_loader.batchify():
    print(frame_batch.shape)
    print(senone_batch.shape)
