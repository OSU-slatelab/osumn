from data_io_seq import DataLoader
import os

# Create loader for train data
train_loader = DataLoader(
            base_dir    = os.getcwd(),
            frame_file  = "data-spectrogram/train_si84_noisy/feats.scp",
            senone_file = "clean_labels_train.txt",
            batch_size  = 5,
            buffer_size = 5*10)

print("Utts in train data:", train_loader.utt_count)

for frame_batch, senone_batch in train_loader.batchify():
    print(len(frame_batch.keys()))
    print(len(senone_batch.keys()))
