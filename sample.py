from data_io import DataLoader

# Create loader for train data
train_loader = DataLoader(
    base_dir        = "/fs/project/PAS1315/group1_chime2_data",
    in_frame_file   = "data-spectrogram/train_si84_delta_noisy/feats.scp",
    out_frame_file  = "data-fbank/train_si84_clean/feats.scp",
    batch_size      = 128,
    buffer_size     = 10,
    context         = 5,
    out_frame_count = 1,
    shuffle         = True)
i = 0
for j in range(3):
    i = 0
    for in_frame_batch, out_frame_batch in train_loader.batchify():
        # print(in_frame_batch.shape)
        # print(out_frame_batch.shape)
        # break
        i += 1
    print(i)