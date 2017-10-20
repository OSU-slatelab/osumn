import tensorflow as tf

from data_io import DataLoader


train_loader = DataLoader(
    base_dir = "/fs/project/PAS1315/group1_chime2_data",
    in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp",
    out_frame_file = "data-fbank/train_si84_clean_global_normalized/feats.scp",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = True,
)

dev_loder = DataLoader(
    base_dir = "/fs/project/PAS1315/group1_chime2_data",
    in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-fbank/dev_dt_05_clean_global_normalized/feats.scp.mod",
    batch_size = 1024,
    buffer_size = 10,
    context = 5,
    out_frame_count = 1,
    shuffle = False,
)


in_frames = tf.placeholder(tf.float32, shape=(None, 11, 771))
out_frames = tf.placeholder(tf.float32, shape=(None, 40))

flat = tf.contrib.layers.flatten(in_frames)
fc1 = tf.layers.dense(flat, 2048, tf.nn.relu)
fc2 = tf.layers.dense(fc1, 2048, tf.nn.relu)
out = tf.layers.dense(fc2, 40)

loss = tf.losses.mean_squared_error(out_frames, out)
train = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(20):
        train_loss = 0
        for in_frame_batch, out_frame_batch in train_loader.batchify():
            fd = {in_frames: in_frame_batch, out_frames: out_frame_batch}
            batch_loss, _ = sess.run([loss, train], fd)
            train_loss += batch_loss
            
        print("Epoch", epoch)
        print("Train loss:", train_loss)

        for in_frame_batch, out_frame_batch in dev_loader.batchify():
            fd = {in_frames: in_frame_batch, out_frames: out_frame_batch}
            test_loss += sess.run(loss, fd)

        print("Test loss:", test_loss)
