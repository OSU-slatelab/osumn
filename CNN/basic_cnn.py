import tensorflow as tf
import argparse

from data_io import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
args = parser.parse_args()
context = 5

train_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp",
    out_frame_file = "data-fbank/train_si84_clean_global_normalized/feats.scp",
    batch_size = 1024,
    buffer_size = 10,
    context = context,
    out_frame_count = 1,
    shuffle = True,
)

dev_loader = DataLoader(
    base_dir = args.base_dir,
    in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
    out_frame_file = "data-fbank/dev_dt_05_clean_global_normalized/feats.scp.mod",
    batch_size = 1024,
    buffer_size = 10,
    context = context,
    out_frame_count = 1,
    shuffle = False,
)


in_frames = tf.placeholder(tf.float32, shape=(None, 2*context+1, 771))
out_frames = tf.placeholder(tf.float32, shape=(None, 40))
training = tf.placeholder(tf.bool)

conv_in = tf.reshape(in_frames, shape=(-1, 2*context+1, 257, 3))
conv1 = tf.layers.conv2d(
    inputs      = conv_in,
    filters     = 32,
    kernel_size = 7,
    padding     = 'same',
    activation  = tf.nn.relu,
)
conv1 = tf.layers.max_pooling2d(
    inputs      = conv1,
    pool_size   = 2,
    strides     = 2,
    padding     = 'same',
)
conv2 = tf.layers.conv2d(
    inputs      = conv1,
    filters     = 64,
    kernel_size = 3,
    padding     = 'same',
    activation  = tf.nn.relu,
)
conv2 = tf.layers.max_pooling2d(
    inputs      = conv2,
    pool_size   = 2,
    strides     = 2,
    padding     = 'same',
)
conv3 = tf.layers.conv2d(
    inputs      = conv2,
    filters     = 96,
    kernel_size = 3,
    padding     = 'same',
    activation  = tf.nn.relu,
)
cnn_out = tf.contrib.layers.flatten(conv3)
fc1 = tf.layers.dense(cnn_out, 2048, tf.nn.relu)
fc1 = tf.layers.dropout(fc1, training=training)
out = tf.layers.dense(fc1, 40)

loss = tf.losses.mean_squared_error(out_frames, out)
train = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(20):
        train_loss = 0
        for in_frame_batch, out_frame_batch in train_loader.batchify():
            fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: True}
            batch_loss, _ = sess.run([loss, train], fd)
            train_loss += batch_loss
            
        print("Epoch", epoch)
        print("Train loss:", train_loss)

        test_loss = 0
        for in_frame_batch, out_frame_batch in dev_loader.batchify():
            fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: False}
            test_loss += sess.run(loss, fd)

        print("Test loss:", test_loss)
