import tensorflow as tf
import argparse

# Add parent directory to current path to import data loader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io import DataLoader

def batch_norm(x, shape, training, decay = 0.999, epsilon = 1e-3):
    """ Batch Norm for controlling batch statistics """
    #Assume 2d [batch, values] tensor
    beta = tf.get_variable(name='beta', shape=shape, initializer=tf.constant_initializer(0.0)
                               , trainable=True)
    gamma = tf.get_variable(name='gamma', shape=shape, initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
    pop_mean = tf.get_variable('pop_mean',
                               shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)
    pop_var = tf.get_variable('pop_var',
                              shape,
                              initializer=tf.constant_initializer(1.0),
                              trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, [0])

    train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, epsilon)

    return tf.cond(training, batch_statistics, population_statistics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
    parser.add_argument("--units", default=2048)
    parser.add_argument("--layers", default=2)
    parser.add_argument("--dropout", default=0.3)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--buffer_size", default=10)
    parser.add_argument("--context", default=5)
    args = parser.parse_args()

    train_loader = DataLoader(
        base_dir = args.base_dir,
        in_frame_file = "data-spectrogram/train_si84_delta_noisy_global_normalized/feats.scp.mod",
        out_frame_file = "data-spectrogram/train_si84_clean_global_normalized/feats.scp.mod",
        batch_size = args.batch_size,
        buffer_size = args.buffer_size,
        context = args.context,
        out_frame_count = 1,
        shuffle = True,
    )

    dev_loader = DataLoader(
        base_dir = args.base_dir,
        in_frame_file = "data-spectrogram/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod",
        out_frame_file = "data-spectrogram/dev_dt_05_clean_global_normalized/feats.scp.mod",
        batch_size = args.batch_size,
        buffer_size = args.buffer_size,
        context = args.context,
        out_frame_count = 1,
        shuffle = False,
    )


    in_frames = tf.placeholder(tf.float32, shape=(None, 2*args.context + 1, 771))
    out_frames = tf.placeholder(tf.float32, shape=(None, 257))
    training = tf.placeholder(tf.bool)

    conv_in = tf.reshape(in_frames, shape=(-1, 2*args.context+1, 257, 3))
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
    fc1 = batch_norm(fc1, 2048, training)
    fc1 = tf.layers.dropout(fc1, training=training)
    out = tf.layers.dense(fc1, 257)

    loss = tf.losses.mean_squared_error(out_frames, out)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 1e4, 0.95)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            print("Epoch", epoch)

            train_loss = 0
            for in_frame_batch, out_frame_batch in train_loader.batchify():
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: True}
                batch_loss, _ = sess.run([loss, train], fd)
                train_loss += batch_loss

            print("Train loss:", train_loss / 5300)

            test_loss = 0
            for in_frame_batch, out_frame_batch in dev_loader.batchify():
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: False}
                test_loss += sess.run(loss, fd)

            print("Test loss:", test_loss / 1612)
