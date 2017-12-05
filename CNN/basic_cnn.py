import tensorflow as tf
import argparse

# Add parent directory to current path to import data loader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io import DataLoader


def batch_norm(x, shape, training, decay = 0.999, epsilon = 1e-3):
    """ Batch Norm for controlling batch statistics """
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
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2])

    train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, epsilon)

    return tf.cond(training, batch_statistics, population_statistics)

def conv_block(inputs, filters, training, dropout):
    reduced = tf.layers.conv2d(
        inputs      = inputs,
        filters     = filters,
        kernel_size = 3,
        strides     = (2,2),
        padding     = 'same',
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
    )

    shape = tf.shape(reduced)
    reduced = tf.layers.dropout(reduced, rate=dropout, noise_shape=[shape[0], 1, 1, filters], training=training)

    with tf.variable_scope("conv1"):

        conv1 = tf.nn.relu(reduced)
        conv1 = tf.layers.conv2d(
            inputs      = conv1,
            filters     = filters,
            kernel_size = 3,
            padding     = 'same',
            activation  = tf.nn.relu,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )
        conv1 = tf.layers.dropout(conv1, rate=dropout, noise_shape=[shape[0], 1, 1, filters], training=training)

    with tf.variable_scope("conv2"):

        conv2 = tf.layers.conv2d(
            inputs      = conv1,
            filters     = filters,
            kernel_size = 3,
            padding     = 'same',
            activation  = tf.nn.relu,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
        )
        conv1 = tf.layers.dropout(conv1, rate=dropout, noise_shape=[shape[0], 1, 1, filters], training=training)
        
    out = conv2 + reduced

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/fs/project/PAS1315/group1_chime2_data")
    parser.add_argument("--units", default=2048)
    parser.add_argument("--layers", default=2)
    parser.add_argument("--dropout", default=0.3)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--buffer_size", default=40)
    parser.add_argument("--context", default=10)
    parser.add_argument("--checkpoints", default="checkpoints")
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

    in_frames = tf.placeholder(tf.float32, shape=(None, 2*args.context + 1, 257))
    out_frames = tf.placeholder(tf.float32, shape=(None, 257))
    training = tf.placeholder(tf.bool)

    block = tf.reshape(in_frames, shape=(-1, 2*args.context+1, 257, 1))

    filters = [128, 128, 256, 256]
    for i, f in enumerate(filters):
        with tf.variable_scope("block{0}".format(i)):
            block = conv_block(block, f, training, args.dropout)

    fc = tf.contrib.layers.flatten(block)
    fc = tf.layers.dropout(fc, rate=args.dropout, training=training)

    for i in range(args.layers):
        with tf.variable_scope("fc{0}".format(i)):
            fc = tf.layers.dense(fc, args.units, tf.nn.relu)
            fc = tf.layers.dropout(fc, rate=args.dropout, training=training)

    out = tf.layers.dense(fc, 257)

    loss = tf.losses.mean_squared_error(out_frames, out)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.0001, global_step, 1e4, 0.95)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        min_loss = 1
        for epoch in range(0, 100):
            print("Epoch", epoch)

            train_loss = 0
            count = 0
            for in_frame_batch, out_frame_batch in train_loader.batchify(shuffle_batches=False,include_deltas=False):
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: True}
                batch_loss, _ = sess.run([loss, train], fd)
                train_loss += batch_loss
                count += 1
                print("\r{0}/{1} - loss: {2}".format(count,5300*1024//args.batch_size,train_loss/count), end='', flush=True)

            print("\nTrain loss:", train_loss / count)

            test_loss = 0
            count = 0
            for in_frame_batch, out_frame_batch in dev_loader.batchify(include_deltas=False):
                fd = {in_frames: in_frame_batch, out_frames: out_frame_batch, training: False}
                test_loss += sess.run(loss, fd)
                count += 1

            test_loss = test_loss / count

            print("Test loss:", test_loss)

            if test_loss < min_loss:
                saver.save(sess, os.path.join(args.checkpoints, "model-{0}.ckpt".format(test_loss)))
