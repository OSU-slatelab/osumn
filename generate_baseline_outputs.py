import tensorflow as tf
import numpy as np
import argparse
import os

from data_io2 import DataLoader, kaldi_write_mats

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

def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def run_training(a):
    """ Define our model and train it """

    with tf.Graph().as_default():

        in_frames = tf.placeholder(tf.float32, shape=(None, 2*a.context+1, a.input_featdim))
        out_frames = tf.placeholder(tf.float32, shape=(None, a.output_featdim))
        training = tf.placeholder(tf.bool)

        layer = tf.contrib.layers.flatten(in_frames)
        layer = tf.layers.dropout(layer, rate=0, training=training)

        for i in range(a.layers):
            with tf.variable_scope("layer" + str(i)):
                layer = tf.layers.dense(
                    inputs             = layer,
                    units              = a.units)

                layer = batch_norm(layer, a.units, training) 
                layer = prelu(layer)

                layer = tf.layers.dropout(layer, rate=0, training=training)


        out = tf.layers.dense(layer, a.output_featdim)


        # Create loader
        loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = False)

        # Saver is also loader
        saver = tf.train.Saver()

        # Begin session
        sess = tf.Session()

        # Load actor weights
        saver.restore(sess, tf.train.latest_checkpoint(a.checkpoints))

        count = 0
        for batch in loader.batchify():
            if count % 100 == 0:
                print("exporting #{0}".format(count))
            count += 1

            fd = {in_frames : batch['frame'], training: False}

            # Generate outputs
            outputs = sess.run(out, fd)

            # Write outputs to file
            kaldi_write_mats(a.output_file, bytes(batch['id'], 'utf-8'), outputs)


def main():
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
    parser.add_argument("--frame_file", default="data-fbank/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod") 
    parser.add_argument("--checkpoints", default="checkpoints/", help="directory with weights")
    parser.add_argument("--output_file", default="outputs.ark")

    # Model
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--units", type=int, default=2048)

    # Data
    parser.add_argument("--input_featdim", type=int, default=771)
    parser.add_argument("--output_featdim", type=int, default=257)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--buffer_size", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    a = parser.parse_args()

    run_training(a)
    
if __name__=='__main__':
    main()    
    

