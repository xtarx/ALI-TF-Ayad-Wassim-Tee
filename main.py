import os
import sys
from tf_utils.common import Logger

from ALI import ALI

import tensorflow as tf
import pprint
import tf_utils.common as common
import cv2
import numpy as np

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'Size of batches to use')
flags.DEFINE_float('g_lr', 0.0001, 'Learning rate for generator')
flags.DEFINE_float('d_lr', 0.0001, 'Learning rate for discriminator')
flags.DEFINE_float('beta1', 0.5, 'Momentum term of adamoptimizer')
flags.DEFINE_integer('z_dim', 100, 'a number of z dimension layer')
flags.DEFINE_string('z_dist', 'normal', 'Distribution for z [normal]')
flags.DEFINE_string('log_dir', 'results_mias/models', 'saved image directory')
flags.DEFINE_integer('max_epoch', 500, 'A number of epochs to train')
flags.DEFINE_boolean('is_train', False, 'True for training, False for testing')
flags.DEFINE_string('data_dir', 'data', 'data directory')
flags.DEFINE_string('data', 'mnist', 'fuel data')
flags.DEFINE_boolean('monitor', True, 'True for monitoring training process')
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    f = open(os.path.join(FLAGS.log_dir, 'training.log'), 'a')
    sys.stdout = Logger(f)

    print('\n======================================')
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=run_config) as sess:
        ali = ALI(
            sess=sess,
            log_dir=FLAGS.log_dir,
            data=FLAGS.data,
            data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            z_dist=FLAGS.z_dist
        )

        if FLAGS.is_train:
            ali.train(FLAGS)
        else:
            print("TEST MODEL")
            if not ali.load_model(FLAGS.log_dir):
                raise Exception('[!] Train a model first, then run test mode')

            # geneartae
            img_dir = os.path.join(FLAGS.log_dir, 'generated_after')

            query = tf.placeholder(shape=[1, 32, 32, 1], dtype=tf.float32)

            eps = tf.random_normal([FLAGS.batch_size, 1, 1, FLAGS.z_dim])
            samples = np.random.normal(size=(100, FLAGS.z_dim)).astype(np.float32)
            reconst_x = ali.decoder(eps)
            gen_imgs = sess.run(
                reconst_x,feed_dict={ali.is_training: True})
            gen_tiled_imgs = common.img_tile(
                gen_imgs[0:100], border_color=1.0, stretch=True)
            gen_tiled_imgs = gen_tiled_imgs[:, :, ::-1]
            file_name = ''.join([img_dir, '/Sampll.jpg'])
            cv2.imwrite(file_name, gen_tiled_imgs * 255.)

            f.close()


if __name__ == '__main__':
    tf.app.run()
