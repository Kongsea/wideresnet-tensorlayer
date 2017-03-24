import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution3D, MaxPooling3D, Dropout, Reshape
from keras.engine import merge, Input, Model
import numpy as np
import os
import time


class CNNEnv:

  def __init__(self, bin_file, test_file):
    # TF fix
    K.set_learning_phase(0)

    # For generator
    self.index_in_epoch = 0
    self.epochs_completed = 0

    # For wide resnets
    self.blocks_per_group = 4
    self.widening_factor = 4

    # Basic info
    self.batch_num = 8
    self.img_depth = 40
    self.img_row = 40
    self.img_col = 40
    self.img_channels = 1
    self.nb_classes = 2

    self.NUM_LABEL = 1
    self.NUM_IMAGE = self.img_depth * self.img_row * self.img_col
    self.PIXEL_LENGTH = 4
    self.SAVE_SIZE = 40
    self.NUM_PREPROCESS_THREADS = 12
    self.NUM_EPOCHS = 10
    self.BIN_FILE = bin_file
    self.TEST_FILE = test_file
    self.WORK_DIRECTORY = os.path.join(os.getcwd(), 'models')
    self.train_size = 0
    self.test_size = 0
    self.length = (self.NUM_LABEL + self.NUM_IMAGE) * self.PIXEL_LENGTH

  def get_size(self):
    with open(self.BIN_FILE, 'rb') as f:
      buf = f.read(self.length)
      while buf:
        self.train_size += 1
        buf = f.read(self.length)

    with open(self.TEST_FILE, 'rb') as f:
      buf = f.read(self.length)
      while buf:
        self.test_size += 1
        buf = f.read(self.length)

  def init_bin_file(self):
    self.get_size()
    bin_file_name = [self.BIN_FILE]
    for f in bin_file_name:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    self.fqb = tf.train.string_input_producer(bin_file_name)
    record_bytes = (self.NUM_LABEL + self.NUM_IMAGE) * self.PIXEL_LENGTH
    self.rb = tf.FixedLengthRecordReader(record_bytes=record_bytes)

  def _get_data(self):
    key, value = self.rb.read(self.fqb)
    record_bytes = tf.decode_raw(value, tf.float32)
    self.label = tf.cast(tf.slice(record_bytes, [0], [self.NUM_LABEL]), tf.int64)
    self.image = tf.reshape(tf.slice(record_bytes, [self.NUM_LABEL], [self.NUM_IMAGE]),
                            shape=[self.SAVE_SIZE, self.SAVE_SIZE, self.SAVE_SIZE, 1])

  def get_train_data(self):
    self._get_data()
    min_queue_examples = self.batch_num * 100
    self.labels, self.images = tf.train.batch(
        [self.label, self.image],
        batch_size=self.batch_num,
        num_threads=self.NUM_PREPROCESS_THREADS,
        capacity=min_queue_examples + 3 * self.batch_num)
    # self.labels = tf.reshape(self.labels, [-1])
    self.labels = tf.one_hot(tf.reshape(self.labels, [-1]), 2)

  def step(self):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    K.set_session(self.sess)

    def zero_pad_channels(x, pad=0):
      """
      Function for Lambda layer
      """
      pattern = [[0, 0], [0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
      return tf.pad(x, pattern)

    def residual_block(x, nb_filters=16, subsample_factor=1):
      prev_nb_channels = K.int_shape(x)[4]

      if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = MaxPooling3D(pool_size=subsample)(x)
      else:
        subsample = (1, 1, 1)
        # shortcut: identity
        shortcut = x

      if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={
                              'pad': nb_filters - prev_nb_channels})(shortcut)

      y = BatchNormalization(axis=4)(x)
      y = Activation('relu')(y)
      y = Convolution3D(nb_filters, 3, 3, 3, subsample=subsample,
                        init='he_normal', border_mode='same')(y)
      y = BatchNormalization(axis=4)(y)
      y = Activation('relu')(y)
      y = Convolution3D(nb_filters, 3, 3, 3, subsample=(1, 1, 1),
                        init='he_normal', border_mode='same')(y)

      out = merge([y, shortcut], mode='sum')

      return out

    # this placeholder will contain our input digits
    img = tf.placeholder(tf.float32, shape=(
        None, self.img_depth, self.img_col, self.img_row, self.img_channels))
    labels = tf.placeholder(tf.float32, shape=(None, self.nb_classes))
    # img = K.placeholder(ndim=4)
    # labels = K.placeholder(ndim=1)

    # Keras layers can be called on TensorFlow tensors:
    x = Convolution3D(16, 3, 3, 3, init='he_normal', border_mode='same')(img)

    for i in range(self.blocks_per_group):
      nb_filters = 16 * self.widening_factor
      x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

    for i in range(self.blocks_per_group):
      nb_filters = 32 * self.widening_factor
      if i == 0:
        subsample_factor = 2
      else:
        subsample_factor = 1
      x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

    for i in range(self.blocks_per_group):
      nb_filters = 64 * self.widening_factor
      if i == 0:
        subsample_factor = 2
      else:
        subsample_factor = 1
      x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

    x = BatchNormalization(axis=4)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid')(x)
    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

    # Readout layer
    preds = Dense(self.nb_classes, activation='softmax')(x)

    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

    optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

    sstt = time.time()
    saver = tf.train.Saver(tf.global_variables())

    TRAIN_FREQUENCY = self.train_size // self.batch_num // 200
    TEST_FREQUENCY = self.train_size // self.batch_num // 200
    SAVE_FREQUENCY = 10 * self.train_size // self.batch_num

    self.sess.run(tf.local_variables_initializer())
    self.sess.run(tf.global_variables_initializer())

    self.init_bin_file()
    self.get_train_data()

    summary_writer = tf.summary.FileWriter(self.WORK_DIRECTORY, self.sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    try:
      while not coord.should_stop():
        start_time = time.time()
        for step in xrange(int(self.NUM_EPOCHS * self.train_size) // self.batch_num):
          train_data, train_label = self.sess.run([self.images, self.labels])
          feed_dict = {img: train_data, labels: train_label}
          _, l = self.sess.run([optimizer, loss], feed_dict=feed_dict)
          if step != 0 and step % TRAIN_FREQUENCY == 0:
            et = time.time() - start_time
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * self.batch_num / self.train_size, 1000 * et / TRAIN_FREQUENCY))
            print('Minibatch loss: %.3f' % (l))
            start_time = time.time()
          if step % SAVE_FREQUENCY == 0 and step != 0:
            if self.SAVE_MODEL:
              checkpoint_path = os.path.join(self.WORK_DIRECTORY, 'model.ckpt')
              saver.save(self.sess, checkpoint_path, global_step=step)
            start_time = time.time()
        else:
          if self.SAVE_MODEL:
            checkpoint_path = os.path.join(self.WORK_DIRECTORY, 'model.ckpt')
            saver.save(self.sess, checkpoint_path, global_step=step)
          coord.request_stop()
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      pass
    coord.join(threads)
    print('All costs {:.2f} seconds...'.format(time.time() - sstt))

    # with sess.as_default():
    #   batch = self.next_batch(self.batch_num)
    #   _, l = sess.run([optimizer, loss],
    #                   feed_dict={img: batch[0], labels: batch[1]})
    #   print('Loss', l)

    # acc_value = accuracy(labels, preds)

    '''
    with sess.as_default():
        acc = acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})
        print(acc)
        '''


def main():
  bin_file = os.path.join(os.getcwd(), 'train_shuffle.bin')
  test_file = os.path.join(os.getcwd(), 'train_shuffle.bin')
  a = CNNEnv(bin_file, test_file)
  a.step()

if __name__ == '__main__':
  main()
