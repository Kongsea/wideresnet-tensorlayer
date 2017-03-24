'''3D wide residual networks to classify malignancy of nodules using Keras

   Clone from Github and modified by Kong Haiyang
'''
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

    # For wide resnets
    self.blocks_per_group = [4, 3, 2]
    self.widening_factor = 4

    # Basic info
    self.batch_num = 16
    self.img_depth = 40
    self.img_row = 40
    self.img_col = 40
    self.img_channels = 1
    self.nb_classes = 2

    self.LABEL_NUMBER = 2
    self.NUM_LABEL = 1
    self.NUM_IMAGE = self.img_depth * self.img_row * self.img_col
    self.PIXEL_LENGTH = 4
    self.SAVE_SIZE = 40
    self.NUM_PREPROCESS_THREADS = 12
    self.NUM_EPOCHS = 10
    self.BIN_FILE = bin_file
    self.TEST_FILE = test_file
    self.WORK_DIRECTORY = os.path.join(os.getcwd(), 'models')
    self.SAVE_MODEL = True
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

  def get_test_data(self):
    img_buf = label_buf = ''
    with open(self.TEST_FILE, 'rb') as ftest:
      buf = ftest.read(self.length)
      while buf:
        img_buf += buf[4:]
        label_buf += buf[:4]
        buf = ftest.read(self.length)
    self.test_data = (np.frombuffer(img_buf, np.float32)).reshape(
        (-1, self.img_depth, self.img_row, self.img_col, self.img_channels))
    self.test_label = np.frombuffer(label_buf, np.float32).astype(np.int64)

  def eval_in_batches(self, data, eval_prediction, img):
    size = data.shape[0]
    if size < self.batch_num:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, self.LABEL_NUMBER), dtype=np.float32)
    for begin in xrange(0, size, self.batch_num):
      end = begin + self.batch_num
      if end <= size:
        predictions[begin:end, :] = self.sess.run(eval_prediction, feed_dict={
            img: data[begin:end, ...]})
      else:
        batch_predictions = self.sess.run(eval_prediction, feed_dict={
            img: data[-self.batch_num:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  def error_rate(self, predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) /
                    predictions.shape[0])

  def step(self):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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

    def inference(img):
      # Keras layers can be called on TensorFlow tensors:
      x = Convolution3D(8, 3, 3, 3, init='he_normal', border_mode='same')(img)

      for i in range(self.blocks_per_group[0]):
        nb_filters = 8 * self.widening_factor
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

      for i in range(self.blocks_per_group[1]):
        nb_filters = 16 * self.widening_factor
        if i == 0:
          subsample_factor = 2
        else:
          subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

      for i in range(self.blocks_per_group[2]):
        nb_filters = 16 * self.widening_factor
        if i == 0:
          subsample_factor = 2
        else:
          subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

      x = BatchNormalization(axis=4)(x)
      x = Activation('relu')(x)
      x = MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid')(x)
      x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

      x = Dense(self.nb_classes, activation='softmax')(x)
      # Readout layer
      return tf.nn.softmax(x)

    # this placeholder will contain our input digits
    img = tf.placeholder(tf.float32, shape=(
        None, self.img_depth, self.img_col, self.img_row, self.img_channels))
    labels = tf.placeholder(tf.float32, shape=(None, self.nb_classes))

    # Readout layer
    preds = inference(img)
    eval_predictions = preds

    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
    batch = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, batch * self.batch_num,
                                               self.train_size // 2, 0.95, staircase=True)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, 0.9).minimize(loss, global_step=batch)

    sstt = time.time()
    saver = tf.train.Saver(tf.global_variables())

    self.init_bin_file()
    self.get_train_data()
    self.get_test_data()

    TRAIN_FREQUENCY = self.train_size // self.batch_num // 20
    TEST_FREQUENCY = self.train_size // self.batch_num // 20
    SAVE_FREQUENCY = 2 * self.train_size // self.batch_num
    TEST_BATCH_SIZE = 10

    with tf.Session(config=config) as sess:
      self.sess = sess
      K.set_session(self.sess)
      self.sess.run(tf.local_variables_initializer())
      self.sess.run(tf.global_variables_initializer())

      summary_writer = tf.summary.FileWriter(self.WORK_DIRECTORY, self.sess.graph)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
      try:
        while not coord.should_stop():
          start_time = time.time()
          for step in xrange(int(self.NUM_EPOCHS * self.train_size) // self.batch_num):
            train_data, train_label = self.sess.run([self.images, self.labels])
            feed_dict = {img: train_data, labels: train_label}
            _, l, lr = self.sess.run(
                [optimizer, loss, learning_rate], feed_dict=feed_dict)
            if step != 0 and step % TRAIN_FREQUENCY == 0:
              et = time.time() - start_time
              print('Step %d (epoch %.2f), %.1f ms' %
                    (step, float(step) * self.batch_num / self.train_size, 1000 * et / TRAIN_FREQUENCY))
              print('Minibatch loss: {:.3f} Learning rate: {:.6f}'.format(l, lr))
              start_time = time.time()
            if step != 0 and step % TEST_FREQUENCY == 0:
              st = time.time()
              test_label_total = np.zeros(self.batch_num * TEST_BATCH_SIZE)
              prediction_total = np.zeros((self.batch_num * TEST_BATCH_SIZE, 2))
              for ti in xrange(TEST_BATCH_SIZE):
                pos = ti * self.batch_num
                offset = np.random.randint(0, self.test_size - self.batch_num, 1)[0]
                batch_data = self.test_data[offset:(offset + self.batch_num), ...]
                batch_labels = self.test_label[offset:(offset + self.batch_num)]
                predictions = self.eval_in_batches(batch_data, eval_predictions, img)
                prediction_total[pos:pos + self.batch_num, :] = predictions
                test_label_total[pos:pos + self.batch_num] = batch_labels
              test_error = self.error_rate(prediction_total, test_label_total)
              stt = np.random.randint(0, self.batch_num * TEST_BATCH_SIZE - 11, 1)[0]
              print(prediction_total[stt:stt + 10])
              print(test_label_total[stt:stt + 10])
              print('Test error: %.3f%%' % test_error)
              print('Test costs {:.2f} seconds.'.format(time.time() - st))
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


def main():
  bin_file = os.path.join(os.getcwd(), 'train_shuffle.bin')
  test_file = os.path.join(os.getcwd(), 'test_shuffle.bin')
  a = CNNEnv(bin_file, test_file)
  a.step()

if __name__ == '__main__':
  main()
