import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import os


class CNNEnv:

  def __init__(self, bin_file, test_file):

    # The data, shuffled and split between train and test sets
    # self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(
    #     shape=(-1, 32, 32, 3), plotable=False)

    # For generator
    # self.num_examples = self.x_train.shape[0]
    self.index_in_epoch = 0
    self.epochs_completed = 0

    # For wide resnets
    self.blocks_per_group = 4
    self.widening_factor = 4

    # Basic info
    self.batch_num = 64
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

  def reset(self, first):
    self.first = first
    if self.first is True:
      self.sess.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.InteractiveSession(config=config)

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
    self.labels = tf.one_hot(tf.reshape(self.labels, [-1]), 2)

  def step(self):

    def zero_pad_channels(x, pad=0):
      """
      Function for Lambda layer
      """
      pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
      return tf.pad(x, pattern)

    def residual_block(x, count, nb_filters=16, subsample_factor=1):
      prev_nb_channels = x.outputs.get_shape().as_list()[4]

      if subsample_factor > 1:
        subsample = [1, subsample_factor, subsample_factor, subsample_factor, 1]
        # shortcut: subsample + zero-pad channel dim
        name_pool = 'pool_layer' + str(count)
        shortcut = tl.layers.PoolLayer(x,
                                       ksize=subsample,
                                       strides=subsample,
                                       padding='VALID',
                                       pool=tf.nn.max_pool3d(),
                                       name=name_pool)

      else:
        subsample = [1, 1, 1, 1, 1]
        # shortcut: identity
        shortcut = x

      if nb_filters > prev_nb_channels:
        name_lambda = 'lambda_layer' + str(count)
        shortcut = tl.layers.LambdaLayer(
            shortcut,
            zero_pad_channels,
            fn_args={'pad': nb_filters - prev_nb_channels},
            name=name_lambda)

      name_norm = 'norm' + str(count)
      y = tl.layers.BatchNormLayer(x,
                                   decay=0.999,
                                   epsilon=1e-05,
                                   is_train=True,
                                   name=name_norm)

      name_conv = 'conv_layer' + str(count)
      y = tl.layers.Conv3dLayer(y,
                                act=tf.nn.relu,
                                shape=[3, 3, 3, prev_nb_channels, nb_filters],
                                strides=subsample,
                                padding='SAME',
                                name=name_conv)

      name_norm_2 = 'norm_second' + str(count)
      y = tl.layers.BatchNormLayer(y,
                                   decay=0.999,
                                   epsilon=1e-05,
                                   is_train=True,
                                   name=name_norm_2)

      prev_input_channels = y.outputs.get_shape().as_list()[4]
      name_conv_2 = 'conv_layer_second' + str(count)
      y = tl.layers.Conv3dLayer(y,
                                act=tf.nn.relu,
                                shape=[3, 3, 3,  prev_input_channels, nb_filters],
                                strides=[1, 1, 1, 1, 1],
                                padding='SAME',
                                name=name_conv_2)

      name_merge = 'merge' + str(count)
      out = tl.layers.ElementwiseLayer([y, shortcut],
                                       combine_fn=tf.add,
                                       name=name_merge)

      return out

    # Placeholders
    learning_rate = tf.placeholder(tf.float32)
    img = tf.placeholder(tf.float32, shape=[
                         self.batch_num, self.img_depth, self.img_row, self.img_col, self.img_channels])
    labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

    x = tl.layers.InputLayer(img, name='input_layer')
    x = tl.layers.Conv3dLayer(x,
                              act=tf.nn.relu,
                              shape=[3, 3, 3, 1, 16],
                              strides=[1, 1, 1, 1, 1],
                              padding='SAME',
                              name='cnn_layer_first')

    for i in range(self.blocks_per_group):
      nb_filters = 16 * self.widening_factor
      count = i
      x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=1)

    for i in range(self.blocks_per_group):
      nb_filters = 32 * self.widening_factor
      if i == 0:
        subsample_factor = 2
      else:
        subsample_factor = 1
      count = i + self.blocks_per_group
      x = residual_block(x, count, nb_filters=nb_filters,
                         subsample_factor=subsample_factor)

    for i in range(self.blocks_per_group):
      nb_filters = 64 * self.widening_factor
      if i == 0:
        subsample_factor = 2
      else:
        subsample_factor = 1
      count = i + 2 * self.blocks_per_group
      x = residual_block(x, count, nb_filters=nb_filters,
                         subsample_factor=subsample_factor)

    x = tl.layers.BatchNormLayer(x,
                                 decay=0.999,
                                 epsilon=1e-05,
                                 is_train=True,
                                 name='norm_last')

    x = tl.layers.PoolLayer(x,
                            ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1],
                            padding='VALID',
                            pool=tf.nn.max_pool3d(),
                            name='pool_last')

    x = tl.layers.FlattenLayer(x, name='flatten')

    x = tl.layers.DenseLayer(x,
                             n_units=self.nb_classes,
                             act=tf.identity,
                             name='fc')

    output = x.outputs

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels))

    correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_params = x.all_params
    train_op = tf.train.MomentumOptimizer(
        learning_rate, 0.9, use_locking=False).minimize(cost, var_list=train_params)

    self.sess.run(tf.global_variables_initializer())

    self.init_bin_file()
    self.get_train_data()

    # for i in range(10):
    #   train_data, train_label = sess.run([self.images, self.labels])
    #   feed_dict = {img: train_data, labels: train_label, learning_rate: 0.01}
    #   feed_dict.update(x.all_drop)
    #   _, l, ac = self.sess.run([train_op, cost, acc], feed_dict=feed_dict)
    #   print('loss', l)
    #   print('acc', ac)

    sstt = time.time()
    saver = tf.train.Saver(tf.global_variables())

    TRAIN_FREQUENCY = self.train_size // self.batch_num // 2
    TEST_FREQUENCY = self.train_size // self.batch_num // 2
    SAVE_FREQUENCY = 10 * self.train_size // self.batch_num

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      summary_writer = tf.summary.FileWriter(self.WORK_DIRECTORY, sess.graph)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        while not coord.should_stop():
          start_time = time.time()
          for step in xrange(int(self.NUM_EPOCHS * self.train_size) // self.batch_num):
            train_data, train_label = sess.run([train_data_node, train_label_node])
            feed_dict = {data_node: train_data,
                         labels_node: train_label, keep_hidden: 0.5}
            _, l, lr = sess.run(
                [optimizer, loss, learning_rate], feed_dict=feed_dict)
            if step != 0 and step % TRAIN_FREQUENCY == 0:
              et = time.time() - start_time
              print('Step %d (epoch %.2f), %.1f ms' %
                    (step, float(step) * self.batch_num / self.train_size, 1000 * et / TRAIN_FREQUENCY))
              print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
              start_time = time.time()
            # if step != 0 and step % TEST_FREQUENCY == 0:
            #   st = time.time()
            #   test_label_total = np.zeros(
            #       (test_size // self.batch_num * self.batch_num))
            #   prediction_total = np.zeros(
            #       (test_size // self.batch_num * self.batch_num, 2))
            #   for ti in xrange(test_size // self.batch_num):
            #     offset = ti * self.batch_num
            #     batch_data = test_data[offset:(offset + self.batch_num), ...]
            #     batch_labels = test_label[offset:(offset + self.batch_num)]
            #     predictions = eval_in_batches(
            #         batch_data, sess, eval_predictions, data_node, keep_hidden)
            #     prediction_total[offset:offset + self.batch_num, :] = predictions
            #     test_label_total[offset:offset + self.batch_num] = batch_labels
            #   test_error = error_rate(prediction_total, test_label_total)
            #   stt = np.random.randint(0, test_size - 11, 1)[0]
            #   print(prediction_total[stt:stt + 10])
            #   print(test_label_total[stt:stt + 10])
            #   print('Test error: %.3f%%' % test_error)
            #   print('Test costs {:.2f} seconds.'.format(time.time() - st))
            #   start_time = time.time()
            if step % SAVE_FREQUENCY == 0 and step != 0:
              if self.SAVE_MODEL:
                checkpoint_path = os.path.join(self.WORK_DIRECTORY, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
              start_time = time.time()
          else:
            if self.SAVE_MODEL:
              checkpoint_path = os.path.join(self.WORK_DIRECTORY, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()
      except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
      finally:
        pass
      coord.join(threads)
    print('All costs {:.2f} seconds...'.format(time.time() - sstt))


def main():
  bin_file = os.path.join(os.getcwd(), 'train_shuffle.bin')
  test_file = os.path.join(os.getcwd(), 'train_shuffle.bin')
  a = CNNEnv(bin_file, test_file)
  a.reset(first=False)
  a.step()

if __name__ == '__main__':
  main()
