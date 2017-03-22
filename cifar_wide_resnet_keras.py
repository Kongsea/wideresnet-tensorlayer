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
from keras.engine.training import collect_trainable_weights
import numpy as np


class CNNEnv:

  def __init__(self):
    # TF fix
    K.set_learning_phase(0)

    # The data, shuffled and split between train and test sets
    (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices
    # self.y_train = np_utils.to_categorical(self.y_train)
    # self.y_test = np_utils.to_categorical(self.y_test)

    # For generator
    self.num_examples = self.x_train.shape[0]
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

  def step(self):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    def zero_pad_channels(x, pad=0):
      """
      Function for Lambda layer
      """
      pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
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
      y = Convolution3D(nb_filters, 3, 3, 3, subsample=(1, 1),
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
    x = MaxPooling3D(pool_size=(8, 8, 8), strides=None, border_mode='valid')(x)
    x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

    # Readout layer
    preds = Dense(self.nb_classes, activation='softmax')(x)

    loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

    optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

    with sess.as_default():
      batch = self.next_batch(self.batch_num)
      _, l = sess.run([optimizer, loss],
                      feed_dict={img: batch[0], labels: batch[1]})
      print('Loss', l)

    acc_value = accuracy(labels, preds)

    '''
    with sess.as_default():
        acc = acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})
        print(acc)
        '''


def main(_):
  a = CNNEnv()
  a.step()

if __name__ == '__main__':
  main()
