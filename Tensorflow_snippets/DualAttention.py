"""
Author: Suesarn Wilainuch
attention mechanism: DualAttention tensorflow 2 version
Paper reference: https://arxiv.org/abs/1809.02983
"""

import tensorflow as tf

class PositionAttention(tf.keras.layers.Layer):

  def __init__(self, c=1280, n=8, **kwargs):
    super(PositionAttention, self).__init__(**kwargs)
    
    self.n = n
    self.c = c
    
    self.k = tf.keras.layers.Conv2D(self.c//self.n, kernel_size=1, strides=(1, 1), padding='same', use_bias=False)
    self.q = tf.keras.layers.Conv2D(self.c//self.n, kernel_size=1, strides=(1, 1), padding='same', use_bias=False)
    self.v = tf.keras.layers.Conv2D(self.c, kernel_size=1, strides=(1, 1), padding='same', use_bias=False)
        
  def build(self, input_shape):
    self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
    super(PositionAttention, self).build(input_shape)

  def call(self, x):
    k = self.k(x)
    q = self.q(x)
    v = self.v(x)
    
    k_reshape = tf.reshape(k, shape=[tf.shape(k)[0], tf.shape(k)[1] * tf.shape(k)[2], tf.shape(k)[3]])
    q_reshape = tf.reshape(q, shape=[tf.shape(q)[0], tf.shape(q)[1] * tf.shape(q)[2], tf.shape(q)[3]])
    v_reshape = tf.reshape(v, shape=[tf.shape(v)[0], tf.shape(v)[1] * tf.shape(v)[2], tf.shape(v)[3]])
        
    s = tf.linalg.matmul(k_reshape, q_reshape, transpose_a=False, transpose_b=True)
    beta = tf.nn.softmax(s, axis=-1)
    v = tf.linalg.matmul(beta, v_reshape) 
    v = tf.reshape(v, shape=tf.shape(x))  
    x = (self.gamma * v) + x
    return x

  def compute_output_shape(self, input_shape):
        return input_shape



class ChannelAttention(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(ChannelAttention, self).__init__(**kwargs)

  def build(self, input_shape):
    self.beta = self.add_weight(name='beta', shape=[1], initializer='zeros', trainable=True)
    super(ChannelAttention, self).build(input_shape)


  def call(self, x):
    
    k_reshape = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3]])
    q_reshape = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3]])
    v_reshape = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[3]])
    
    s = tf.linalg.matmul(k_reshape, q_reshape, transpose_a=True, transpose_b=False)
    beta = tf.nn.softmax(s, axis=-1)
    v = tf.linalg.matmul(v_reshape, beta) 
    v = tf.reshape(v, shape=tf.shape(x))  
    x = (self.beta * v) + x
    return x

  def compute_output_shape(self, input_shape):
        return input_shape


class DualAttention(tf.keras.layers.Layer):

  def __init__(self, c=1280, n=8, **kwargs):
    super(DualAttention, self).__init__(**kwargs)
    
    self.n = n
    self.c = c
    
    self.PositionAttention = PositionAttention(c=self.c, n=self.n)
    self.ChannelAttention = ChannelAttention()
    
  def build(self, input_shape):
    super(DualAttention, self).build(input_shape)
    
  def call(self, x):
    pa = self.PositionAttention(x)
    ca = self.ChannelAttention(x)
    fuse = pa + ca
    return fuse

  def get_config(self):
        config = super(DualAttention, self).get_config()
        config.update({
            'n': self.n,
            'c': self.c,
        })
        return config
    
  def compute_output_shape(self, input_shape):
    return input_shape