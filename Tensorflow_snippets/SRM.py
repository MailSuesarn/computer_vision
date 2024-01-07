"""
Author: Suesarn Wilainuch
attention mechanism: Style-based Recalibration Module(SRM) tensorflow 2 version
Paper reference: https://arxiv.org/abs/1903.10829
"""

import tensorflow as tf

class SRM(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(SRM, self).__init__(**kwargs)
    
  def build(self, input_shape):
    (_, h, w, c) = input_shape
    
    self.c = c
    # CFC: channel-wise fully connected layer
    self.conv = tf.keras.layers.Conv1D(self.c, kernel_size=2, strides=1, use_bias=False, groups=self.c)
    self.batchNorm = tf.keras.layers.BatchNormalization()
    self.eps = tf.keras.backend.epsilon()
        
    super(SRM, self).build(input_shape)

  def call(self, x):
    x_reshape = tf.keras.layers.Reshape((-1, self.c))(x) # [b, h*w, c]
    # Style pooling
    x_mean, x_var = tf.nn.moments(x_reshape, axes=[1], keepdims=True) # [b, 1, c]
    x_std = tf.sqrt(x_var + self.eps) 
    t = tf.concat([x_mean, x_std], axis=1) # [b, 2, c]
    
    # Style integration
    z = self.conv(t)
    z = self.batchNorm(z)
    g = tf.math.sigmoid(z)
    g = tf.keras.layers.Reshape((1, 1, self.c))(g)
    
    x = x * g
    return x