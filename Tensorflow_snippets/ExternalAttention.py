"""
Author: Suesarn Wilainuch
attention mechanism: ExternalAttention tensorflow 2 version
Paper reference: https://arxiv.org/pdf/2105.02358.pdf
"""


import tensorflow as tf

class ExternalAttention(tf.keras.layers.Layer):

  def __init__(self, c=64, n=8, **kwargs):
    super(ExternalAttention, self).__init__(**kwargs)
    
    self.c = c
    self.n = n
    
    self.q = tf.keras.layers.Conv2D(self.c, kernel_size=1, strides=(1, 1), padding='same', use_bias=False)
    self.mk = tf.keras.layers.Dense(self.c//self.n, use_bias=False)
    self.mv = tf.keras.layers.Dense(self.c, use_bias=False)
        
  def call(self, x):
    query = self.q(x)
    
    q_reshape = tf.keras.layers.Reshape((-1, self.c))(query)

    beta = self.mk(q_reshape)
    beta = tf.nn.softmax(beta, axis=1)
    beta = beta / tf.math.reduce_sum(beta, axis=2, keepdims=True)
    
    out = self.mv(beta)
    out = tf.reshape(out, shape=tf.shape(x))  
    
    return out
