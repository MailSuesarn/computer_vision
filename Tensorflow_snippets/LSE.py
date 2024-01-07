"""
Author: Suesarn Wilainuch
Aggregate function Log-Sum-Exp (LSE) tensorflow 2 version
Paper reference: https://arxiv.org/abs/1411.6228
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops

class LSE(tf.keras.layers.Layer):
    
    def __init__(self, r=5, **kwargs):
        super(LSE, self).__init__(**kwargs)
        self.r = r
        
    def call(self, x):
        alpha = x - math_ops.reduce_max(x, axis=[1, 2], keepdims=True)
        beta = math_ops.exp(alpha * self.r)
        gamma = math_ops.reduce_sum(beta, axis=[1, 2], keepdims=False)
        lse = math_ops.log(gamma) * (1./self.r) 
        return lse
    
    def get_config(self):
        config = super(LSE, self).get_config()
        config.update({
            'r': self.r
        })
        return config