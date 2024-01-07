"""
Author: Suesarn Wilainuch
Attention-based Dropout Layer(ADL) tensorflow 2 version
Paper reference: https://arxiv.org/abs/1908.10028
"""


import tensorflow as tf

class ADL(tf.keras.layers.Layer):

  def __init__(self, drop_prob=0.1, drop_thr=0.7, **kwargs):
        super(ADL, self).__init__(**kwargs)
        
        self.drop_prob = tf.math.minimum(1., tf.math.maximum(0., drop_prob))
        self.drop_thr = tf.math.minimum(1., tf.math.maximum(0., drop_thr))
        
        
  def get_importance_map(self, attention_map):
    return tf.math.sigmoid(attention_map)
        
        
  def get_drop_mask(self, attention_map):
    max_val = tf.math.reduce_max(attention_map, axis=[1, 2, 3], keepdims=True)
    thr_val = max_val * self.drop_thr
    return tf.cast(attention_map < thr_val, dtype=tf.float32, name='drop_mask')

  def select_component(self, importance_map, drop_mask):
    random_tensor = tf.random.uniform([], self.drop_prob, 1. + self.drop_prob)
    binary_tensor = tf.cast(tf.floor(random_tensor), dtype=tf.float32)
    return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask
    

  def call(self, x, training=None):
    attention_map = tf.math.reduce_mean(x, axis=-1, keepdims=True)
    importance_map = self.get_importance_map(attention_map)
    drop_mask = self.get_drop_mask(attention_map)
    selected_map = self.select_component(importance_map, drop_mask)
    adl_x = x * selected_map
    
    return tf.keras.backend.in_train_phase(adl_x, x, training=training)
 

  def get_config(self):
        config = super(ADL, self).get_config()
        config.update({
            'drop_prob': self.drop_prob,
            'drop_thr': self.drop_thr,
        })
        return config 
