"""
Author: Suesarn Wilainuch
Aggregate function Top T% Pooling tensorflow 2 version
Paper reference: https://arxiv.org/abs/2002.07613
"""

import tensorflow as tf

class TopTPercentAggregation(tf.keras.layers.Layer):
    def __init__(self, percent_t=2., **kwargs):
        super(TopTPercentAggregation, self).__init__(**kwargs)
        self.percent_t = percent_t

    def call(self, x):
        h, w, num_class = x.shape[1], x.shape[2], x.shape[3]
        x_flatten = tf.keras.layers.Reshape((num_class, -1))(x)
        top_t = int(round(h * w * (self.percent_t/100)))
        selected_area, _ = tf.math.top_k(x_flatten, k=top_t)
        selected_area = tf.math.reduce_mean(selected_area, axis=2, keepdims=False)
        return selected_area
    
    def get_config(self):
        config = super(TopTPercentAggregation, self).get_config()
        config.update({
            'percent_t': self.percent_t
        })
        return config