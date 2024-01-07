import tensorflow as tf


class CSRA(tf.keras.layers.Layer):
    def __init__(self, lam=0.2, T=1, **kwargs):
        super(CSRA, self).__init__(**kwargs)
        
        self.lam = lam  # Lambda
        self.T = T  # Temperature

    def call(self, x):
        
        base_logit = tf.math.reduce_mean(x, axis=[1, 2], keepdims=False)
        
        if self.T == 99:
            att_logit = tf.math.reduce_max(x, axis=[1, 2], keepdims=False)
            
        else:
            _, h, w, c = x.shape
            x = tf.reshape(x, [-1, h*w, c])
            
            score_soft = tf.nn.softmax(x*self.T, axis=1)
            att_logit = tf.math.reduce_sum(x*score_soft, axis=1, keepdims=False)
            
        out = base_logit + (self.lam * att_logit)
        return out
       
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
          config = super(CSRA, self).get_config()
          config.update({
              'lam': self.lam,
              'T': self.T,
          })
          return config

