import tensorflow as tf


class BatchInstanceNormalization(tf.keras.layers.Layer):
  
    def __init__(self, **kwargs):
        
        super(BatchInstanceNormalization, self).__init__(**kwargs)
            
        self.eps = 1e-6
        
    def build(self, input_shape):
        ch = input_shape[-1]
          
        self.rho = self.add_weight(shape=[ch], 
                                   name='rho',
                                   initializer="ones",
                                   constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
                                   trainable=True)
                                   
        self.gamma = self.add_weight(shape=[ch], 
                                     name='gamma',  
                                     initializer="ones",
                                     trainable=True)
                                   
        self.beta = self.add_weight(shape=[ch], 
                                    name='beta', 
                                    initializer="zeros",
                                    trainable=True)
        
        super(BatchInstanceNormalization, self).build(input_shape)
            
        
    def call(self, x):
               
        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + self.eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.eps))
        
        x_hat = (self.rho * x_batch) + ((1 - self.rho) * x_ins)
        x_hat = (x_hat * self.gamma) + self.beta

        return x_hat
    
    
