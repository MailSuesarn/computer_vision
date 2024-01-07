import tensorflow as tf


class PCAMPool(tf.keras.layers.Layer):
    
    def __init__(self, sigmoid=False, probability_map=True, **kwargs):
        super(PCAMPool, self).__init__(**kwargs)
        
        self.sigmoid = sigmoid
        self.probability_map = probability_map
        
        self.conv = tf.keras.layers.Conv2D(1, 1, padding="same")

    def call(self, feat_map):
        
        logit_map = self.conv(feat_map)
        prob_map = tf.math.sigmoid(logit_map)
        
        weight_map = prob_map / tf.reduce_sum(prob_map, axis=[1, 2], keepdims=True)
        
        feat = tf.reduce_sum((feat_map * weight_map), axis=[1, 2], keepdims=True)
       
        feat_emb = self.conv(feat)
        feat_emb = tf.reshape(feat_emb, shape=(-1, 1))
        
        if self.sigmoid:
            feat_emb = tf.math.sigmoid(feat_emb)
            
        if self.probability_map:
            return feat_emb, prob_map
        else:
            return feat_emb, logit_map
        
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    def get_config(self):
          config = super(PCAMPool, self).get_config()
          config.update({
              'sigmoid': self.sigmoid,
              'probability_map': self.probability_map,
          })
          return config