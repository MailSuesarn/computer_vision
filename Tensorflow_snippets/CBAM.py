import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction, dim_out, kernel_size=7, activation='relu', dropout_rate=0.1, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        
        self.reduction = reduction
        self.dim_out = dim_out
        self.activation = activation
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
    
        self.avgpool =  tf.keras.layers.GlobalAveragePooling2D()
        self.maxpool =  tf.keras.layers.GlobalMaxPooling2D()
        
        self.dense_1 = tf.keras.layers.Dense(reduction, activation=activation)
        self.dense_2 = tf.keras.layers.Dense(dim_out, activation=activation)
        
        # kernel filter 7x7 follow the paper
        self.conv = tf.keras.layers.Conv2D(1, kernel_size, strides=1, padding='same')

    def call(self, x):
        
        # Channel Attention mudule
        avgpool = self.avgpool(x) # channel avgpool
        maxpool = self.maxpool(x) # channel maxpool
        # Shared MLP
        avg_out = self.dense_2(tf.nn.dropout(self.dense_1(avgpool), self.dropout_rate))
        max_out = self.dense_2(tf.nn.dropout(self.dense_1(maxpool), self.dropout_rate))

        channel = tf.keras.layers.add([avg_out, max_out])
        channel = tf.keras.layers.Activation('sigmoid')(channel) # channel sigmoid
        channel = tf.keras.layers.Reshape((1, 1, self.dim_out))(channel)
        channel_out = tf.multiply(x, channel)
    
        # Spatial Attention mudule
        avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True) # spatial avgpool
        maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True) # spatial maxpool
        spatial = tf.keras.layers.Concatenate(axis=3)([avgpool, maxpool])
        spatial = tf.nn.dropout(self.conv(spatial), self.dropout_rate) # spatial conv2d
        spatial_out = tf.keras.layers.Activation('sigmoid')(spatial) # spatial sigmoid

        CBAM_out = tf.multiply(channel_out, spatial_out)
        
        return CBAM_out
       
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
          config = super(CBAM, self).get_config()
          config.update({
              'reduction': self.reduction,
              'dim_out': self.dim_out,
              'activation': self.activation,
              'kernel_size': self.kernel_size,
              'dropout_rate': self.dropout_rate,
          })
          return config
