import tensorflow as tf
import tensorflow_probability as tfp

    
class MixStyle(tf.keras.layers.Layer):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.1, **kwargs):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
        """
        super(MixStyle, self).__init__(**kwargs)
        
        self.p = p
        self.alpha = alpha
        self.beta = tfp.distributions.Beta(self.alpha, self.alpha)
        self.eps = 1e-6
        
        
    def call(self, x, training=None):
        if tf.random.uniform([]) <= tf.constant(self.p):
            return x
        else:
            batch_size = tf.shape(x)[0]

            mu, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            sig = tf.sqrt(var + self.eps)
            x_normed = (x - mu) / sig

            lmda = self.beta.sample([batch_size, 1, 1, 1])

            mu2 = tf.stop_gradient(tf.random.shuffle(mu, seed=1))
            sig2 = tf.stop_gradient(tf.random.shuffle(sig, seed=1))

            mu_mix = (mu*lmda) + (mu2 * (1-lmda))
            sig_mix = (sig*lmda) + (sig2 * (1-lmda))

            x_normed = (x_normed*sig_mix) + mu_mix

            return tf.keras.backend.in_train_phase(x_normed, x, training=training)


    def get_config(self):
        config = super(MixStyle, self).get_config()
        config.update({
                        'p': self.p,
                        'alpha': self.alpha,
        })
        return config 