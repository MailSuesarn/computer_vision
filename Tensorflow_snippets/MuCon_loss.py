class MuConLoss(tf.keras.losses.Loss):
    '''
    Computes the Multi-label Classification with Contrastive Loss (MuCon loss)
    
    A variant of Multi-class N-pair Loss
    Later used in SimCLR and SupCon and modify to NT-Xent (Normalized temperature-scaled cross entropy loss)
    Reference paper: 
        Contrast Learning Visual Attention for Multi Label Classification
    Args:
        labels: multi-hot encoding of shape or in multi-class it must be one-hot encoding [bsz, num_class].
        logits: label-level embedding of all images of shape [bsz, num_class, dim].
    
    ** make sure logits(embedding) should be l2-normalized **
    '''
    def __init__(self, temperature=0.1, **kwargs):
        
        super().__init__(**kwargs)
        
        self.temperature = temperature
        
    def call(self, labels, logits):
        
        # mask out for create set "I" (label-level embedding with "active" labels)
        labels = tf.cast(labels, tf.float32)
                
        mask_I = tf.math.greater(labels, 0.0)
        
        logit_I = tf.boolean_mask(logits, mask_I)
        
        labels_index = tf.cast(tf.range(tf.shape(labels)[-1]), dtype=tf.float32) * tf.cast(tf.ones_like(labels), dtype=tf.float32)

        labels_I = tf.boolean_mask(labels_index, mask_I)

        labels_I = tf.expand_dims(labels_I, axis=-1)

        mask = tf.cast(tf.equal(labels_I, tf.transpose(labels_I)), tf.float32)
        
        mask_i_neq_j = tf.ones_like(mask) - tf.eye(tf.shape(mask)[0], dtype=tf.float32)
                
        # exclude diagonal
        mask = mask * mask_i_neq_j

        # calculate pair-wise similarity matrix
        anchor_dot_contrast = tf.math.divide(
                              tf.linalg.matmul(logit_I, tf.transpose(logit_I)), 
                              tf.constant(self.temperature, dtype=tf.float32))

        # for numerical stability
        logits_max = tf.math.reduce_max(anchor_dot_contrast, axis=-1, keepdims=True)
        anchor_dot_contrast_stable = anchor_dot_contrast - logits_max

        # exclude self-pair for denominator
        exp_logits = tf.math.exp(anchor_dot_contrast_stable) * mask_i_neq_j
        log_prob = anchor_dot_contrast_stable - tf.math.log(tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True))

        # cardinality of positive set
        P_i = tf.reduce_sum(mask, axis=-1)

        # compute mean of log-likelihood over positive
        # this may introduce NaNs due to zero division
        mean_log_prob_pos = -tf.reduce_sum(mask * log_prob, axis=-1)[P_i > 0] / P_i[P_i > 0]

        loss = tf.reduce_mean(mean_log_prob_pos)

        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature
        })
        return config