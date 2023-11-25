from keras.layers import Layer
import tensorflow as tf
import numpy as np
class CustomAttentionLayer(tf.keras.layers.Layer):
    """Custom Attention Layer.

    Args:
        units (int): The number of attention units. 

    Usage:
        This layer adds a custom attention mechanism to your neural network model.

    Example:
        ```python
        attn_layer = CustomAttentionLayer(units=64)
        output = attn_layer(input_data)
        ```
    """
    def __init__(self, units, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.units = units 

    def build(self, input_shape):
        self.Wa = self.add_weight(name='Wa', shape=(self.units, input_shape[-1]),
                                 initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(self.units,),
                                 initializer='uniform', trainable=True)
        self.va = self.add_weight(name='va', shape=(1, self.units),
                                 initializer='uniform', trainable=True)
        super(CustomAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        hl = inputs 

        matmul = tf.matmul(hl, tf.transpose(self.Wa))
        e = tf.exp(tf.matmul(self.va, tf.transpose(tf.tanh(self.ba + matmul))))
        #e = tf.exp(tf.reduce_sum(self.ba + tf.tanh(tf.matmul(hl, tf.transpose(self.Wa))), axis=-1, keepdims=True))

        alpha = e / tf.reduce_sum(e, axis=0, keepdims=True)

        alpha = tf.transpose(alpha)
        
        return alpha

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units) 
    
    def get_weighted_sum(self, inputs, alphas):
        # Calculate the weighted sum of input vectors based on attention scores
        weighted_sum = np.concatenate(inputs * alphas, axis=0)
        return weighted_sum

    def summary(self):
        print("Custom Attention Layer with {} units".format(self.units))
        #print("va shape:", self.va.shape)
        print("Trainable weights:", self.trainable_weights)
        print("Non-trainable weights:", self.non_trainable_weights)