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
        self.va = self.add_weight(name='va', shape=(self.units,),
                                 initializer='uniform', trainable=True)
        super(CustomAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        hl = inputs 

        e = tf.exp(self.va@tf.tanh(self.ba + tf.matmul(hl, tf.transpose(self.Wa))))

        alpha = e / tf.reduce_sum(e, axis=0, keepdims=True)

        
        return alpha

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units) 
    
    def get_weighted_sum(self, inputs, alphas):
        # Calculate the weighted sum of input vectors based on attention scores
        weighted_sum = np.concatenate(inputs * alphas, axis=0)
        return weighted_sum
