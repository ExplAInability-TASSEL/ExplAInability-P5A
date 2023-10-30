from keras.layers import Layer
import tensorflow as tf

class CustomAttentionLayer(Layer):
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
        super(CustomAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        hl = inputs 

        e = tf.exp(tf.reduce_sum(self.ba + tf.tanh(tf.matmul(hl, tf.transpose(self.Wa))), axis=-1, keepdims=True))

        alpha = e / tf.reduce_sum(e, axis=0, keepdims=True)

        
        return alpha

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units) 
