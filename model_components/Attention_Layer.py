from keras.layers import Layer
import tensorflow as tf
import numpy as np

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attn_size=512):
        super(AttentionLayer, self).__init__()
        self.attn_size = attn_size

    def build(self, input_shape):
        d = input_shape[-1]

        self.Wa = self.add_weight(name='Wa', shape=(d, self.attn_size),
                                  initializer='random_normal', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(self.attn_size,),
                                  initializer='random_normal', trainable=True)
        self.ua = self.add_weight(name='ua', shape=(self.attn_size,),
                                  initializer='random_normal', trainable=True)   

    def call(self, inputs):
        v = tf.tanh(tf.tensordot(inputs, self.Wa, axes=1) + self.ba)
        vu = tf.tensordot(v, self.ua, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output, alphas 