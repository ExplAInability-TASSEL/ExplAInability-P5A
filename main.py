from typing import Any
from py3.k_means import PixelValueGenerator, CustomKMeans
import numpy as np
from py3.CNN_model import Cplx_CustomCNN_1D
from py3.Attention_Layer import CustomAttentionLayer
from py3.classification import CustomClassifierModel
import tensorflow as tf
 
class MODEL():
    def __init__(self, num_classes=7, num_clusters=2):
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.kmeans = CustomKMeans(n_clusters=self.num_clusters) 
        self.input_shape = self.calculate_input_shape()
        self.enc = Cplx_CustomCNN_1D(input_shape_custom=self.input_shape, num_classes=self.num_classes)
        self.attn = CustomAttentionLayer(units=64)
        self.classifier = CustomClassifierModel(num_classes=self.num_classes)
        
    def calculate_input_shape(self):
        self.kmeans.fit(pixels)
        cluster_centers = self.kmeans.get_cluster_centers()
        print("cluster_centers shape ",cluster_centers.shape)
        return (cluster_centers[0].reshape((1,) + cluster_centers[0].shape)).shape[1:]
    
    def build(self, learning_rate=0.001):
        self.model = tf.keras.models.Sequential([
            self.enc,
            self.attn,
            self.classifier
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        
        
# Usage example, With an segment of 100 pixels, each with shape (73, 10)
pixels = np.random.random((100, 73, 10))

# we want to generate more realistic values for the pixels, so we help the clustering algorithm
generator = PixelValueGenerator(low_value_percentage=0.8)
generator.generate_values(pixels.shape)
pixels = generator.get_values()
print("pixels shape ",pixels.shape)

# object of the class CustomKMeans
n_clusters=2
custom_kmeans = CustomKMeans(n_clusters=n_clusters)

# Fit the K-Means model and retrieve cluster labels and centers
custom_kmeans.fit(pixels)

MODEL = MODEL(num_classes=7, num_clusters=2)

MODEL.build(learning_rate=0.001)

MODEL.train(pixels, custom_kmeans.get_cluster_labels(), epochs=10, batch_size=32)

 