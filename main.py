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
        self.kmeans.fit(reshaped_array) # AAA CHANGER !!!! 
        cluster_centers = self.kmeans.get_cluster_centers()
        print("cluster_centers shape ",cluster_centers.shape)
        return (cluster_centers[0].reshape((1,) + cluster_centers[0].shape)).shape[1:]
    
    def build(self, learning_rate=0.0001):
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
        
"""
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
"""

import re
import numpy as np

vectors = []

with open('1_segment_test.txt', 'r') as file:
    for line in file:
        values = re.findall(r'\[([^]]*)\]', line)
        for sublist in values:
            vector = [float(value) for value in sublist.split(',')]
            vectors.append(vector)

numpy_array = np.array(vectors)

print("Shape of the numpy...:", numpy_array.shape)
reshaped_array = numpy_array.reshape(62, 73, 10)
shuffled_arrays = []
for _ in range(3):
    np.random.shuffle(reshaped_array)
    shuffled_arrays.append(reshaped_array.copy())


# Create the data variable
data = np.array(shuffled_arrays)
print("data shape:", data.shape)

n_clusters=2
custom_kmeans = CustomKMeans(n_clusters=n_clusters)

clustered_data = []
#
for i in range(data.shape[0]):
    custom_kmeans.fit(data[i])
    clustered_data.append(custom_kmeans.get_cluster_centers())
     
clusters = np.array(clustered_data)
print("clusters shape:", clusters.shape)
   
 



class Custom_Model(tf.keras.Model):
    
    def __init__(self):
        super(Custom_Model, self).__init__()
        self.enc = Cplx_CustomCNN_1D()
        self.attn = CustomAttentionLayer(units=64)
        self.classifier = CustomClassifierModel(num_classes=8)
    
    def call(self, inputs):
        intermediate = []
        for el in inputs:
            intermediate.append(self.enc(el))
            print("intermediate shape ", np.array(intermediate).shape)
        # Apply attention individually
        attention_outputs = [self.attn(x) for x in intermediate]
        print("attention_outputs shape:", np.array(attention_outputs).shape)
        print("attention_outputs:", attention_outputs)
        emb = [alpha * tensor for alpha, tensor in zip(attention_outputs, intermediate)]
        emb = np.array(emb)
        print("emb shape before reshape:", np.array(emb).shape)
        emb = emb.reshape(emb.shape[0], -1)
        print("emb shape after reshape:", np.array(emb).shape)
# tf.stack 
        return self.classifier(emb)
    
    def summary(self):
        self.enc.summary()
        self.attn.summary()
        self.classifier.summary()
        
    def compile(self, optimizer, loss, metrics=None):
        super(Custom_Model, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    """def fit(self, x, y, epochs, batch_size):
        self.call(x)
        self.classifier.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.classifier.fit(x, y, epochs=epochs, batch_size=batch_size)"""
         
model = Custom_Model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(clusters, clusters, epochs=10, batch_size=3)
 
  
    
    
 