from typing import Any
from py3.k_means import PixelValueGenerator, CustomKMeans
import numpy as np
from py3.CNN_model import Cplx_CustomCNN_1D
from py3.Attention_Layer import CustomAttentionLayer
from py3.classification import CustomClassifierModel

# Usage example, With an segment of 100 pixels, each with shape (73, 10)
pixels = np.random.random((100, 73, 10))

# we want to generate more realistic values for the pixels, so we help the clustering algorithm
generator = PixelValueGenerator(low_value_percentage=0.8)
generator.generate_values(pixels.shape)
pixels = generator.get_values()

# object of the class CustomKMeans
n_clusters=2
custom_kmeans = CustomKMeans(n_clusters=n_clusters)

# Fit the K-Means model and retrieve cluster labels and centers
custom_kmeans.fit(pixels)
print("hey")
# results
cluster_labels = custom_kmeans.get_cluster_labels()
cluster_centers = custom_kmeans.get_cluster_centers()
print("cluster_labels shape ",cluster_labels.shape)
print("Number of pixels in each cluster: ", np.bincount(cluster_labels))
print("cluster_centers shape ",cluster_centers.shape)

cluster_1 = cluster_centers[0]
print(f'cluster_1 shape: {cluster_1.shape}')
# Reshape cluster_1 to match the input shape of CustomCNN
cluster_1 = cluster_1.reshape((1,) + cluster_1.shape)  # Add the batch dimension

cluster_2 = cluster_centers[1]
input_shape = cluster_2.shape  # Use the shape directly without adding a batch dimension

custom_cnn = Cplx_CustomCNN_1D(input_shape=input_shape, num_classes=7)

# Compile the model
custom_cnn.compile_model()
 
# Summarize the model
custom_cnn.summary()
 
#custom_cnn.model to make predictions
predictions2 = custom_cnn.model.predict(cluster_2)

# prediction 1 
predictions1 = custom_cnn.model.predict(cluster_1)

# Print the predictions
print(predictions2)
print(predictions1)
print(f"predictions1 shape: {predictions1.shape}")
print(f"predictions2 shape: {predictions2.shape}")

# join predictions as in cluster_centers 
predictions = np.concatenate((predictions1, predictions2), axis=0)

print(predictions)
print(f"predictions shape: {predictions.shape}")

# Create an instance of CustomAttentionLayer
attn_layer = CustomAttentionLayer(units=1)

# Get the attention weights
attn_weights = attn_layer(predictions)
 
 
# Print the attention weights
print(attn_weights)
print(f"attn_weights shape: {attn_weights.shape}")
 
h = attn_layer.get_weighted_sum(predictions, attn_weights)
print(h)
print(f"h shape: {h.shape}")

classifier = CustomClassifierModel(num_classes=8)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
 
# Make predictions
predictions = classifier.predict(h)
print(predictions)
print(f"predictions shape: {predictions.shape}")

 
class MODEL():
    def __init__(self, num_classes=7, num_clusters=2):
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.input_shape = self.calculate_input_shape()
        self.cluster_centers = CustomKMeans(n_clusters=self.num_clusters) 
        self.enc = Cplx_CustomCNN_1D(input_shape=self.input_shape, num_classes=self.num_classes)
        self.attn = CustomAttentionLayer(units=1)
        self.classifier = CustomClassifierModel(num_classes=self.num_classes)
        
    # encode the cluster with Cplx_CustomCNN_1D and return the "predictions"
    def calculate_input_shape(self):
        cluster_centers = self.cluster_centers.get_cluster_centers()
        return cluster_centers[0].shape[1:]
        
        
 
