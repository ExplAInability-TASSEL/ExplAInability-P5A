from py3.k_means import PixelValueGenerator, CustomKMeans
from py3.CNN_model import CustomCNN
import numpy as np


# Usage example, With an segment of 100 pixels, each with shape (73, 10)
pixels = np.random.random((100, 73, 10))

# we want to generate more realistic values for the pixels, so we help the clustering algorithm
generator = PixelValueGenerator(low_value_percentage=0.8)
generator.generate_values(pixels.shape)
pixels = generator.get_values()

# object of the class CustomKMeans
custom_kmeans = CustomKMeans(n_clusters=2)

# Fit the K-Means model and retrieve cluster labels and centers
custom_kmeans.fit(pixels)

# results
cluster_labels = custom_kmeans.get_cluster_labels()
cluster_centers = custom_kmeans.get_cluster_centers()
print("cluster_labels shape ",cluster_labels.shape)
print("Number of pixels in each cluster: ", np.bincount(cluster_labels))
print("cluster_centers shape ",cluster_centers.shape)

# Appl




 
