import numpy as np
from sklearn.cluster import KMeans

class CustomKMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.cluster_labels = None
        self.cluster_centers = None

    def fit(self, pixels):
        # Reshape to have a single feature vector per pixel, 73x10 to 730
        reshaped_pixels = pixels.reshape(len(pixels), -1)

        # Fit the K-Means model
        self.kmeans.fit(reshaped_pixels)
        self.cluster_labels = self.kmeans.labels_

        # We want the cluster centers and to reshape them back to 73x10 for the CNN
        self.cluster_centers = self.kmeans.cluster_centers_.reshape(self.n_clusters, 73, 10)

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_cluster_centers(self):
        return self.cluster_centers
    
    
class PixelValueGenerator:
    """This class generates pixel values for a given shape and percentage of low values
    This is useful for the first tests, to see if we can have unbalanced clusters
    """
    def __init__(self, low_value_percentage):
        self.low_value_percentage = low_value_percentage
        self.values = None

    def generate_values(self, shape):
         
        num_low_values = int(np.prod(shape) * self.low_value_percentage)
        num_high_values = np.prod(shape) - num_low_values

        low_values = np.random.uniform(0, 0.2, num_low_values)

        high_values = np.random.uniform(0.8, 1, num_high_values)

        np.random.shuffle(low_values)
        np.random.shuffle(high_values)

        self.values = np.zeros(np.prod(shape))
        self.values[:num_low_values] = low_values
        self.values[num_low_values:] = high_values

        self.values = self.values.reshape(shape)

    def get_values(self):
        return self.values

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


""" results
cluster_labels shape  (100,)
Number of pixels in each cluster:  [80 20]
cluster_centers shape  (2, 73, 10)
parfait !!!!
"""
