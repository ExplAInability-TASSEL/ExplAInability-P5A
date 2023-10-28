import numpy as np
from sklearn.cluster import KMeans

class CustomKMeans:
    """Custom K-Means Clustering.

    Args:
        n_clusters (int): The number of clusters.

    Attributes:
        n_clusters (int): The number of clusters in the model.
        kmeans (KMeans): The K-Means clustering model.
        cluster_labels (ndarray): Labels of the clustered data.
        cluster_centers (ndarray): Cluster centers.

    Methods:
        fit: Fit the K-Means model to input data.
        get_cluster_labels: Get the cluster labels of the input data.
        get_cluster_centers: Get the cluster centers.
    """
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
    
"""This class generates pixel values for a given shape and percentage of low values
    This is useful for the first tests, to see if we can have unbalanced clusters
"""
class PixelValueGenerator:
    """
    Generate Pixel Values with Low and High Intensities.

    Args:
        low_value_percentage (float): Percentage of low-intensity values.

    Attributes:
        low_value_percentage (float): The percentage of low-intensity values.
        values (ndarray): Generated pixel values.

    Methods:
        generate_values: Generate pixel values with specified low and high intensities.
        get_values: Get the generated pixel values
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
