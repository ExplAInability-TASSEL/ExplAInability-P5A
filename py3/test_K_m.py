import numpy as np
from sklearn.cluster import KMeans

class PixelTransformer:
    def __init__(self):
        self.pixels = None

    def from_list(self, pixel_list: list):
        self.pixels = np.array(pixel_list, dtype=np.uint8)

    def to_2d(self, shape: tuple):
        if self.pixels is not None:
            self.pixels = self.pixels.reshape(shape)
        else:
            print("No pixel data available. Call 'from_list' method first.")

    def to_1d(self):
        if self.pixels is not None:
            self.pixels = self.pixels.reshape(-1)
        else:
            print("No pixel data available. Call 'from_list' method first.")

    def get_pixels(self) -> np.ndarray:
        return self.pixels

class PixelClusterer:
    def __init__(self, pixel_data, n_clusters):
        self.pixel_data = pixel_data
        self.n_clusters = n_clusters
        self.cluster_labels = None

    def cluster(self):
        """
        Fit the K-Means model and retrieve cluster labels
        """
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_labels = kmeans.fit_predict(self.pixel_data)

    def get_cluster_labels(self):
        return self.cluster_labels