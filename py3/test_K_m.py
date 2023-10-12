import numpy as np
from sklearn.cluster import KMeans

class PixelTransformer:
    def __init__(self):
        self.pixels = None

    def from_list(self, pixel_list):
        self.pixels = np.array(pixel_list)

    def to_2d(self, shape):
        if self.pixels is not None:
            self.pixels = self.pixels.reshape(shape)
        else:
            print("No pixel data available. Call 'from_list' method first.")

    def to_1d(self):
        if self.pixels is not None:
            self.pixels = self.pixels.reshape(-1)
        else:
            print("No pixel data available. Call 'from_list' method first.")

    def get_pixels(self):
        return self.pixels

class PixelClusterer:
    def __init__(self, pixel_data, n_clusters):
        self.pixel_data = pixel_data
        self.n_clusters = n_clusters
        self.cluster_labels = None

    def cluster(self):
        # Implement clustering logic here, such as K-Means or other algorithms
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.cluster_labels = kmeans.fit_predict(self.pixel_data)

    def get_cluster_labels(self):
        return self.cluster_labels
     
    # Create a NumPy array filled with 100 matrices of shape (73, 10) with random numbers
pixels = np.random.random((100, 73, 10))

# Define the percentage of pixels to be low values (e.g., 80%)
low_value_percentage = 0.8

# Calculate the number of low and high values to fill
num_low_values = int(np.prod(pixels.shape) * low_value_percentage)
num_high_values = np.prod(pixels.shape) - num_low_values

# Create an array of random values for low values (e.g., close to 0)
low_values = np.random.uniform(0, 0.2, num_low_values)

# Create an array of random values for high values (e.g., close to 1)
high_values = np.random.uniform(0.8, 1, num_high_values)

# Shuffle the low and high values
np.random.shuffle(low_values)
np.random.shuffle(high_values)

# Initialize an array with low values (e.g., 0)
values = np.zeros(np.prod(pixels.shape))

# Fill in the values with high values (e.g., 1) where needed
values[:num_low_values] = low_values
values[num_low_values:] = high_values

# Reshape the values array to the shape of pixels
values = values.reshape(pixels.shape)

# Apply the values to the pixels
pixels = values
print(pixels.shape)

# Create a K-Means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Reshape the pixels to (100, 730)
reshaped_pixels = pixels.reshape(100, -1)

# Fit the K-Means model to your data
kmeans.fit(reshaped_pixels)

# Get cluster labels for each pixel
cluster_labels = kmeans.labels_

print('Cluster labels shape: ', cluster_labels.shape)
print('Number of values in each cluster: ', np.bincount(cluster_labels))
print("k means 1 and 2 shape ",kmeans.cluster_centers_.shape)