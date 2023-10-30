from py3.k_means import PixelValueGenerator, CustomKMeans
import numpy as np
from py3.CNN_model import Cplx_CustomCNN_1D
#hey

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
print(f'cluster_2 shape: {cluster_2.shape}')

# Reshape cluster_2 to match the input shape of CustomCNN
cluster_2 = cluster_2.reshape((1,) + cluster_2.shape)  # Add the batch dimension
 
# Create an instance of CustomCNN
custom_cnn = Cplx_CustomCNN_1D(input_shape=cluster_2.shape[1:], num_classes=7)

# Compile the model
custom_cnn.compile_model()
 
# Summarize the model
custom_cnn.summary()
 
#custom_cnn.model to make predictions
predictions2 = custom_cnn.model.predict(cluster_2)


# Print the predictions
print(predictions2)


 
