from typing import Any
from py3.k_means import PixelValueGenerator, CustomKMeans
import numpy as np
from py3.CNN_model import Cplx_CustomCNN_1D
from py3.Attention_Layer import CustomAttentionLayer
from py3.classification import CustomClassifierModel
import tensorflow as tf

import re
import numpy as np
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    
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
for _ in range(100):
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
# add random labels to the clusters (between 0 and 7)
labels = np.random.randint(0, 8, clusters.shape[0])
print("labels shape:", labels.shape) 

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

train_data, test_data, train_labels, test_labels = train_test_split(clusters, labels, test_size=0.2, random_state=42)
train_labels = to_categorical(train_labels, num_classes=8)
test_labels = to_categorical(test_labels, num_classes=8)
 
print(Color.GREEN + "train_data shape:", str(train_data.shape) + Color.END)
print(Color.GREEN + "test_data shape:", str(test_data.shape) + Color.END)

tf.config.run_functions_eagerly(True) # SAVE MY LIFE !!!!!

class Custom_Model(tf.keras.Model):
    
    def __init__(self):
        super(Custom_Model, self).__init__()
        self.enc = Cplx_CustomCNN_1D()
        self.attn = CustomAttentionLayer(units=64)
        self.classifier = CustomClassifierModel(num_classes=8)
    
    def call(self, inputs):
        
        # shape of the input 
        print(Color.GREEN + "inputs shape:", str(tf.shape(inputs))+ Color.END)
        print(Color.GREEN + "inputs shape:", str(np.shape(inputs))+ Color.END)
        
        intermediate = tf.map_fn(self.enc, inputs, dtype=tf.float32)
        print(Color.BLUE +"Intermediate shape after map_fn:", str(intermediate.shape) + Color.END)
        attention_outputs = tf.map_fn(lambda x: self.attn(self.enc(x)), inputs, dtype=tf.float32)
        print(Color.BLUE +"attention_outputs shape after map_fn:", str(attention_outputs.shape) + Color.END)


        emb = [tf.multiply(alpha, tensor) for alpha, tensor in zip(attention_outputs, intermediate)]
        print(Color.YELLOW +"emb shape after map_fn:", str(np.array(emb).shape) + Color.END)

        emb = tf.reduce_sum(emb, axis=1)
        print(Color.YELLOW +"result shape after reduce_sum:", str(emb.shape) + Color.END)
        
        return self.classifier(emb)
    
    def summary(self):
        self.enc.summary()
        self.attn.summary()
        #self.classifier.summary()
        
         
model = Custom_Model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_data, train_labels, epochs=10, batch_size=32)
 
  
    
    
 