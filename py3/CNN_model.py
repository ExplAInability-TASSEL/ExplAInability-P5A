import tensorflow as tf
from keras import layers

class Cplx_CustomCNN_1D(layers.Layer):
    """Custom 1D Convolutional Neural Network.

    Args:
        input_shape (tuple): Input shape (default: (730, 10)).
        num_classes (int): Number of classes (default: 7).

    Attributes:
        input_shape (tuple): The input shape of the model.
        num_classes (int): The number of target classes.
        model (keras.Sequential): The built CNN model.
    """
    def __init__(self, input_shape_custom=(73, 10), num_classes=7):
        super(Cplx_CustomCNN_1D, self).__init__()
        self.input_shape_custom = input_shape_custom
        self.num_classes = num_classes

        self.model = tf.keras.Sequential()

        self.model.add(layers.Conv1D(256, kernel_size=3, activation='relu', input_shape=self.input_shape_custom))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv1D(512, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv1D(512, kernel_size=1, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv1D(512, kernel_size=1, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.GlobalAveragePooling1D())

    def call(self, inputs):
        return self.model(inputs)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

class CustomCNN:
    def __init__(self, input_shape=(73, 10), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu', input_shape=self.input_shape))#
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(256, (3, 3), strides=2, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (3, 3), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (1, 1), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(512, (1, 1), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.GlobalAveragePooling2D())
        
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()
        
class CustomCNN_1D:
    def __init__(self, input_shape=(73, 10), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        
        model.add(keras.layers.Conv1D(256, kernel_size=3, strides=1, activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        
        model.add(keras.layers.GlobalAveragePooling1D()) 
        
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()
   