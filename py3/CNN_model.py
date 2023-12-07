import tensorflow as tf
from keras import layers

class Cplx_CustomCNN_1D(layers.Layer):
    """Custom 1D Convolutional Neural Network.

    Args:
        input_shape (tuple): Input shape (default: (73, 10)).
        num_classes (int): Number of classes (default: 8).

    Attributes:
        input_shape (tuple): The input shape of the model.
        num_classes (int): The number of target classes.
        model (keras.Sequential): The built CNN model.
    """
    def __init__(self, input_shape_custom=(73, 10), num_classes=8):
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
        
        
from efficientnet.keras import EfficientNetB0

class New_CustomCNN1D(tf.keras.Model):
    def __init__(self, input_shape=(73, 10), num_classes=8):
        super(New_CustomCNN1D, self).__init__()

        self.model = tf.keras.Sequential()

        self.model.add(layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Conv1D(128, kernel_size=2, activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Conv1D(256, kernel_size=2, activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.GlobalAveragePooling1D())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.GlobalAveragePooling1D())
    def call(self, inputs):
        return self.model(inputs)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()
