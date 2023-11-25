from keras.layers import Layer, BatchNormalization, ReLU, Dense
import tensorflow as tf

class CustomFullyConnectedLayer(Layer):
    def __init__(self, units, **kwargs):
        super(CustomFullyConnectedLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.fc = Dense(self.units, activation=None)
        self.bn = BatchNormalization()
        self.relu = ReLU()
        super(CustomFullyConnectedLayer, self).build(input_shape)

    def call(self, inputs):
        x = self.fc(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    

class CustomClassifierModel(tf.keras.Model):
    """
    Custom Classifier Model.

    This model is designed for multi-class classification. It uses two fully connected layers with Batch Normalization
    and ReLU activation, followed by an output layer with softmax activation.

    Args:
        num_classes (int): The number of output classes.
        fc_units (int, optional): The number of units in the fully connected layers. Defaults to 512.

    """
    def __init__(self, num_classes, fc_units=512, **kwargs):
        super(CustomClassifierModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.fc_units = fc_units

        # Define Fully Connected Layers
        self.fc1 = CustomFullyConnectedLayer(units=fc_units)
        self.fc2 = CustomFullyConnectedLayer(units=fc_units)

        # Output Layer
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        output = self.output_layer(x)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)
    
    

