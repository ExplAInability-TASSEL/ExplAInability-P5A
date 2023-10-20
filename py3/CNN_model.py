from tensorflow import keras

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

        # Add other Conv1D layers and other components as needed
        
        model.add(keras.layers.GlobalAveragePooling1D())  # Global Average Pooling for 1D data
        
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()
        
class Cplx_CustomCNN_1D:
    def __init__(self, input_shape=(730, 10), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()#keras.model
        
        model.add(keras.layers.Conv1D(256, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv1D(256, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv1D(256, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        #model.add(keras.layers.MaxPooling1D(pool_size=2))  # Utilisation de MaxPooling au lieu de GlobalAveragePooling
        
        model.add(keras.layers.Conv1D(512, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv1D(512, kernel_size=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv1D(512, kernel_size=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.GlobalAveragePooling1D()) 
        
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()
