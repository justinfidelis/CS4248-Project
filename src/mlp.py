import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class MLP():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(413,)),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy())
        
        self.oh_encoder = OneHotEncoder(sparse_output=False)
        
    def train(self, train_x, train_y):
        train_y = train_y.to_numpy().reshape((-1, 1))
        one_hot_y = self.oh_encoder.fit_transform(train_y)
        self.model.fit(train_x.to_numpy(), one_hot_y, batch_size=32, epochs=40)

    def predict(self, X):
        one_hot_y = self.model.predict(X.to_numpy())
        return self.oh_encoder.inverse_transform(one_hot_y)