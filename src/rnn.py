import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

class RNN():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(),
            tf.keras.layers.SimpleRNN(),
            tf.keras.layers.Dense(),
            tf.keras.layers.Softmax(),
        ])