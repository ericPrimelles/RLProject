import tensorflow as tf
from keras. layers import InputLayer, Conv2D, MaxPool2D, Flatten

#class BaseDQN(tf.keras.Model):
#    """Red neuronal fully connected."""
#    def __init__(self, input_dim, action_dim):
#        super(BaseDQN, self).__init__()
#        self.input_dim = input_dim
#        self.action_dim = action_dim
#
#        self.model = self.nn_model()
#
#    def nn_model(self):
#        state_input = tf.keras.layers.Input((self.input_dim,))
#        dense1 = tf.keras.layers.Dense(75, activation="relu")(state_input)
#        dense2 = tf.keras.layers.Dense(75, activation="relu")(dense1)
#        out = tf.keras.layers.Dense(self.action_dim)(dense2)
#
#        model = tf.keras.models.Model(inputs=state_input, outputs=out)
#
#        return model

class BaseDQN(tf.keras.Model):
    """Dense neural network class."""
    def __init__(self, input_dim, action_dim):
        super(BaseDQN, self).__init__()
        #self.state_input = tf.keras.layers.Input((input_dim,))
        self.conv1 = Conv2D(32, input_shape=input_dim, activation='relu', kernel_size=(3, 3))
        self.mp1 = MaxPool2D()
        self.conv2 = Conv2D(64, activation='relu', kernel_size=(3, 3))
        self.mp2 = MaxPool2D()
        self.fltt = Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(action_dim, activation='relu')

    def call(self, states):
        """Forward pass."""
        x = self.conv1(states)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.fltt(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)