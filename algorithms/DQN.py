import tensorflow as tf

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
        self.dense1 = tf.keras.layers.Dense(75, activation="relu")
        self.dense2 = tf.keras.layers.Dense(75, activation="relu")
        self.dense3 = tf.keras.layers.Dense(action_dim, dtype=tf.float32) # No activation

    def call(self, state):
        """Forward pass."""
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)