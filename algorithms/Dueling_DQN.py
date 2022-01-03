import tensorflow as tf

#class DuelingDQN(object):
#    """Red neuronal fully connected."""
#    def __init__(self, input_dim, action_dim):
#        super(DuelingDQN, self).__init__()
#        self.input_dim = input_dim
#        self.action_dim = action_dim
#
#        self.model = self.nn_model()
#
#    def nn_model(self):
#        state_input = tf.keras.layers.Input((self.input_dim,))
#        dense1 = tf.keras.layers.Dense(75, activation="relu")(state_input)
#        dense2 = tf.keras.layers.Dense(75, activation="relu")(dense1)
#        V = tf.keras.layers.Dense(1, activation=None)(dense2)
#        A = tf.keras.layers.Dense(self.action_dim, dtype=tf.float32)(dense2)
#        #Q = V + (A - A.mean(dim=1, keepdim=True))
#        Q = tf.keras.layers.Add()([V, A])
#
#        model = tf.keras.models.Model(inputs=state_input, outputs=Q)
#
#        #_A = A - A.mean(dim=1, keepdim=True)
#        #Q = tf.keras.layers.Add()([V, _A])
#
#        return model

class DuelingDQN(tf.keras.Model):
    """Convolutional neural network for the Atari games."""
    def __init__(self, input_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(75, activation="relu")
        self.dense2 = tf.keras.layers.Dense(75, activation="relu")
        self.V = tf.keras.layers.Dense(1)
        self.A = tf.keras.layers.Dense(action_dim)

    @tf.function
    def call(self, states):
        """Forward pass of the neural network with some inputs."""
        x = self.dense1(states)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
        return Q