import tensorflow as tf

class DuelingDQN(tf.keras.Model):
    """Red neuronal fully connected."""
    def __init__(self, input_dims, num_actions):
        super(DuelingDQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(input_dims, activation="relu")
        self.dense2 = tf.keras.layers.Dense(75, activation="relu")
        self.V = tf.keras.layers.Dense(1, activation=None)
        self.A = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation

    def call(self, input_dims):
        """Forward pass."""
        x = self.dense1(input_dims)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
    
    def advantage(self, state):
        """For action selection."""
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A