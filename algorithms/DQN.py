import tensorflow as tf

class BaseDQN(tf.keras.Model):
    """Red neuronal fully connected."""
    def __init__(self, num_actions):
        super(BaseDQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(75, activation="relu")
        self.dense2 = tf.keras.layers.Dense(75, activation="relu")
        self.out = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation

    def call(self, input_dims):
        """Forward pass."""
        x = self.dense1(input_dims)
        x = self.dense2(x)
        return self.out(x)
    
#    def advantage(self, state):
#        """For action selection."""
#        x = self.dense1(state)
#        x = self.dense2(x)
#        A = self.A(x)
#        return A