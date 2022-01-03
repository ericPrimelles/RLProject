import tensorflow as tf

from algorithms.DQN import BaseDQN
from algorithms.Dueling_DQN import DuelingDQN

def choose_algorithm(method, input_dim, action_dim, lr, loss):
    # Metodo a utilizar
    if method == 'DQN':
        main_nn = BaseDQN(input_dim, action_dim)
        target_nn = BaseDQN(input_dim, action_dim)

    elif method == 'Dueling_DQN':
        main_nn = DuelingDQN(input_dim, action_dim)
        target_nn = DuelingDQN(input_dim, action_dim)

    # Optimizador Adam predeterminado
    optimizer = tf.keras.optimizers.Adam(lr)

    # Función de perdida
    if loss == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif loss == 'Huber_loss':
        loss_fn = tf.keras.losses.Huber()

    return main_nn, target_nn, optimizer, loss_fn