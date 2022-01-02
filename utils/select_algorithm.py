from algorithms.DQN import *
from algorithms.Dueling_DQN import DuelingDQN

def choose_algorithm(method, lr, loss):
    if method == 'DQN':
        main_nn = BaseDQN()
        target_nn = BaseDQN()

    elif method == 'Dueling_DQN':
        main_nn = DuelingDQN()
        target_nn = DuelingDQN()

    optimizer = tf.keras.optimizers.Adam(lr)

    mse = tf.keras.losses.MeanSquaredError()
    #main_nn.compile(loss=loss, optimizer=optimizer)
    #target_nn.compile(loss=loss, optimizer=optimizer)

    return main_nn, target_nn, optimizer, mse