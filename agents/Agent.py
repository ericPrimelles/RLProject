import tensorflow as tf
import numpy as np
import random

from agents.AbstractAgent import AbstractAgent

from minigames.utils import state_of_marine, move_to_position
from utils.select_algorithm import choose_algorithm
from utils.replay_buffer import UniformBuffer

class Agent(AbstractAgent):
    def __init__(self, env, action_dim, screen_size, method, gamma=0.99, epsilon=1.0, lr=1e-4, loss='mse', batch_size=32,
                epsilon_decrease=0.001, epsilon_min=0.05, update_target=2000, num_episodes=5000, max_memory=100000):
        super(Agent, self).__init__(screen_size)

        obs = env.reset()
        screen = np.array(obs.observation['feature_screen'])
        screen = np.reshape(screen, (screen.shape[1], screen.shape[2], screen.shape[0]))
        screen = tf.convert_to_tensor(screen, dtype=tf.float64)
        self.input_dim = screen.shape
        self.action_dim = action_dim

        # Hiperparametros
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.loss = loss
        self.batch_size = batch_size

        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.update_target = update_target

        self.num_episodes = num_episodes
        self.memory_size = max_memory
        self.cur_frame = 0

        # Red principal y target.
        self.main_nn, self.target_nn, \
        self.optimizer, self.loss_fn = choose_algorithm(method, self.input_dim, self.action_dim,
                                                        self.lr, self.loss)
            
        # Buffer donde se almacenaran las experiencias del agente.

        self.buffer = UniformBuffer(self.memory_size)

    def step(self, state, pos_marine):
        action = self.select_epsilon_greedy_action(state)

        # Dependiendo de la acción se mueve ciertas coordenadas
        destination = move_to_position(action, self.screen_size)

        return action, self._MOVE_SCREEN("now", self._xy_offset(pos_marine, destination[0], destination[1]))

    def state_marine(self, obs):
        # Representación del beacon y marino
        beacon = self.get_beacon(obs)
        marine = self.get_marine(obs)

        dist = np.hypot((beacon.x - marine.x), (beacon.y - marine.y))

        screen = np.array(obs.observation['feature_screen'])
        screen = np.reshape(screen, (screen.shape[1], screen.shape[2], screen.shape[0]))
        state = tf.convert_to_tensor(screen, dtype=tf.float64)
        pos_marine = self.get_unit_pos(marine)

        return state, pos_marine, dist

    def select_army(self, obs):
        # La primera acción selecciona a los army
        if obs.first():
            return self._SELECT_ARMY

    def select_epsilon_greedy_action(self, state, aux_epsilon=1.0):
        """Realiza una acción aleatoria con prob. épsilon; de lo contrario, realiza la mejor acción."""
        result = tf.random.uniform((1,))
        if result < self.epsilon and result < aux_epsilon:
            return random.choice(range(self.action_dim)) #env.action_space.sample() # Acción aleatoria.
        else:
            state = np.reshape(state, (1, tf.shape(state)[0].numpy(), tf.shape(state)[1].numpy(), tf.shape(state)[2].numpy()))
            return tf.argmax(self.main_nn.predict(state)[0]).numpy() # Acción greddy.

    def train_step(self, states, actions, rewards, next_states, dones):
        """Realiza una iteración de entrenamiento en un batch de datos."""

        next_qs = self.target_nn.predict(next_states, batch_size=self.batch_size)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.gamma * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.loss_fn(target, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        return loss

    def decrease_epsilon(self):
        """Decrecimiento del epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrease
        else:
            self.epsilon = self.epsilon_min


    def copy_weights(self, Copy_from, Copy_to):
        """
        Function to copy weights of a model to other
        """
        variables2 = Copy_from.trainable_variables
        variables1 = Copy_to.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
    def save_model(self, filename):
        self.learner.save_q_table(filename + '/model.pkl')

    def load_model(self, filename):
        self.learner.load_model(filename + '/model.pkl')
