#from learning.Q_Learning import Q_Learning
#from pysc2.agents import base_agent

import tensorflow as tf
import random

from agent import AbstractAgent

from minigames.utils import state_of_marine, move_to_position
from utils.select_algorithm import choose_algorithm
from utils.replay_buffer import UniformBuffer


class Agent(AbstractAgent):

    def __init__(self, actions, screen_size, method, gamma=0.99, epsilon=1.0, lr=1e-4, loss='mse', batch_size=32,
                epsilon_decrease=0.001, epsilon_min=0.05, update_target=2000, num_episodes=5000, max_memory=100000):
        
        super(Agent, self).__init__(screen_size)
        self.actions = actions

        # Hiperparametros
        #self.input_dims = input_dims
        #self.num_actions = num_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size

        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.update_target = update_target

        self.num_episodes = num_episodes
        self.memory_size = max_memory
        self.cur_frame = 0

        # Red principal y target.
        self.main_nn, self.target_nn, \
        self.optimizer, self.loss_fn = choose_algorithm(method, self.lr, loss)
            
        # Buffer donde se almacenaran las experiencias del agente.
        self.buffer = UniformBuffer(self.memory_size)

    def step(self, state, pos_marine):
        action = self.select_epsilon_greedy_action(state)

        # Dependiendo de la acción se mueve ciertas coordenadas
        dest = move_to_position(action, self.screen_size)

        return self._MOVE_SCREEN("now", self._xy_offset(pos_marine, dest[0], dest[1]))

    def state_marine(self, obs):
        # Representación del beacon y marino
        beacon = self.get_beacon(obs)
        marine = self.get_marine(obs)

        # Estado en el que se encuentra el marino con respecto al beacon
        state = state_of_marine(marine, beacon, self.screen_size, 10)
        pos_marine = self.get_unit_pos(marine)

        return str(state), pos_marine

    def select_army(self, obs):
        # La primera acción selecciona a los army
        if obs.first():
            return self._SELECT_ARMY

    def select_epsilon_greedy_action(self, state, aux_epsilon=1.0):
        """Realiza una acción aleatoria con prob. épsilon; de lo contrario, realiza la mejor acción."""
        result = tf.random.uniform((1,))
        if result < self.epsilon and result < aux_epsilon:
            return random.choice(self.actions) #env.action_space.sample() # Acción aleatoria.
        else:
            return tf.argmax(self.main_nn(state)[0]).numpy() # Acción greddy.

    def train_step(self, states, actions, rewards, next_states, dones):
        """Realiza una iteración de entrenamiento en un batch de datos."""
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.gamma * max_next_qs
        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, self.actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(target, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        return loss

    def decrease_epsilon(self):
        """Decrecimiento del epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrease
        else:
            self.epsilon = self.epsilon_min

    def save_model(self, filename):
        self.learner.save_q_table(filename + '/model.pkl')

    def load_model(self, filename):
        self.learner.load_model(filename + '/model.pkl')
