from abc import ABCMeta, abstractmethod
from pysc2.lib import actions, features
import numpy as np

class AbstractAgent: #(metaclass=ABCMeta):

    """Sc2 Actions"""
    _MOVE_SCREEN = actions.FUNCTIONS.Move_screen
    _NO_OP = actions.FUNCTIONS.no_op()
    _SELECT_ARMY = actions.FUNCTIONS.select_army("select")

    def __init__(self, screen_size):
        self.screen_size = screen_size

    # Metodos implementados en las subclases
    @abstractmethod
    def step(self, obs, pos_marine): ...

    @abstractmethod
    def state_marine(self, obs): ...

    @abstractmethod
    def select_army(self, obs): ...

    @abstractmethod
    def select_epsilon_greedy_action(self, state, aux_epsilon=1.0): ...

    @abstractmethod
    def train_step(self, states, actions, rewards, next_states, dones): ...

    @abstractmethod
    def decrease_epsilon(self): ...

    @abstractmethod
    def save_model(self, filename): ...

    @abstractmethod
    def load_model(self, filename): ...

    # Metodos de la clase Abstract
    def get_beacon(self, obs):
        """Devuelve la representación de la baliza."""
        beacon = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.NEUTRAL)
        return beacon

    def get_marine(self, obs):
        """Devuelve la representación del marino."""
        marine = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == features.PlayerRelative.SELF)
        return marine

    def get_unit_pos(self, unit):
        """Devuelve las posición (x,y) de un objeto unitario."""
        return np.array([unit.x, unit.y])

    def _xy_offset(self, start, offset_x, offset_y):
        """Devuelve la nueva posición (x', y') desde la posición (x, y) del objeto."""
        destination = start + np.array([offset_x, offset_y])

        # Considera no establecer el punto más allá del borde de la pantalla
        if destination[0] < 0:
            destination[0] = 0
        elif destination[0] >= self.screen_size:
            destination[0] = self.screen_size - 1

        if destination[1] < 0:
            destination[1] = 0
        elif destination[1] >= self.screen_size:
            destination[1] = self.screen_size - 1

        return destination
