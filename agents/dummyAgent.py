from pysc2.agents import base_agent
from  pysc2.lib import actions
from  pysc2.agents.random_agent import RandomAgent

class Dummy(base_agent.BaseAgent):
    def step(self, obs):
        super(Dummy, self).step(obs)
        return actions.FUNCTIONS.no_op()