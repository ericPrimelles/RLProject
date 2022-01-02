from pysc2.env import sc2_env
from pysc2.env import available_actions_printer

from pysc2.lib import features

map_name='MoveToBeacon'
screen_size=32 
minimap_size=32 
step_mul=8 
visualize=False

with sc2_env.SC2Env(
                    map_name=map_name,
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=screen_size, minimap=minimap_size),
                        use_feature_units=True
                    ),
                    step_mul=step_mul,
                    visualize=visualize
                    ) as env:

                    env = available_actions_printer.AvailableActionsPrinter(env)
                    print(f"env: {env}")
