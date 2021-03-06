from absl import app

from environment import Environment
from runner import Runner
from agents.Agent import Agent

_CONFIG = dict(
    episodes=200,
    actions=8,

    map_name='MoveToBeacon',
    screen_size=32,
    minimap_size=32,
    step_mul=50,
    visualize=False,    
    
    method='DQN',
    gamma=0.99, 
    epsilon=1.0,
    lr=1e-4, 
    loss='mse', 
    batch_size=64,
    epsilon_decrease=0.005,
    epsilon_min=0.05, 
    update_target=2000,
    num_episodes=5000, 
    max_memory=100000,

    train=True,
    load_path='./graphs/train_PGAgent_190226_1942'
)

def main(unused_argv):

    env = Environment(
                    map_name=_CONFIG['map_name'],
                    screen_size=_CONFIG['screen_size'],
                    minimap_size=_CONFIG['minimap_size'],
                    step_mul=_CONFIG['step_mul'],
                    visualize=_CONFIG['visualize']
                    )

    agent = Agent(
                    env=env, # Para obtener la cantidad de features units
                    action_dim=_CONFIG['actions'],
                    screen_size=_CONFIG['screen_size'],
                    method=_CONFIG['method'],
                    gamma=_CONFIG['gamma'], 
                    epsilon=_CONFIG['epsilon'], 
                    lr=_CONFIG['lr'], 
                    loss=_CONFIG['loss'], 
                    batch_size=_CONFIG['batch_size'],
                    epsilon_decrease=_CONFIG['epsilon_decrease'], 
                    epsilon_min=_CONFIG['epsilon_min'], 
                    update_target=_CONFIG['update_target'], 
                    num_episodes=_CONFIG['num_episodes'], 
                    max_memory=_CONFIG['max_memory']
                    )

    runner = Runner(
                    agent=agent,
                    env=env,
                    train=_CONFIG['train'],
                    load_path=_CONFIG['load_path']
                    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
