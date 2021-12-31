from absl import app
from env import Env
from runner import Runner

from agents.dummyAgent import Dummy

def main(unused_argv):
    _CONFIG = dict(
        episodes=200,
        screen_size=128,
        minimap_size=128,
        visualize=False,
        train=False,
        agent=Dummy,
        load_path='./graphs/train_PGAgent_190226_1942'
    )

    agent = _CONFIG['agent'](
    )

    env = Env(
        map_name= 'CollectMineralShards',
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )
    #agent.setup(env.sc2_env.observation_spec(), env.sc2_env.action_spec())
    #agent.reset()
    print(type(env.sc2_env.action_spec()))

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])

if __name__ == '__main__':
    app.run(main)