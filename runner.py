import datetime
import os
import tensorflow as tf
#from learning.Q_Learning import Q_Learning

import numpy as np

class Runner(object):
    def __init__(self, agent, env, train, load_path):
        self.agent = agent
        self.env = env
        self.train = train  # True: entrenar agente, False: se carga agente entrenado

        self.score = 0  
        self.episode = 1 
        self.last_10_ep_rewards = []

        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        self.writer = tf.summary.create_file_writer(self.path)

        # Se carga modelo entrenado
        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        """Guarda los valores obtenidos por episodio."""
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='Score per Episode', simple_value=self.score)]),
            self.episode
        )
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)

            try:
                self.agent.update_target_model() # No se que hace
            except AttributeError:
                ...

        self.episode += 1
        
    def run(self, episodes):
        """Entrenamiento del agente en la cantidad de episodios dados."""
        while self.episode <= episodes:
            obs = self.env.reset()
            obs = self.agent.select_army(obs)

            self.score = 0
            done = False

            while True:
                if obs.last():
                    break
                state, pos_marine = self.agent.state_marine(obs) 
                action = self.agent.step(state, pos_marine)
                obs = self.env.step(action) # next_state, reward, done, info = env.step(action)
                
                next_state, pos_marine = self.agent.state_marine(obs) 
                reward = obs.reward
                done = reward > 0

                self.score += reward

                # Guarda las experiencias del agente en cada paso de tiempo.
                self.agent.buffer.add(state, action, reward, next_state, done)
                state = next_state
                self.agent.cur_frame += 1

                # Copia los pesos de la red principal hacia la red target.
                if self.agent.cur_frame % self.agent.update_target == 0:
                    self.agent.target_nn.load_state_dict(self.agent.main_nn.state_dict())
            
                # Entrenamiento de la red neuronal.
                if len(self.agent.buffer) > self.agent.batch_size:
                    if not self.agent.buffer.isPER():
                        states, actions, rewards, next_states, dones = self.agent.buffer.sample(self.agent.batch_size)
                        loss = self.agent.train_step(states, actions, rewards, next_states, dones)

                    else:
                        tree_idx, states, actions, rewards, next_states, dones = self.MEMORY.sample(self.batch_size)
                        loss = self.agent.train_step(states, actions, rewards, next_states, dones)
                        self.agent.buffer.batch_update(tree_idx, loss)


            # Decrecimiento del epsilon.
            if self.episode < self.agent.num_episodes:
                self.agent.decrease_epsilon()

            # Guarda recompensas de los ultimos 10 episodios.
            if len(self.last_10_ep_rewards) == 10:
                self.last_10_ep_rewards = self.last_10_ep_rewards[1:]
            self.last_10_ep_rewards.append(self.score)

            # Guarda recompensa y explota el conocimiento del agente cada 10 episodios.
            if self.episode % 10 == 0:
                #aux_reward = agent.explotation(iteraciones)
                mean_rewards = np.mean(self.last_100_ep_rewards)

                print(f'Episode {self.episode}/{self.agent.num_episodes}, Epsilon: {self.agent.epsilon:.3f}, '\
                    f'Reward in last 100 episodes: {mean_rewards:.2f}')
                
                #episodes.append(episode)
                #eps_history.append(agent.epsilon)
                #prom_rewards_greedy.append(aux_reward)
                #last_100_mean_rewards.append(mean_rewards) 

            self.summarize()
