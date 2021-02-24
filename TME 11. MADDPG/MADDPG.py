import matplotlib

import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from DDPG import *
from utils import *


def make_env(scenario_name, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world



class MADDPG:
    def __init__(self,state_dim,
                 action_dim,
                 lr_actor,
                 lr_critic,
                 lr_decay,
                 replay_buff_size,
                 gamma,
                 batch_size,
                 random_seed, 
                 soft_update_tau):
    
        
        super(MADDPG, self).__init__()
        self.adversarial_agents = []
        for i in range(len(state_dim)):
          agent = DDPGAgent(state_dim[i],
                 action_dim,
                 )
          self.adversarial_agents.append(agent)

        
    def get_actors(self):

        actors = [ddpg_agent.actor_local for ddpg_agent in self.adversarial_agents]
        return actors

    def get_target_actors(self):
 
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.adversarial_agents]
        return target_actors

    def act(self, states_all_agents, add_noise=False):

        actions = [agent.act(state, add_noise) for agent, state in zip(self.adversarial_agents, states_all_agents)]
        return np.stack(actions, axis=0)

    def update(self, *experiences):

        states, actions, rewards, next_states, dones = experiences
        for agent_idx, agent in enumerate(self.adversarial_agents):
            state = states[agent_idx]
            action = actions[agent_idx]
            reward = rewards[agent_idx]
            next_state = next_states[agent_idx]
            done = dones[agent_idx]
            agent.update_model(state, action, reward, next_state, done)
            




            

if __name__ == '__main__':
 # prepare environment
    env,scenario,world = make_env('simple_spread')

    
    num_agents = len(env.agents)
    print('Number of agents:', num_agents)
    


    num_episodes = 500
    
    state_size = env.observation_space
    action_size = 2

    o=env.reset()
    state_size=[len(i) for i in o]

    agent = MADDPG(state_size, 
                   action_size,
                   lr_actor = 1e-5,
                   lr_critic = 1e-4,
                   lr_decay = .995,
                   replay_buff_size = int(1e6),
                   gamma = .95,
                   batch_size = 64,
                   random_seed = 999,
                   soft_update_tau = 1e-3
                 )
    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    threshold_init = 20
    noise_t = 1.0
    noise_decay = .995
    worsen_tolerance = threshold_init  # for early-stopping training if consistently worsen for # episodes
    reward = 0
    num_agents = 3
    for i_episode in range(1, num_episodes+1):
        obs = env.reset()   # reset the environment        
        scores = np.zeros(num_agents)                        # initialize score array
        dones = [False]*num_agents

        for i in range(100):
            if np.any(dones):
                break
            actions = agent.act(obs,noise_t)              # select an action
            next_states , rewards, dones, _ = env.step(actions)         # send the action to the environment     
          
            agent.update(obs, actions, rewards, next_states, dones)
            noise_t *= noise_decay
            scores += rewards       
            obs = next_states                         
            # update scores 
        episode_score = np.max(scores)
        total_rewards.append(episode_score)
        print("Episode {} Score: {:.4f}".format(i_episode, episode_score))
                    
    draw(total_rewards,"./training_score_plot.png", "Training Scores (Per Episode)")
    #draw(avg_scores,"./training_100avgscore_plot.png", "Training Scores (Average of Latest 100 Episodes)", ylabel="Avg. Score")
    env.close()
