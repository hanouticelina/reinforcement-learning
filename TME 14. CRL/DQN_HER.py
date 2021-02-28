import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.nn.functional as F
from memory import *
from utils import *
from gridworld import *
import math
import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)




class Q(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q, self).__init__()
        self.num_states = num_inputs
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, num_actions)
        )
        
    def forward(self, state):
        return self.layers(state)
    
    
class DQNAgent(nn.Module):
    def __init__(self, num_inputs, num_actions,buffer_size=1000000,gamma=0.99,device=device):
        super(DQNAgent, self).__init__()
        self.num_states = num_inputs
        self.num_actions = num_actions
        self.current_model = Q( self.num_states, num_actions).to(device)
        self.target_model = Q( self.num_states, num_actions).to(device)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=1e-3)
        self.gamma = gamma
        self.device = device
        self.train, self.test = True, False
    def act(self, state, epsilon):
       
        if random.random() > epsilon:
            q_value = self.current_model(state)
            action  = q_value.argmax().item()
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
        
    def compute_td_loss(self, batch_size):
        nb = min(batch_size, len(self.replay_buffer))
        state, goal, action, reward, next_state, done = self.replay_buffer.sample(nb)

        state      = Variable(torch.FloatTensor(np.float32(state))).to(self.device)
        goal = Variable(torch.FloatTensor(np.float32(goal))).to(self.device)
        next_state = Variable(torch.FloatTensor(np.float32(next_state))).to(self.device)
        action     = Variable(torch.LongTensor(action)).to(self.device)
        reward     = Variable(torch.FloatTensor(reward)).to(self.device)
        done       = Variable(torch.FloatTensor(done)).to(self.device)
        x = torch.cat((state, goal),1)
        q_values      = self.current_model(x)
        x_next = torch.cat((next_state, goal),1)
        next_q_values = self.current_model(x_next)
        next_q_state_values = self.target_model(x_next) 

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value -expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
def main():
    
    env = gym.make("gridworld-v0")
    env.setPlan("gridworld/gridworldPlansGoals/plan2.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    config = load_yaml("configs/config_random_gridworld.yaml")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = 100
    
    featureExtractor = config.featExtractor(env)
    batch_size = 1000
    gamma      = 0.99
    agent = DQNAgent(featureExtractor.outSize*2, env.action_space.n).to(device)
    
    losses = []
    all_rewards = []
    episode_reward = 0

    
    episode_count = config["nbEpisodes"]
    ob = env.reset()
    outdir = "checkpoints/"
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))
    #loadTensorBoard(SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    rsum = 0
    mean = 0
    itest = 0
    reward = 0
    done = False
    G = set()
    for i in range(episode_count):
        epsilon = epsilon_by_frame(i+1)
        ob = env.reset()
        ob = torch.tensor(featureExtractor.getFeatures(ob), dtype=torch.float).to(device)
        goal, _ = env.sampleGoal()
        goal = torch.tensor(featureExtractor.getFeatures(goal), dtype=torch.float).to(device)
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

      

        j = 0
         
        
        while True:
            env.render()
            x = torch.cat((ob, goal)).flatten()
            action = agent.act(x, 0.2)
            new_ob, _, _, _=env.step(action)
            new_ob = torch.tensor(featureExtractor.getFeatures(new_ob), dtype=torch.float).to(device)
            
            done=(new_ob==goal).all()
            #sparse rewards
            reward = 1.0 if done else -0.1
            # HER rewards
            reward_HER = 1.0 if (tuple(new_ob.reshape(-1).tolist()) in G) else -0.1
            if reward_HER == 1.0:
                agent.replay_buffer.push(ob, new_ob, action, reward_HER, new_ob, done)
                
            else:
                agent.replay_buffer.push(ob, goal, action, reward, new_ob, done)
            ob = new_ob
            rsum += reward
            j+=1
            if j % 10 == 0 and not agent.test:
                loss = agent.compute_td_loss(batch_size)
                losses.append(loss.item())
            if done:
                G.add(tuple(ob.reshape(-1).tolist()))
                
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ", "dernier état:")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
            
            if j > 100:
                ob = env.reset()
                rsum = 0
                break
                
        if i % 1000 == 0:
            agent.update_target()
    env.close()
main()