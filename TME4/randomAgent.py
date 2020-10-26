import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torch.nn as nn


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def save(self,outputDir):
        pass

    def load(self,inputDir):
        pass

class EpsilonGreedyDecay:
    def __init__(self, epsilon, eta, episode):
        self.epsilon = epsilon
        self.eta = eta
    def act(self, episode, Q_states):
        decay = self.epsilon / (1 + (self.eta * self.episode))
        if np.random.random() > decay:
            return np.argmax(Q_states)
        return  np.random.randint(len(Q_states))

class DQNAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt, explorer, gamma, batch_size):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.explorer = explorer
        self.buffer = Memory(10000)
        self.optimizer = SGD()
        self.loss = nn.SmoothL1Loss()
        self.Q = NN(self.featureExtractor.outSize, self.env.action_space.n,[100])
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        self.batch_size = batch_size

    def act(self,episode, observation, reward, done): 
        s = self.featureExtractor.getFeatures(observation) # faut le caster en torch.Tensor
        self.current_state = s
        action =  explorer.act(episode, self.Q.forward(self.current_state))
        if self.current_state is not None:
            transition = (self.prev_state, self.prev_action, self.current_state, reward, done)
            self.update_buffer(transition)
        if self.buffer.nentities == self.buffer.mem_size:
            self.fit()
        self.prev_state = self.current_state if not done else None
        self.prev_action = action
        return action

    def fit(self):
        samples = self.buffer.sample(self.batch_size) # liste de transitions de taille batch_size
        # construire des batch : st =  torch.cat([sample[0].view(1,-1) for sample in samples],) pareil pour les st+1, reward et done
        # done est utile pour construire un masque pour la target ( soit rt soit rt + gamma * max Q_{target}(s,a))
        # targetmax = gamma * max ... 
        # mask = torch.Tensor( 0 si done 1 si pas done)
        # target = mask * target + rt

    def save(self,outputDir):
        pass

    def load(self,inputDir):
        pass
    def update_buffer(self, transition):
        self.buffer.store(transition)
if __name__ == '__main__':
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/random_" + "-" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    agent = RandomAgent(env,config)

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            j+=1

            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()
