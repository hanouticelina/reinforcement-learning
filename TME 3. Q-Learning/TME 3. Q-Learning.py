import matplotlib

matplotlib.use("Qt5agg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class EpsilonGreedyDecay:
    def __init__(self, epsilon, eta, epsilon_min):
        self.eta = eta
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
    def act(self, episode, values):
        decay = self.epsilon / (1 + (self.eta * episode))
        if decay<self.epsilon_min:
            decay=self.epsilon_min
        if np.random.random() > decay:
            return np.argmax(values)
        return  np.random.randint(len(values))



class QLearning(object):

    def __init__(self, env, learning_rate, discount, epsilon=1):
 
        self.env = env
        self.action_space = env.action_space
        self.Q = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None
        self.lasta = None
        self.epsilonGreedyDecay=EpsilonGreedyDecay(self.epsilon,0.0001,0.01)
        self.episode=0

    def act(self, observation, reward, done):
        self.episode+=1
        state = self.env.state2str(observation)
        self.obs = state
        self.Q.setdefault(state, [0, 0, 0, 0])
        self.reward = reward
        action = self.epsilonGreedyDecay.act(self.episode,self.Q[self.obs])
        self.update(action)
        return action

    def update(self, action):
        if not self.lastobs is None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate * (self.reward + self.discount * np.max(self.Q[st1]) - self.Q[st][self.lasta])
        self.lastobs = self.obs
        self.lasta = action


class Sarsa(object):
    
    def __init__(self, env, learning_rate, discount, epsilon=1):
 
        self.env = env
        self.action_space = env.action_space
        self.Q = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None
        self.lasta = None
        self.epsilonGreedyDecay=EpsilonGreedyDecay(self.epsilon,0.0001,0.01)
        self.episode=0

    def act(self, observation, reward, done):
        self.episode+=1
        state = self.env.state2str(observation)
        self.obs = state
        self.Q.setdefault(state, [0, 0, 0, 0])
        self.reward = reward
        action = self.epsilonGreedyDecay.act(self.episode,self.Q[self.obs])
        self.update(action)
        return action

    def update(self, action):
        if not self.lastobs is None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate * (self.reward + self.discount * self.Q[st1][action] - self.Q[st][self.lasta])
        self.lastobs = self.obs
        self.lasta = action

        
class Dyna_Q(object):
    def __init__(self, action_space, explorer, gamma, alpha, alpha_R, alpha_P, k):
        self.action_space = action_space
        self.explorer = explorer
        self.Q = defaultdict(lambda : np.zeros(4))
        self.old_at = None
        self.old_st= None
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_P = alpha_P
        self.alpha_R = alpha_R
        self.k = k
        self.P = defaultdict(lambda : defaultdict(lambda : np.zeros(4)))
        self.R = defaultdict(lambda : defaultdict(lambda : np.zeros(4)))


    # after execution of a_(t-1) we go to st and we get reward = r_(t-1)
    def act(self, s_t, reward, done):
        s_t = str(s_t.tolist())
        a_t = self.explorer.choose(s_t = s_t,Q = self.Q)
        self.learn(self.old_st, self.old_at, reward, s_t)
        self.old_at = a_t
        self.old_st = s_t
        return a_t

    def learn(self, old_st, old_at, reward, s_t):
        max = self.Q[s_t].max()
        # value-based part
        self.Q[old_st][old_at] = self.Q[old_st][old_at] + self.alpha * (reward + self.gamma * max  - self.Q[old_st][old_at])
        # model-based part
        self.R[old_st][s_t][old_at] = self.R[old_st][s_t][old_at] + self.alpha_R * (reward - self.R[old_st][s_t][old_at])
        self.P[s_t][old_st][old_at] =  self.P[s_t][old_st][old_at] + self.alpha_R * (int((s_t != old_st)) - self.P[s_t][old_st][old_at])

        states = np.random.choice(np.array(list(self.P.keys())) , size = self.k )
        actions = np.array( [self.action_space.sample() for i in range(self.k) ] )
        for state, action in zip(states, actions):
            d1 = self.Q[state][action]
            d2 = sum(self.P[FutureState][state][action] * (self.R[state][FutureState][action] + self.gamma * self.Q[FutureState].max()) for FutureState in self.P)
            self.Q[state][action] =  d1+ self.alpha *( d2 - d1)       

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.01, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    learning_rate=0.5
    discount= 0.75 # à revoir
    # Execution avec un Agent
    agent = Sarsa(env,learning_rate,discount)


    episode_count = 10000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
          env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
               env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
