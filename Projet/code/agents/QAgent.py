import random
from kaggle_environments.envs.rps.utils import get_score
import numpy as np

   

class QAgent:
  
    
    def __init__(self, alpha=.7, alpha_decay=1, discount=.3, epsilon=.8, epsilon_decay=.9):
        # Parameters
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # Initialize states
        self.Q = np.zeros((9, 3))
        self.current_state = 0
        self.current_action = 0
        self.last_action = 0
        self.reward = 0
        self.state_dict = {(0, 0): 0,
                           (0, 1): 1,
                           (0, 2): 2,
                           (1, 0): 3,
                           (1, 1): 4,
                           (1, 2): 5,
                           (2, 0): 6,
                           (2, 1): 7,
                           (2, 2): 8}


    def act(self, observation, reward=0):
        
        reward = observation.reward
        self.reward = reward
        if observation.step == 0:
            self.current_action = int(np.random.randint(0, 3))
            self.last_action = self.current_action
        elif observation.step == 1:
            self.current_state = self.state_dict[(self.current_action, observation.lastOpponentAction)]
            if self.epsilon > np.random.exponential(0, 1):
                self.current_action = int(np.random.randint(0, 3))
            else:
                self.current_action = int(self.Q[self.current_state, :].argmax())
            self.last_action = self.current_action
        else:
            tmp_state = self.current_state
            discounted_tmp_state = self.alpha*(self.reward+self.discount *
                                               self.Q[tmp_state, self.Q[tmp_state, :].argmax()] -
                                               self.Q[self.current_state, self.current_action])
            self.Q[self.current_state, self.current_action] = (self.Q[self.current_state, self.current_action] +
                                                discounted_tmp_state)
            self.current_state = self.state_dict[(self.current_action, observation.lastOpponentAction)]
            if self.epsilon > np.random.exponential(0, 1):
                self.current_action = int(np.random.randint(0, 3))
            else:
                self.current_action = int(self.Q[self.current_state, :].argmax())
            self.alpha *= self.alpha_decay
            self.epsilon *= self.epsilon_decay
        return self.current_action
    def play(self, obs, configuration):
        if obs.step == 0:
            self.current_action = int(np.random.choice(2,1)[0])
            return self.current_action
        self.current_state = self.state_dict[(self.current_action, obs.lastOpponentAction)]
        self.current_action = int(self.Q[self.current_state, :].argmax())
        return self.current_action