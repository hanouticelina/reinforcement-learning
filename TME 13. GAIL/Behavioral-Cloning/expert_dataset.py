import pickle
import torch



class ExpertDataset:
    """
    Dataset to store expert's transitions
    """
    def __init__(self, env, path_to_file, device):
        self.path_to_file = path_to_file
        self.env = env
        self.floatTensor = torch.empty(1, dtype=float, device=device)
        self.longTensor = torch.empty(1, dtype=torch.long, device=device)
        self.nbFeatures = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n
        self.loadExpertTransitions(path_to_file)
        self.featureExtractor = opt.featExtractor(env)
        self.device = device
        
        
    def __len__(self):
        return self.expert_states.shape[0]
    
    
    def loadExpertTransitions(self, file):
        with open(file, "rb") as handle:
            expert_data = pickle.load(handle).to(self.floatTensor)
            expert_states = expert_data[:,:self.nbFeatures]
            expert_actions = expert_data[:,self.nbFeatures:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous().argmax(dim=-1).to(dtype=int)
            
            
  
    def get_expert_data(self):

        s = torch.tensor(self.expert_states, dtype=float, device=self.device)
        a = torch.tensor(self.expert_actions, dtype=float, device=self.device)
        return s, a
    def toOneHot(self, actions):
        actions = actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], self.nb_actions).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1.0
        return oneHot.to(self.floatTensor)

    def toIndexAction(self, oneHot):
        ac = self.longTensor.new(range(self.nb_actions)).view(1, -1)
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1)
        actions = ac[oneHot.view(-1) > 0].view(-1)
        return actions