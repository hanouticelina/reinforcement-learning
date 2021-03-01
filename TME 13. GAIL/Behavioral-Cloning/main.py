from utils import *
from expert_dataset import *
from behavioral_cloning import *
from torch import *
import torch
import datetime
import yaml
import gym
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# Load args

# behavioral cloning args
lr = 0.003
epochs = 100

with open('./configs/config_random_lunar.yaml', 'r') as stream:
    config  = yaml.load(stream,Loader=yaml.Loader)
env = gym.make(config['env'])
    
# expert data
expert_dataset = ExpertDataset(env, "./expert_data/expert.pkl", device)
batch_size = len(expert_dataset)
expert_dataloader = DataLoader(expert_dataset, batch_size=batch_size)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n


agent = BehavioralCloning(input_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(agent.parameters(), lr=lr)

# training loop


nb_checks = 500
nb_episodes = 100
    


# ---environment---#
config = load_yaml("./configs/config_random_lunar.yaml")
env = gym.make(config.env)
if hasattr(env, "setPlan"):
    env.setPlan(config.map, config.rewards)
tstart = str(time.time()).replace(".", "_")
env.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
episode_count = config.nbEpisodes
ob = env.reset()

agent_id = "BC"
# ---yaml and tensorboard---#
outdir = "./runs/"+ agent_id + "_" + tstart
print("Saving in " + outdir)
os.makedirs(outdir, exist_ok=True)
save_src(os.path.abspath(outdir))
write_yaml(os.path.join(outdir, "info.yaml"), config)
writer = SummaryWriter(outdir)


for k in range(nb_checks):

    for epoch in range(epochs):

        for states, actions in expert_dataloader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            output = agent(states.float())
            _, targets = actions.max(dim=1)
            loss = criterion(output, Variable(targets))
            loss.backward()
            optimizer.step() 

    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False

    for i in range(nb_episodes):
        done = False

        while not (done):

            # get the action from the imitator
            probs = agent(torch.tensor(ob).to(device)) # generate distributions over states
            action = torch.argmax(nn.Softmax()(probs), dim=0).item() # action that maximize log likelihood

            ob_new, reward, done, _ = env.step(action)  # process action
            ob = ob_new
            # reward cumulé
            rsum += reward

            # if game is done
            if done:
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
    print(f"{k} mean_rewards={mean/nb_episodes}")
    writer.add_scalar("reward", mean/nb_episodes, k)
    env.close()