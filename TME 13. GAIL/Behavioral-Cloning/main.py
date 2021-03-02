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
lr = 3e-3
epochs = 1000

with open('./configs/config_random_lunar.yaml', 'r') as stream:
    config  = yaml.load(stream,Loader=yaml.Loader)
env = gym.make(config['env'])
    
# expert data
expert_dataset = ExpertDataset(env, "./expert_data/expert.pkl", device)


input_size = env.observation_space.shape[0]
output_size = env.action_space.n


agent = BehavioralCloning(input_size, output_size).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

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


rsum = 0
mean = 0
verbose = True
itest = 0
it = 0
reward = 0
done = False
for i in range(episode_count):
    if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
        verbose = False #True
    else:
        verbose = False

    if i % config.freqTest == 0 and i >= config.freqTest:
        print("Test time! ")
        mean = 0
        agent.test = True

    if i % config.freqTest == config.nbTest and i > config.freqTest:
        print("End of test, mean reward=", mean / config.nbTest)
        itest += 1
        writer.add_scalar("rewardTest", mean / config.nbTest, itest)
        agent.test = False

    if i % config.freqSave == 0:
        with open(savepath, 'wb') as f:
            torch.save(agent, f)
    j = 0
    if verbose:
        env.render()

    done = False
    while not(done):              

        action, prob = agent.act(ob)
        ob_new, reward, done, _ = env.step(action)
        ob = ob_new
        j += 1
        it += 1
        rsum += reward
        if it % epochs == 0 and i > 0:
            states, actions = expert_dataset.get_expert_data()
            actions = actions.to(dtype=int)
            ids = torch.arange(0,len(states), dtype=int, device=device)
            pi = torch.log(agent(s)[ids, a])
            L = - pi.mean()
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            writer.add_scalar("loss/actor", L, it)
            
        if done:
            if i % config.freqPrint == 0:
                print(f'{i} rsum={int(rsum)}, {j} actions')
            writer.add_scalar("reward", rsum, i)
            mean += rsum
            rsum = 0
            ob = env.reset()
            break
env.close()