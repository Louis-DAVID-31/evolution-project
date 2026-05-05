from evogym import sample_robot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from evogym.utils import get_full_connectivity
from evogym import is_connected, has_actuator
from tqdm import tqdm
from datetime import datetime
import json
import os
from multiprocessing import Pool

class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
 
        self.n_out = n_out

    def reset(self):
        pass
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, Net, config, genes = None):
        self.config = config
        self.Net = Net
        self.model = None
        self.fitness = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.make_network()
        if genes is not None:
            self.genes = genes

    def __repr__(self):  # pragma: no cover
        return f"Agent {self.model} > fitness={self.fitness}"

    def __str__(self):  # pragma: no cover
        return self.__repr__()

    def make_network(self):
        n_in = self.config["n_in"]
        h_size = self.config["h_size"]
        n_out = self.config["n_out"]
        self.model = self.Net(n_in, h_size, n_out).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        assert len(params) == len(
            self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def mutate_ga(self):
        genes = self.genes
        n = len(genes)
        f = np.random.choice([False, True], size=n, p=[1/n, 1-1/n])
        
        new_genes = np.empty(n)
        new_genes[f] = genes[f]
        noise = np.random.randn(n-sum(f))
        new_genes[~f] = noise
        return new_genes

    def act(self, obs):
        # continuous actions
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions


base = np.array([
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4],
    [4, 4, 4, 4, 4]
    ])

def make_env(env_name, robot=None, seed=None, **kwargs):
    if robot is None: 
        env = gym.make(env_name)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot)
    env.robot = robot
    if seed is not None:
        env.seed(seed)
        
    return env

def evaluate(agent, env, max_steps=500, render=False):
    obs, i = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render()
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        reward += r
        steps += 1
    return reward

def get_cfg(env_name, robot=None):
    env = make_env(env_name, robot=base)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return cfg

def mp_eval(a, cfg):
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fit = evaluate(a, env, max_steps=cfg["max_steps"])
    env.close()
    return fit

def save_solution(a, cfg, base_dir="results"):
    save_cfg = {}
    for i in ["env_name", "robot", "n_in", "h_size", "n_out"]:
        assert i in cfg, f"{i} not in config"
        save_cfg[i] = cfg[i]
        
    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = a.genes.tolist()
    save_cfg["fitness"] = float(a.fitness)

    date_heure = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env_name = cfg["env_name"]
    target_dir = os.path.join(base_dir, f"{env_name}")
    os.makedirs(target_dir, exist_ok=True) 
    fitness_str = f"{save_cfg['fitness']:.2f}" 
    filename = f"{fitness_str}.json"
    full_path = os.path.join(target_dir, filename)
    
    with open(full_path, "w") as f:
        json.dump(save_cfg, f, indent=4) 
    return save_cfg

# ====================================================
# ====================================================

def _worker_eval(args):
    """Top-level function required for multiprocessing pickling."""
    genes, cfg = args
    agent = Agent(Network, cfg, genes=genes)
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fit = evaluate(agent, env, max_steps=cfg["max_steps"])
    env.close()
    return fit, genes

def ES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    # Update weights
    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i)
                          for i in range(1, mu + 1)])
    w /= np.sum(w)

    # Center of the distribution
    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fits = []
    total_evals = []

    n_workers = cfg.get("n_workers", os.cpu_count())

    bar = tqdm(range(cfg["generations"]))
    with Pool(processes=n_workers) as pool:
        for gen in bar:
            # Sample genes for the whole population
            noise = [np.random.randn(len(theta)) * cfg["sigma"] for _ in range(cfg["lambda"])]
            pop_genes = [theta + n for n in noise]

            # Evaluate all individuals in parallel
            args = [(genes, cfg) for genes in pop_genes]
            results = pool.map(_worker_eval, args)

            pop_fitness = [r[0] for r in results]

            # Sort by fitness (descending)
            idx = np.argsort([-f for f in pop_fitness])

            # Reconstruct population with fitness
            population = []
            for i, (fit, genes) in enumerate(results):
                agent = Agent(Network, cfg, genes=genes)
                agent.fitness = fit
                population.append(agent)

            # ES update step
            step = np.zeros(d)
            for i in range(mu):
                step += w[i] * (population[idx[i]].genes - theta)
            theta = theta + step * cfg["lr"]

            if pop_fitness[idx[0]] > elite.fitness:
                elite.genes = population[idx[0]].genes
                elite.fitness = pop_fitness[idx[0]]

            fits.append(elite.fitness)
            total_evals.append(cfg["lambda"] * (gen + 1))

            bar.set_description(f"Best: {elite.fitness:.4f}")
        
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.show()
    return elite

# ====================================================
# ====================================================

walker = np.array([
    [0, 0, 4, 4, 4],
    [0, 0, 4, 4, 4],
    [0, 0, 4, 4, 4],
    [0, 0, 4, 4, 4],
    [0, 0, 4, 4, 4]
    ])

env_name = 'Climber-v2'
robot = walker

cfg = get_cfg(env_name, robot)
a = Agent(Network, cfg)

config = {
    "env_name": "Climber-v2",
    "robot": walker,
    "generations": 100, 
    "lambda": 10,  # Population size
    "mu": 5,       # Parents pop size
    "sigma": 0.1,  # Mutation std
    "lr": 1,       # Learning rate
    "max_steps": 500,
    "n_workers": os.cpu_count(),  # Number of parallel workers
}

if __name__ == "__main__":
    a = ES(config)
    print(a.fitness)

    cfg = {**config, **cfg}
    save_solution(a, cfg)