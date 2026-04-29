import argparse
import copy
import contextlib
from datetime import datetime
import json
import logging
from multiprocessing import Pool
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


def _flush_standard_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


@contextlib.contextmanager
def silence_external_output():
    """Mute noisy third-party simulator output while preserving our own prints."""
    _flush_standard_streams()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
    finally:
        _flush_standard_streams()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

with silence_external_output():
    import evogym.envs
    from evogym import has_actuator, is_connected, sample_robot
    from evogym.sim import EvoSim
    from evogym.utils import get_full_connectivity

EvoSim._has_displayed_version = True


def configure_quiet_libraries():
    logging.getLogger("gymnasium").setLevel(logging.ERROR)
    logging.getLogger("evogym").setLevel(logging.ERROR)
    try:
        gym.logger.set_level(logging.ERROR)
    except AttributeError:
        pass


configure_quiet_libraries()

# ====================================================
# Hyperparamètres globaux
# ====================================================

N_ROBOTS = 5       # Nombre de robots à générer
M_GENERATIONS = 30  # Générations d'ES par robot
MAX_STEP = 200     # Nombre maximum de steps par simulation

# Types EvoGym :
# 0 = vide, 1 = rigide, 2 = mou, 3 = actuateur horizontal, 4 = actuateur vertical
BLOCK_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.2]

# ====================================================
# Neural Network
# ====================================================

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, Net, config, genes=None):
        self.config = config
        self.Net = Net
        self.model = None
        self.fitness = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.make_network()
        if genes is not None:
            self.genes = genes

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
            vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        assert len(params) == len(self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise ValueError("NaN in genes")
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions


# ====================================================
# Environment & Evaluation
# ====================================================

def normalize_block_probabilities(block_probs=None):
    if block_probs is None:
        block_probs = BLOCK_PROBABILITIES

    if isinstance(block_probs, dict):
        probs = np.array([block_probs.get(block_type, 0.0) for block_type in range(5)], dtype=float)
    else:
        probs = np.array(block_probs, dtype=float)

    if probs.shape != (5,):
        raise ValueError("block_probs must contain exactly 5 probabilities, one for each block type 0..4")
    if np.any(probs < 0):
        raise ValueError("block_probs cannot contain negative probabilities")
    if probs.sum() <= 0:
        raise ValueError("block_probs must contain at least one positive probability")
    if probs[3] + probs[4] <= 0:
        raise ValueError("block_probs must allow at least one actuator type: 3 or 4")

    return probs / probs.sum()


def generate_valid_robot_shape(width=5, height=5, block_probs=None, max_attempts=10000):
    block_types = np.arange(5)
    probabilities = normalize_block_probabilities(block_probs)

    body = np.random.choice(block_types, size=(width, height), p=probabilities)
    attempts = 1
    while not (is_connected(body) and has_actuator(body)):
        if attempts >= max_attempts:
            raise RuntimeError(
                "Could not generate a valid robot shape. Try increasing actuator/non-empty block probabilities."
            )
        body = np.random.choice(block_types, size=(width, height), p=probabilities)
        attempts += 1
    return body

def make_env(env_name, robot=None, seed=None):
    with silence_external_output():
        if robot is None:
            env = gym.make(env_name)
        else:
            env = gym.make(env_name, body=robot)
        if seed is not None:
            env.seed(seed)
    env.robot = robot
    return env

def evaluate(agent, env, max_steps=500):
    with silence_external_output():
        obs, _ = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.act(obs)
        with silence_external_output():
            obs, r, done, trunc, _ = env.step(action)
        reward += r
        steps += 1
    return reward

def get_cfg(env_name, robot):
    env = make_env(env_name, robot=robot)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    with silence_external_output():
        env.close()
    return cfg


# ====================================================
# Multiprocessing worker
# ====================================================

def _worker_init():
    configure_quiet_libraries()


def _worker_eval(args):
    """Top-level function required for multiprocessing pickling."""
    genes, cfg = args
    agent = Agent(Network, cfg, genes=genes)
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fit = evaluate(agent, env, max_steps=cfg["max_steps"])
    with silence_external_output():
        env.close()
    return fit, genes


# ====================================================
# ES Optimisation
# ====================================================

def ES(config, robot_idx=None):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    w /= np.sum(w)

    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    n_workers = cfg.get("n_workers", os.cpu_count())

    prefix = f"[Robot {robot_idx}] " if robot_idx is not None else ""
    bar = tqdm(range(cfg["generations"]), desc=f"{prefix}Best: -inf", leave=False)

    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        for gen in bar:
            pop_genes = [theta + np.random.randn(d) * cfg["sigma"] for _ in range(cfg["lambda"])]
            args = [(genes, cfg) for genes in pop_genes]
            results = pool.map(_worker_eval, args)

            pop_fitness = [r[0] for r in results]
            idx = np.argsort([-f for f in pop_fitness])

            population = [Agent(Network, cfg, genes=g) for _, g in results]
            for i, fit in enumerate(pop_fitness):
                population[i].fitness = fit

            step = np.zeros(d)
            for i in range(mu):
                step += w[i] * (population[idx[i]].genes - theta)
            theta = theta + step * cfg["lr"]

            if pop_fitness[idx[0]] > elite.fitness:
                elite.genes = population[idx[0]].genes
                elite.fitness = pop_fitness[idx[0]]

            bar.set_description(f"{prefix}Best: {elite.fitness:.4f}")

    return elite


# ====================================================
# Save
# ====================================================

def save_solution(agent, cfg, run_dir):
    save_cfg = {
        "env_name": cfg["env_name"],
        "robot": cfg["robot"].tolist(),
        "n_in": cfg["n_in"],
        "h_size": cfg["h_size"],
        "n_out": cfg["n_out"],
        "genes": agent.genes.tolist(),
        "fitness": float(agent.fitness),
    }
    fitness_str = f"{save_cfg['fitness']:.5f}"
    filename = f"{fitness_str}.json"
    full_path = os.path.join(run_dir, filename)
    with open(full_path, "w") as f:
        json.dump(save_cfg, f, indent=4)
    return full_path


# ====================================================
# Main loop : generate → optimise → save
# ====================================================

def run_population_search(
    env_name="Walker-v0",
    n_robots=N_ROBOTS,
    m_generations=M_GENERATIONS,
    es_config=None,
    block_probs=None,
):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("results", f"{env_name}_{date_str}")
    os.makedirs(run_dir, exist_ok=True)

    default_es = {
        "lambda": 10,
        "mu": 5,
        "sigma": 0.1,
        "lr": 1.0,
        "max_steps": 500,
        "n_workers": os.cpu_count(),
    }
    if es_config:
        default_es.update(es_config)

    all_results = []

    print(default_es)
    print(block_probs)

    print(f"\n{'='*50}")
    print(f"Population search: {n_robots} robots × {m_generations} generations")
    print(f"Results → {run_dir}")
    print(f"{'='*50}\n")

    for i in tqdm(range(n_robots), desc="Robots", position=0):
        # 1. Generate a valid random robot
        robot = generate_valid_robot_shape(block_probs=block_probs)

        # 2. Build ES config for this robot
        config = {
            "env_name": env_name,
            "robot": robot,
            "generations": m_generations,
            **default_es,
        }

        # 3. Optimise
        try:
            elite = ES(config, robot_idx=i + 1)
        except Exception as e:
            print(f"\n[Robot {i+1}] Failed: {e}")
            continue

        # 4. Save
        cfg_full = {**config, **get_cfg(env_name, robot)}
        path = save_solution(elite, cfg_full, run_dir)
        all_results.append({"robot_idx": i + 1, "fitness": elite.fitness, "path": path})

        tqdm.write(f"[Robot {i+1:3d}/{n_robots}] fitness={elite.fitness:.4f}  → {path}")

    # Summary
    if all_results:
        fitnesses = [r["fitness"] for r in all_results]
        best = all_results[int(np.argmax(fitnesses))]
        print(f"\n{'='*50}")
        print(f"Done. Best robot: #{best['robot_idx']} — fitness={best['fitness']:.4f}")
        print(f"Saved to: {best['path']}")
        print(f"{'='*50}\n")

    return all_results


# ====================================================
# Entry point
# ====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EvoGym Walker evolution."
    )
    parser.add_argument(
        "n_robots",
        nargs="?",
        type=int,
        default=N_ROBOTS,
        help=f"Nombre de robots à générer. Défaut: {N_ROBOTS}",
    )
    parser.add_argument(
        "m_generation",
        nargs="?",
        type=int,
        default=M_GENERATIONS,
        help=f"Nombre de générations ES par robot. Défaut: {M_GENERATIONS}",
    )
    parser.add_argument(
        "max_step",
        nargs="?",
        type=int,
        default=MAX_STEP,
        help=f"Nombre maximum de steps par simulation. Défaut: {MAX_STEP}",
    )
    args = parser.parse_args()

    for name in ("n_robots", "m_generation", "max_step"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be a positive integer")

    return args


if __name__ == "__main__":
    args = parse_args()
    results = run_population_search(
        env_name="Walker-v0",
        n_robots=args.n_robots,
        m_generations=args.m_generation,
        block_probs=[0.2, 0.1, 0.1, 0.3, 0.3],
        es_config={
            "lambda": 10,
            "mu": 5,
            "sigma": 0.1,
            "lr": 1.0,
            "max_steps": args.max_step,
            "n_workers": os.cpu_count(),
        },
    )
