import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


MAX_STEPS = 500
FPS = 100.0


def _flush_standard_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


@contextlib.contextmanager
def silence_external_output():
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
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

with silence_external_output():
    import evogym.envs
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
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network(
            config["n_in"],
            config["h_size"],
            config["n_out"],
        ).to(self.device).double()
        self.genes = np.array(config["genes"], dtype=np.float64)

    @property
    def genes(self):
        with torch.no_grad():
            vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        params = np.array(params, dtype=np.float64)
        if len(params) != len(self.genes):
            raise ValueError("Genome size does not fit the network size")
        if np.isnan(params).any():
            raise ValueError("NaN in genes")
        tensor = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(tensor, self.model.parameters())
        self.model = self.model.to(self.device).double()

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            return self.model(x).cpu().detach().numpy()


def load_solution(path):
    with open(path, "r") as f:
        config = json.load(f)

    required_keys = ["env_name", "robot", "n_in", "h_size", "n_out", "genes"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing keys in solution file: {', '.join(missing_keys)}")

    config["robot"] = np.array(config["robot"])
    config["genes"] = np.array(config["genes"], dtype=np.float64)
    return config


def make_env(env_name, robot):
    with silence_external_output():
        if robot is None:
            env = gym.make(env_name, render_mode="rgb_array")
        else:
            connections = get_full_connectivity(robot)
            try:
                env = gym.make(
                    env_name,
                    body=robot,
                    connections=connections,
                    render_mode="rgb_array",
                )
            except TypeError:
                env = gym.make(env_name, body=robot, render_mode="rgb_array")
    env.robot = robot
    return env


def evaluate_and_record(agent, env, max_steps=MAX_STEPS):
    with silence_external_output():
        obs, _ = env.reset()

    agent.model.reset()
    reward = 0
    frames = []
    done = False
    truncated = False
    steps = 0

    progress = tqdm(total=max_steps, desc="Simulation", unit="step")
    while not (done or truncated) and steps < max_steps:
        with silence_external_output():
            frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        action = agent.act(obs)
        with silence_external_output():
            obs, step_reward, done, truncated, _ = env.step(action)

        reward += step_reward
        steps += 1
        progress.update(1)

    progress.close()
    return reward, frames


def safe_filename_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "simulation"


def default_gif_path(solution_path, config):
    env_name = safe_filename_part(config["env_name"])
    stem = safe_filename_part(solution_path.stem)
    fitness = config.get("fitness")

    if fitness is not None:
        fitness_str = safe_filename_part(f"{float(fitness):.2f}")
        if stem != fitness_str:
            stem = f"{stem}_{fitness_str}"

    return solution_path.with_name(f"{env_name}_{stem}.gif")


def create_gif(solution_path):
    configure_quiet_libraries()

    solution_path = Path(solution_path).expanduser()
    config = load_solution(solution_path)
    agent = Agent(config)
    output_path = default_gif_path(solution_path, config)

    env = make_env(config["env_name"], config["robot"])
    try:
        reward, frames = evaluate_and_record(agent, env)
    finally:
        with silence_external_output():
            env.close()

    if not frames:
        raise RuntimeError("No frame was rendered; GIF was not created.")

    imageio.mimsave(output_path, frames, fps=FPS, loop=0)
    print(f"Fitness: {reward:.4f}")
    print(f"GIF saved to: {output_path}")
    return output_path


def find_pending_solutions(results_dir):
    pending = []
    skipped = []

    for solution_path in sorted(results_dir.rglob("*.json")):
        try:
            config = load_solution(solution_path)
            output_path = default_gif_path(solution_path, config)
        except Exception as exc:
            skipped.append((solution_path, exc))
            continue

        if not output_path.exists():
            pending.append(solution_path)

    return pending, skipped


def create_all_missing_gifs():
    results_dir = Path(__file__).resolve().parent / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    pending, skipped = find_pending_solutions(results_dir)
    for solution_path, exc in skipped:
        tqdm.write(f"Skipped {solution_path}: {exc}")

    if not pending:
        print("All GIFs are already generated.")
        return []

    created = []
    for solution_path in tqdm(pending, desc="Solutions", unit="file"):
        tqdm.write(f"Generating GIF for: {solution_path}")
        created.append(create_gif(solution_path))

    print(f"Generated {len(created)} GIF(s).")
    return created


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a simulation GIF from an EvoGym solution JSON file."
    )
    parser.add_argument("solution_file", nargs="?", help="Path to the solution JSON file.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate GIFs for every solution JSON in results/ that does not have one yet.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        create_all_missing_gifs()
    elif args.solution_file:
        create_gif(args.solution_file)
    else:
        raise SystemExit("Usage: python gif.py <solution_file> or python gif.py --all")
