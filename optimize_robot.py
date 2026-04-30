import argparse
import json
import os
from pathlib import Path

import numpy as np

from walker_shape_opti import ES, get_cfg


def load_robot_shape(solution_file):
    solution_path = Path(solution_file).expanduser()
    with open(solution_path, "r") as f:
        cfg = json.load(f)

    for key in ("env_name", "robot"):
        if key not in cfg:
            raise KeyError(f"{key} not found in solution file: {solution_path}")

    return {
        "env_name": cfg["env_name"],
        "robot": np.array(cfg["robot"]),
    }


def save_solution_old_style(agent, cfg, base_dir="results"):
    save_cfg = {}
    for key in ["env_name", "robot", "n_in", "h_size", "n_out"]:
        if key not in cfg:
            raise KeyError(f"{key} not found in config")
        save_cfg[key] = cfg[key]

    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = agent.genes.tolist()
    save_cfg["fitness"] = float(agent.fitness)

    target_dir = Path(base_dir) / save_cfg["env_name"]
    target_dir.mkdir(parents=True, exist_ok=True)

    fitness_str = f"{save_cfg['fitness']:.2f}"
    output_path = target_dir / f"{fitness_str}.json"
    with open(output_path, "w") as f:
        json.dump(save_cfg, f, indent=4)

    return output_path


def optimize_fixed_robot(solution_file, generations, max_steps, lambda_, workers=None):
    robot_cfg = load_robot_shape(solution_file)
    workers = workers or os.cpu_count()

    es_config = {
        "env_name": robot_cfg["env_name"],
        "robot": robot_cfg["robot"],
        "generations": generations,
        "lambda": lambda_,
        "mu": min(5, lambda_),
        "sigma": 0.1,
        "lr": 1.0,
        "max_steps": max_steps,
        "n_workers": workers,
    }

    print(f"Optimizing fixed robot from: {solution_file}")
    print(f"Environment: {es_config['env_name']}")
    print(f"Generations: {generations}")
    print(f"Max steps: {max_steps}")
    print(f"Lambda: {lambda_}")
    print(f"Mu: {es_config['mu']}")
    print(f"Workers: {workers}")

    elite = ES(es_config)
    save_cfg = {**es_config, **get_cfg(es_config["env_name"], es_config["robot"])}
    output_path = save_solution_old_style(elite, save_cfg)

    print(f"Best fitness: {elite.fitness:.4f}")
    print(f"Saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a controller for a fixed EvoGym robot shape from a solution JSON file."
    )
    parser.add_argument("solution_file", help="Path to a result JSON containing env_name and robot.")
    parser.add_argument("generations", type=int, help="Number of ES generations.")
    parser.add_argument("max_steps", type=int, help="Maximum number of simulation steps per evaluation.")
    parser.add_argument("lambda_", type=int, metavar="lambda", help="ES population size per generation.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: os.cpu_count().",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.generations <= 0:
        raise SystemExit("generations must be a positive integer")
    if args.max_steps <= 0:
        raise SystemExit("max_steps must be a positive integer")
    if args.lambda_ <= 0:
        raise SystemExit("lambda must be a positive integer")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")

    optimize_fixed_robot(
        args.solution_file,
        generations=args.generations,
        max_steps=args.max_steps,
        lambda_=args.lambda_,
        workers=args.workers,
    )
