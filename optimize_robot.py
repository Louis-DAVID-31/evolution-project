import argparse
from datetime import datetime
import json
from multiprocessing import Pool
import os
from pathlib import Path

PLOT_CACHE_DIR = Path("/private/tmp/evolution_project_plot_cache")
(PLOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(PLOT_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from walker_shape_opti import Agent, Network, _worker_eval, _worker_init, get_cfg


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


def save_solution_old_style(agent, cfg, base_dir="results", suffix=None):
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
    filename = f"{fitness_str}.json" if suffix is None else f"{fitness_str}_{suffix}.json"
    output_path = target_dir / filename
    duplicate = 2
    while output_path.exists():
        if suffix is None:
            output_path = target_dir / f"{fitness_str}_{duplicate}.json"
        else:
            output_path = target_dir / f"{fitness_str}_{suffix}_{duplicate}.json"
        duplicate += 1

    with open(output_path, "w") as f:
        json.dump(save_cfg, f, indent=4)

    return output_path


def ES_with_history(config, run_idx=None):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    mu = cfg["mu"]
    weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    weights /= np.sum(weights)

    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)
    history = []

    n_workers = cfg.get("n_workers", os.cpu_count())
    prefix = f"[Run {run_idx}] " if run_idx is not None else ""
    bar = tqdm(range(cfg["generations"]), desc=f"{prefix}Best: -inf", leave=False)

    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        for _ in bar:
            pop_genes = [theta + np.random.randn(d) * cfg["sigma"] for _ in range(cfg["lambda"])]
            args = [(genes, cfg) for genes in pop_genes]
            results = pool.map(_worker_eval, args)

            pop_fitness = [result[0] for result in results]
            idx = np.argsort([-fitness for fitness in pop_fitness])

            population = [Agent(Network, cfg, genes=genes) for _, genes in results]
            for i, fit in enumerate(pop_fitness):
                population[i].fitness = fit

            step = np.zeros(d)
            for i in range(mu):
                step += weights[i] * (population[idx[i]].genes - theta)
            theta = theta + step * cfg["lr"]

            if pop_fitness[idx[0]] > elite.fitness:
                elite.genes = population[idx[0]].genes
                elite.fitness = pop_fitness[idx[0]]

            history.append(float(elite.fitness))
            bar.set_description(f"{prefix}Best: {elite.fitness:.4f}")

    return elite, history, cfg


def save_fitness_plot(histories, output_dir, solution_file):
    if not histories:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = Path(solution_file).stem
    output_path = output_dir / f"{stem}_fitness_{date_str}.png"

    plt.figure(figsize=(9, 5))
    for run_idx, history in enumerate(histories, start=1):
        generations = np.arange(1, len(history) + 1)
        plt.plot(generations, history, label=f"run {run_idx}")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Fitness over ES generations")
    plt.grid(True, alpha=0.3)
    if len(histories) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def optimize_fixed_robot(solution_file, generations, max_steps, lambda_, workers=None, multiple=1):
    robot_cfg = load_robot_shape(solution_file)
    workers = workers or os.cpu_count()

    es_config = {
        "env_name": robot_cfg["env_name"],
        "robot": robot_cfg["robot"],
        "generations": generations,
        "lambda": lambda_,
        "mu": min(5, lambda_),
        "sigma": 1,
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
    print(f"Runs: {multiple}")

    output_paths = []
    histories = []
    target_dir = Path("results") / es_config["env_name"]

    for run_idx in range(1, multiple + 1):
        print(f"\n=== Optimization {run_idx}/{multiple} ===")
        elite, history, save_cfg = ES_with_history(es_config, run_idx=run_idx if multiple > 1 else None)
        suffix = f"run_{run_idx:02d}" if multiple > 1 else None
        output_path = save_solution_old_style(elite, save_cfg, suffix=suffix)
        output_paths.append(output_path)
        histories.append(history)

        print(f"Best fitness: {elite.fitness:.4f}")
        print(f"Saved to: {output_path}")

    plot_path = save_fitness_plot(histories, target_dir, solution_file)
    if plot_path is not None:
        print(f"Fitness graph saved to: {plot_path}")

    return output_paths, plot_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a controller for a fixed EvoGym robot shape from a solution JSON file."
    )
    parser.add_argument("solution_file", help="Path to a result JSON containing env_name and robot.")
    parser.add_argument("generations", type=int, help="Number of ES generations.")
    parser.add_argument("max_steps", type=int, help="Maximum number of simulation steps per evaluation.")
    parser.add_argument("lambda_", type=int, metavar="lambda", help="ES population size per generation.")
    parser.add_argument(
        "--multiple",
        type=int,
        default=1,
        help="Run the same optimization several independent times. Default: 1.",
    )
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
    if args.multiple <= 0:
        raise SystemExit("--multiple must be a positive integer")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")

    optimize_fixed_robot(
        args.solution_file,
        generations=args.generations,
        max_steps=args.max_steps,
        lambda_=args.lambda_,
        workers=args.workers,
        multiple=args.multiple,
    )
