import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from walker_shape_opti import ES, get_cfg, has_actuator, is_connected


BLOCK_TYPES = np.array([1, 2, 3, 4])
NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def load_generation_results(folder):
    results_dir = Path(folder).expanduser()
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    results_dir = results_dir.resolve()

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Generation folder not found: {results_dir}")

    entries = []
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            with open(json_path, "r") as f:
                cfg = json.load(f)

            if "env_name" not in cfg or "robot" not in cfg:
                continue

            entries.append(
                {
                    "path": json_path,
                    "env_name": cfg["env_name"],
                    "robot": np.array(cfg["robot"], dtype=int),
                    "fitness": float(cfg.get("fitness", json_path.stem)),
                }
            )
        except Exception as exc:
            print(f"Skipped {json_path.name}: {exc}")

    entries.sort(key=lambda entry: entry["fitness"], reverse=True)
    return results_dir, entries


def validate_same_environment(entries):
    env_names = {entry["env_name"] for entry in entries}
    if len(env_names) != 1:
        raise ValueError(f"Expected one environment in the folder, found: {sorted(env_names)}")
    return entries[0]["env_name"]


def is_valid_robot(robot):
    return is_connected(robot) and has_actuator(robot)


def in_bounds(robot, row, col):
    return 0 <= row < robot.shape[0] and 0 <= col < robot.shape[1]


def is_adjacent_to_robot(robot, row, col):
    for d_row, d_col in NEIGHBOR_OFFSETS:
        n_row = row + d_row
        n_col = col + d_col
        if in_bounds(robot, n_row, n_col) and robot[n_row, n_col] != 0:
            return True
    return False


def mutation_probabilities(robot, p_add=None, p_remove=None, p_change=None):
    default_probability = 1.0 / (3 * robot.size)
    return {
        "add": default_probability if p_add is None else p_add,
        "remove": default_probability if p_remove is None else p_remove,
        "change": default_probability if p_change is None else p_change,
    }


def validate_mutation_probability(name, value):
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def try_add_block(robot, row, col):
    if robot[row, col] != 0 or not is_adjacent_to_robot(robot, row, col):
        return None

    mutated = robot.copy()
    mutated[row, col] = int(np.random.choice(BLOCK_TYPES))
    return mutated if is_valid_robot(mutated) else None


def try_remove_block(robot, row, col):
    if robot[row, col] == 0:
        return None

    mutated = robot.copy()
    mutated[row, col] = 0
    return mutated if is_valid_robot(mutated) else None


def try_change_block(robot, row, col):
    current = robot[row, col]
    if current == 0:
        return None

    choices = BLOCK_TYPES[BLOCK_TYPES != current]
    mutated = robot.copy()
    mutated[row, col] = int(np.random.choice(choices))
    return mutated if is_valid_robot(mutated) else None


def apply_probabilistic_mutations(robot, probabilities):
    mutated = robot.copy()
    mutation_count = 0
    cells = [(row, col) for row in range(robot.shape[0]) for col in range(robot.shape[1])]
    np.random.shuffle(cells)

    for row, col in cells:
        if mutated[row, col] == 0:
            if np.random.random() < probabilities["add"]:
                candidate = try_add_block(mutated, row, col)
                if candidate is not None:
                    mutated = candidate
                    mutation_count += 1
        else:
            if np.random.random() < probabilities["remove"]:
                candidate = try_remove_block(mutated, row, col)
                if candidate is not None:
                    mutated = candidate
                    mutation_count += 1
                    continue

            if np.random.random() < probabilities["change"]:
                candidate = try_change_block(mutated, row, col)
                if candidate is not None:
                    mutated = candidate
                    mutation_count += 1

    return mutated, mutation_count


def enumerate_single_valid_mutations(robot):
    candidates = []

    for row in range(robot.shape[0]):
        for col in range(robot.shape[1]):
            if robot[row, col] == 0:
                if not is_adjacent_to_robot(robot, row, col):
                    continue
                for block_type in BLOCK_TYPES:
                    mutated = robot.copy()
                    mutated[row, col] = int(block_type)
                    if is_valid_robot(mutated):
                        candidates.append(mutated)
            else:
                removed = robot.copy()
                removed[row, col] = 0
                if is_valid_robot(removed):
                    candidates.append(removed)

                for block_type in BLOCK_TYPES[BLOCK_TYPES != robot[row, col]]:
                    changed = robot.copy()
                    changed[row, col] = int(block_type)
                    if is_valid_robot(changed):
                        candidates.append(changed)

    return candidates


def mutate_robot(robot, probabilities, max_attempts=100):
    for _ in range(max_attempts):
        mutated, mutation_count = apply_probabilistic_mutations(robot, probabilities)
        if mutation_count > 0 and not np.array_equal(mutated, robot) and is_valid_robot(mutated):
            return mutated

    candidates = enumerate_single_valid_mutations(robot)
    if not candidates:
        raise RuntimeError("No valid mutation found for robot")
    return candidates[int(np.random.randint(len(candidates)))]


def build_mutated_population(parents, population_size, p_add=None, p_remove=None, p_change=None):
    if population_size % len(parents) != 0:
        raise ValueError("population_size must be divisible by n_best so each parent has the same number of children")

    children_per_parent = population_size // len(parents)
    population = []

    for parent_index, parent in enumerate(parents, start=1):
        probabilities = mutation_probabilities(parent["robot"], p_add, p_remove, p_change)
        for child_index in range(1, children_per_parent + 1):
            child_robot = mutate_robot(parent["robot"], probabilities)
            population.append(
                {
                    "robot": child_robot,
                    "parent_path": parent["path"],
                    "parent_fitness": parent["fitness"],
                    "parent_rank": parent_index,
                    "child_index": child_index,
                }
            )

    return population


def default_output_dir(project_root, source_dir, env_name):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_source_name = source_dir.name.replace("/", "_")
    return project_root / "results" / f"{safe_source_name}_new_generation_{date_str}"


def save_solution_unique(agent, cfg, output_dir, robot_idx):
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
    output_path = output_dir / f"{fitness_str}.json"
    if output_path.exists():
        output_path = output_dir / f"{fitness_str}_robot_{robot_idx:03d}.json"

    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"{fitness_str}_robot_{robot_idx:03d}_{suffix}.json"
        suffix += 1

    with open(output_path, "w") as f:
        json.dump(save_cfg, f, indent=4)

    return str(output_path)


def optimize_population(env_name, population, generations, lambda_, max_steps, output_dir, workers=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count()

    all_results = []
    for robot_idx, item in enumerate(tqdm(population, desc="Robots", unit="robot"), start=1):
        config = {
            "env_name": env_name,
            "robot": item["robot"],
            "generations": generations,
            "lambda": lambda_,
            "mu": min(5, lambda_),
            "sigma": 0.1,
            "lr": 1.0,
            "max_steps": max_steps,
            "n_workers": workers,
        }

        try:
            elite = ES(config, robot_idx=robot_idx)
            cfg_full = {**config, **get_cfg(env_name, item["robot"])}
            path = save_solution_unique(elite, cfg_full, output_dir, robot_idx)
        except Exception as exc:
            tqdm.write(f"[Robot {robot_idx}] Failed: {exc}")
            continue

        all_results.append(
            {
                "robot_idx": robot_idx,
                "fitness": elite.fitness,
                "path": path,
                "parent_rank": item["parent_rank"],
                "parent_fitness": item["parent_fitness"],
                "parent_path": str(item["parent_path"]),
            }
        )
        tqdm.write(
            f"[Robot {robot_idx:3d}/{len(population)}] "
            f"fitness={elite.fitness:.5f} parent=#{item['parent_rank']} -> {path}"
        )

    return all_results


def run_new_generation(
    folder,
    n_best,
    population_size,
    generations,
    lambda_,
    max_steps,
    output_dir=None,
    workers=None,
    p_add=None,
    p_remove=None,
    p_change=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    validate_mutation_probability("--p-add", p_add)
    validate_mutation_probability("--p-remove", p_remove)
    validate_mutation_probability("--p-change", p_change)

    project_root = Path(__file__).resolve().parent
    source_dir, entries = load_generation_results(folder)
    if not entries:
        raise RuntimeError(f"No robot JSON found in: {source_dir}")
    if n_best > len(entries):
        raise ValueError(f"n_best={n_best} but only {len(entries)} robots were found")

    env_name = validate_same_environment(entries)
    parents = entries[:n_best]
    population = build_mutated_population(
        parents,
        population_size,
        p_add=p_add,
        p_remove=p_remove,
        p_change=p_change,
    )

    if output_dir is None:
        output_dir = default_output_dir(project_root, source_dir, env_name)
    else:
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    print(f"Source: {source_dir}")
    print(f"Environment: {env_name}")
    print(f"Parents: {n_best}")
    print(f"Population: {population_size}")
    print(f"Children per parent: {population_size // n_best}")
    print(f"Generations: {generations}")
    print(f"Lambda: {lambda_}")
    print(f"Max steps: {max_steps}")
    print(f"Output: {output_dir}")

    results = optimize_population(
        env_name,
        population,
        generations,
        lambda_,
        max_steps,
        output_dir,
        workers=workers,
    )

    if results:
        best = max(results, key=lambda result: result["fitness"])
        print(f"Done. Best fitness: {best['fitness']:.5f}")
        print(f"Best saved to: {best['path']}")
    else:
        print("Done. No robot was successfully optimized.")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a new generation by mutating the best robots from a result folder and optimizing each child."
    )
    parser.add_argument("folder", help="Result folder containing one generation of robot JSON files.")
    parser.add_argument("n_best", type=int, help="Number of best parent robots to keep.")
    parser.add_argument("population_size", type=int, help="Number of mutated child robots to create.")
    parser.add_argument("generations", type=int, help="Number of ES generations for each child robot.")
    parser.add_argument("lambda_", type=int, metavar="lambda", help="ES population size for each generation.")
    parser.add_argument("max_steps", type=int, help="Maximum simulation steps per evaluation.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: results/<source>_new_generation_<date>.")
    parser.add_argument("--workers", type=int, default=None, help="ES multiprocessing workers. Default: os.cpu_count().")
    parser.add_argument("--p-add", type=float, default=None, help="Per-cell probability of adding a connected block.")
    parser.add_argument("--p-remove", type=float, default=None, help="Per-cell probability of removing a block.")
    parser.add_argument("--p-change", type=float, default=None, help="Per-cell probability of changing a non-empty block type.")
    parser.add_argument("--seed", type=int, default=None, help="Optional numpy random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for name in ("n_best", "population_size", "generations", "lambda_", "max_steps"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be a positive integer")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")

    run_new_generation(
        args.folder,
        n_best=args.n_best,
        population_size=args.population_size,
        generations=args.generations,
        lambda_=args.lambda_,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        workers=args.workers,
        p_add=args.p_add,
        p_remove=args.p_remove,
        p_change=args.p_change,
        seed=args.seed,
    )
