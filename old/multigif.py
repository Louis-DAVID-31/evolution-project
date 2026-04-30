import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_OPACITY = 0.25
DEFAULT_THRESHOLD = 24.0
DEFAULT_BACKGROUND_SAMPLES = 40


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


def find_gif_path(solution_path, config):
    env_name = safe_filename_part(config["env_name"])
    stem = safe_filename_part(solution_path.stem)
    fitness = config.get("fitness")

    candidates = [
        default_gif_path(solution_path, config),
        solution_path.with_name(f"{env_name}_{stem}.gif"),
        solution_path.with_suffix(".gif"),
    ]
    if fitness is not None:
        candidates.append(solution_path.with_name(f"{env_name}_{float(fitness):.2f}.gif"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(solution_path.parent.glob(f"*{stem}*.gif"))
    return matches[0] if matches else None


def load_robot_entries(results_dir):
    entries = []
    skipped = []

    for solution_path in sorted(results_dir.glob("*.json")):
        try:
            with open(solution_path, "r") as f:
                config = json.load(f)

            if "env_name" not in config:
                raise KeyError("env_name")

            fitness = float(config.get("fitness", solution_path.stem))
            gif_path = find_gif_path(solution_path, config)
            if gif_path is None:
                skipped.append((solution_path, "missing gif"))
                continue

            entries.append(
                {
                    "solution_path": solution_path,
                    "gif_path": gif_path,
                    "fitness": fitness,
                    "env_name": config["env_name"],
                }
            )
        except Exception as exc:
            skipped.append((solution_path, exc))

    entries.sort(key=lambda entry: entry["fitness"], reverse=True)
    return entries, skipped


def estimate_background_rgb(frame):
    rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    h, w, _ = rgb.shape
    patch = max(1, min(10, h // 5, w // 5))
    samples = np.concatenate(
        [
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def center_on_canvas(frame, canvas_size, background=None):
    if background is None:
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", canvas_size, background)

    x = (canvas_size[0] - frame.width) // 2
    y = (canvas_size[1] - frame.height) // 2
    canvas.alpha_composite(frame, dest=(x, y))
    return canvas


def frame_on_canvas(frame, canvas_size):
    rgba = frame.convert("RGBA")
    background = estimate_background_rgb(rgba)
    background_rgba = tuple(int(v) for v in background) + (255,)
    return center_on_canvas(rgba, canvas_size, background=background_rgba)


def build_background_model(gif, canvas_size, sample_count=DEFAULT_BACKGROUND_SAMPLES):
    sample_count = min(sample_count, gif.n_frames)
    sample_indices = np.linspace(0, gif.n_frames - 1, sample_count, dtype=int)
    samples = []

    for frame_index in sample_indices:
        frame = read_frame(gif, int(frame_index))
        samples.append(np.asarray(frame_on_canvas(frame, canvas_size).convert("RGB"), dtype=np.uint8))

    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def robot_mask(frame, background_model, canvas_size, threshold):
    canvas = frame_on_canvas(frame, canvas_size)
    arr = np.array(canvas, dtype=np.uint8)
    rgb = arr[:, :, :3].astype(np.int16)
    source_alpha = arr[:, :, 3]

    distance = np.linalg.norm(rgb - background_model.astype(np.int16), axis=2)
    return (distance > threshold) & (source_alpha > 0)


def robot_layer(frame, background_model, canvas_size, opacity, threshold):
    canvas = frame_on_canvas(frame, canvas_size)
    arr = np.array(canvas, dtype=np.uint8)
    mask = robot_mask(frame, background_model, canvas_size, threshold)

    source_alpha = arr[:, :, 3]
    alpha = np.zeros_like(source_alpha)
    alpha[mask] = np.minimum(source_alpha[mask], int(255 * opacity))
    arr[:, :, 3] = alpha

    return Image.fromarray(arr)


def synced_environment_layer(frame, background_model, canvas_size, threshold):
    canvas = frame_on_canvas(frame, canvas_size)
    arr = np.array(canvas, dtype=np.uint8)
    mask = robot_mask(frame, background_model, canvas_size, threshold)
    rgb = arr[:, :, :3]
    rgb[mask] = background_model[mask]
    arr[:, :, :3] = rgb
    arr[:, :, 3] = 255
    return Image.fromarray(arr)


def read_frame(gif, frame_index):
    gif.seek(frame_index % gif.n_frames)
    return gif.convert("RGBA")


def compose_multigif(
    entries,
    output_path,
    opacity=DEFAULT_OPACITY,
    threshold=DEFAULT_THRESHOLD,
    max_frames=None,
    background_samples=DEFAULT_BACKGROUND_SAMPLES,
):
    gifs = [Image.open(entry["gif_path"]) for entry in entries]
    try:
        best_gif = gifs[0]
        frame_count = best_gif.n_frames
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        duration = best_gif.info.get("duration", 100)
        canvas_size = (
            max(gif.size[0] for gif in gifs),
            max(gif.size[1] for gif in gifs),
        )

        print("Building synchronized backgrounds...")
        background_models = [
            build_background_model(gif, canvas_size, sample_count=background_samples)
            for gif in tqdm(gifs, desc="Backgrounds", unit="gif")
        ]

        frames = []
        for frame_index in tqdm(range(frame_count), desc="Composing", unit="frame"):
            best_frame = read_frame(best_gif, frame_index)
            composite = synced_environment_layer(
                best_frame,
                background_models[0],
                canvas_size,
                threshold,
            )

            for gif, background_model, entry_index in reversed(list(zip(gifs, background_models, range(len(gifs))))):
                layer_opacity = 1.0 if entry_index == 0 else opacity
                layer = robot_layer(
                    read_frame(gif, frame_index),
                    background_model,
                    canvas_size,
                    layer_opacity,
                    threshold,
                )
                composite = Image.alpha_composite(composite, layer)
            frames.append(composite)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )
    finally:
        for gif in gifs:
            gif.close()

    return output_path


def build_multigif(
    folder,
    output=None,
    opacity=DEFAULT_OPACITY,
    threshold=DEFAULT_THRESHOLD,
    max_frames=None,
    background_samples=DEFAULT_BACKGROUND_SAMPLES,
):
    results_dir = Path(folder).expanduser()
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    results_dir = results_dir.resolve()

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Folder not found: {results_dir}")

    entries, skipped = load_robot_entries(results_dir)
    for solution_path, reason in skipped:
        print(f"Skipped {solution_path.name}: {reason}")

    if not entries:
        raise RuntimeError(f"No JSON/GIF robot pairs found in: {results_dir}")

    output_path = Path(output).expanduser() if output else results_dir / f"{results_dir.name}_multigif.gif"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    print(f"Robots: {len(entries)}")
    print(f"Best fitness: {entries[0]['fitness']:.4f}")
    print(f"Best GIF: {entries[0]['gif_path']}")

    output_path = compose_multigif(
        entries,
        output_path.resolve(),
        opacity=opacity,
        threshold=threshold,
        max_frames=max_frames,
        background_samples=background_samples,
    )
    print(f"Saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay all robot GIFs from a result folder into one comparison GIF."
    )
    parser.add_argument("folder", help='Result folder, e.g. "results/Walker-v0_2026-04-29_17-28-20".')
    parser.add_argument("--output", default=None, help="Output GIF path. Default: <folder>/<folder>_multigif.gif")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY, help=f"Opacity for non-best robots. Default: {DEFAULT_OPACITY}")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Robot extraction threshold. Default: {DEFAULT_THRESHOLD}")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick previews.")
    parser.add_argument("--background-samples", type=int, default=DEFAULT_BACKGROUND_SAMPLES, help=f"Frames sampled to estimate each GIF background. Default: {DEFAULT_BACKGROUND_SAMPLES}")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0 <= args.opacity <= 1:
        raise SystemExit("--opacity must be between 0 and 1")
    if args.threshold < 0:
        raise SystemExit("--threshold must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be a positive integer")
    if args.background_samples <= 0:
        raise SystemExit("--background-samples must be a positive integer")

    build_multigif(
        args.folder,
        output=args.output,
        opacity=args.opacity,
        threshold=args.threshold,
        max_frames=args.max_frames,
        background_samples=args.background_samples,
    )
