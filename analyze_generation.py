import argparse
from pathlib import Path
import webbrowser

from gif import create_gifs_parallel, default_gif_path, load_solution


HTML_FILENAME = "analysis.html"


def load_robot_results(results_dir):
    robots = []

    for solution_path in sorted(results_dir.glob("*.json")):
        config = load_solution(solution_path)
        fitness = float(config.get("fitness", solution_path.stem))
        gif_path = default_gif_path(solution_path, config)
        robots.append(
            {
                "solution_path": solution_path,
                "gif_path": gif_path,
                "env_name": config["env_name"],
                "fitness": fitness,
            }
        )

    robots.sort(key=lambda robot: robot["fitness"], reverse=True)
    return robots


def ensure_gifs_exist(robots, workers=None):
    missing = [robot for robot in robots if not robot["gif_path"].exists()]
    if not missing:
        return

    create_gifs_parallel(
        [robot["solution_path"] for robot in missing],
        workers=workers,
    )


def html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def relative_path(path, start):
    return Path(path).resolve().relative_to(start.resolve()).as_posix()


def render_robot_cards(robots, results_dir):
    cards = []
    for rank, robot in enumerate(robots, start=1):
        gif_src = html_escape(relative_path(robot["gif_path"], results_dir))
        json_name = html_escape(robot["solution_path"].name)
        env_name = html_escape(robot["env_name"])
        fitness = robot["fitness"]
        card = f"""
        <article class="robot-card">
            <div class="media-wrap">
                <img src="{gif_src}" alt="{env_name} robot {rank}" loading="lazy">
            </div>
            <div class="meta">
                <span class="rank">#{rank}</span>
                <span class="fitness">fitness {fitness:.4f}</span>
                <span class="file">{json_name}</span>
            </div>
        </article>
        """
        cards.append(card)
    return "\n".join(cards)


def render_html(results_dir, robots):
    title = html_escape(results_dir.name)
    cards = render_robot_cards(robots, results_dir)
    best = max((robot["fitness"] for robot in robots), default=0.0)
    worst = min((robot["fitness"] for robot in robots), default=0.0)

    return f"""<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        :root {{
            color-scheme: light;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f7f8;
            color: #172026;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            background: #f5f7f8;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 2;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 24px;
            padding: 18px 28px;
            border-bottom: 1px solid #d8dee3;
            background: rgba(245, 247, 248, 0.94);
            backdrop-filter: blur(8px);
        }}

        h1 {{
            margin: 0;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0;
        }}

        .summary {{
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
            font-size: 13px;
            color: #42515c;
        }}

        .summary span {{
            padding: 5px 8px;
            border: 1px solid #d8dee3;
            background: #ffffff;
        }}

        main {{
            padding: 22px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 14px;
        }}

        .robot-card {{
            overflow: hidden;
            border: 1px solid #d8dee3;
            border-radius: 8px;
            background: #ffffff;
        }}

        .media-wrap {{
            display: grid;
            place-items: center;
            aspect-ratio: 4 / 3;
            background: #eef2f4;
        }}

        .media-wrap img {{
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .meta {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 3px 8px;
            padding: 9px 10px 10px;
            font-size: 13px;
            line-height: 1.25;
        }}

        .rank {{
            grid-row: span 2;
            align-self: center;
            font-weight: 800;
            color: #0f766e;
        }}

        .fitness {{
            font-weight: 700;
            color: #172026;
        }}

        .file {{
            overflow-wrap: anywhere;
            color: #64717b;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="summary">
            <span>{len(robots)} robots</span>
            <span>best {best:.4f}</span>
            <span>worst {worst:.4f}</span>
        </div>
    </header>
    <main>
        <section class="grid">
            {cards}
        </section>
    </main>
    <script>
        document.querySelectorAll("img").forEach((img) => {{
            img.addEventListener("error", () => {{
                img.closest(".robot-card").classList.add("missing-gif");
            }});
        }});
    </script>
</body>
</html>
"""


def write_analysis_page(results_dir, robots):
    html_path = results_dir / HTML_FILENAME
    html_path.write_text(render_html(results_dir, robots), encoding="utf-8")
    return html_path


def analyze_generation(folder, open_page=True, workers=None, best_count=None):
    results_dir = Path(folder).expanduser()
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    results_dir = results_dir.resolve()

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results folder not found: {results_dir}")

    robots = load_robot_results(results_dir)
    if not robots:
        raise RuntimeError(f"No solution JSON found in: {results_dir}")

    total_robots = len(robots)
    if best_count is not None:
        robots = robots[:best_count]

    ensure_gifs_exist(robots, workers=workers)
    html_path = write_analysis_page(results_dir, robots)

    print(f"Robots: {len(robots)}/{total_robots}")
    print(f"Best fitness: {robots[0]['fitness']:.4f}")
    print(f"Page saved to: {html_path}")

    if open_page:
        webbrowser.open(html_path.as_uri())

    return html_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one HTML page with every robot GIF in a generation folder."
    )
    parser.add_argument("folder", help='Generation folder, e.g. "results/Walker-v0_2026-04-29_17-28-20".')
    parser.add_argument("--best", type=int, default=None, help="Only render and generate GIFs for the N best robots.")
    parser.add_argument("--no-open", action="store_true", help="Create the page without opening it.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel GIF workers when missing GIFs must be generated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.best is not None and args.best <= 0:
        raise SystemExit("--best must be a positive integer")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be a positive integer")
    analyze_generation(
        args.folder,
        open_page=not args.no_open,
        workers=args.workers,
        best_count=args.best,
    )
