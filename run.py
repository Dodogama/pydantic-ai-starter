"""CLI runner: loads .env, picks models, copies the project to a temp dir, runs the graph."""

import argparse
import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from ml_graph import graph, run

HERE = Path(__file__).parent


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip().removeprefix("export "), value.strip().strip("'\""))


def main() -> None:
    from fake_models import EXAMPLE_TASK

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", default=EXAMPLE_TASK)
    ap.add_argument("--project", type=Path, default=HERE / "example_project")
    ap.add_argument("--in-place", action="store_true", help="edit the project directly instead of a temp copy")
    ap.add_argument("--offline", action="store_true", help="scripted fake models for the example task; no API key")
    ap.add_argument("--yes", action="store_true", help="skip the approval prompt before full eval")
    ap.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid and exit")
    a = ap.parse_args()

    if a.mermaid:
        print(graph.render(title="ml_graph"))
        return
    load_dotenv(HERE / ".env")
    models = None
    if a.offline:
        from fake_models import offline_models

        models = offline_models()
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example) or use --offline")

    root = a.project.resolve()
    if not a.in_place:
        root = Path(tempfile.mkdtemp(prefix="ml-graph-"))
        shutil.copytree(a.project, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    result, _ = asyncio.run(run(a.task, root, models, approve=(lambda _: True) if a.yes else None))
    print(f"\n{result}\nWorkspace: {root}")


if __name__ == "__main__":
    main()
