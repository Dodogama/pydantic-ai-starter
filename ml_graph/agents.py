"""The three LLM roles, their structured outputs, and the repo tools they share.

Agents have no model baked in; the graph passes one per role at run time.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from .workspace import repo_files


class Step(BaseModel):
    goal: str
    files: list[str]
    acceptance: str = Field(
        description="Shell command run from the repo root that exits 0 only when this step is done. "
        "Write {python} for the Python interpreter."
    )


class Plan(BaseModel):
    steps: list[Step] = Field(min_length=1, max_length=5)


class Review(BaseModel):
    verdict: Literal["approve", "fix", "replan"]
    notes: str


def _resolve(root: Path, path: str) -> Path:
    p = (root / path).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ModelRetry(f"{path} is outside the repo")
    return p


def list_files(ctx: RunContext[Path]) -> list[str]:
    """List all files in the repo."""
    return sorted(str(p.relative_to(ctx.deps)) for p in repo_files(ctx.deps))


def read_file(ctx: RunContext[Path], path: str) -> str:
    """Read a file from the repo."""
    p = _resolve(ctx.deps, path)
    if not p.is_file():
        raise ModelRetry(f"{path} does not exist")
    return p.read_text()


def write_file(ctx: RunContext[Path], path: str, content: str) -> str:
    """Create or overwrite a repo file. `content` is the complete new file."""
    p = _resolve(ctx.deps, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {path}"


READ_TOOLS = [list_files, read_file]

planner = Agent(
    output_type=Plan,
    deps_type=Path,
    tools=READ_TOOLS,
    instructions=(
        "You plan changes to an ML codebase for a less capable coding model to implement. "
        "Read the relevant code first. Split the task into 1-5 small, ordered steps; each must leave the repo "
        "importable and have an acceptance command that proves that step alone is done. "
        "If a previous attempt failed, fix the plan so the same failure cannot recur."
    ),
)
coder = Agent(
    deps_type=Path,
    tools=[*READ_TOOLS, write_file],
    instructions=(
        "Implement exactly the one step you are given by writing files. Read files before changing them and "
        "write complete file contents. Don't touch unrelated code. Reply with a one-line summary."
    ),
)
reviewer = Agent(
    output_type=Review,
    deps_type=Path,
    tools=READ_TOOLS,
    instructions=(
        "Review one step of an ML code change. Automated checks already passed, so look for what tests miss: "
        "train/validation leakage, fitting on eval data, wrong loss reduction or gradient, train/eval mode "
        "mix-ups, unseeded randomness, shape/broadcast bugs, silent dtype or device issues. "
        "Verdict 'fix' for a local bug (say exactly what to change), 'replan' if the step itself is wrong, "
        "'approve' otherwise. Don't nitpick style."
    ),
)
