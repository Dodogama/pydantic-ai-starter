# pydantic-ai-starter

An agent graph for ML engineering tasks. A strong model plans and reviews, a weak model writes code, and shell commands decide whether checks and metrics pass.

```
Plan(strong) → Code(weak) → Verify(shell) → Review(strong) → Approve(human) → Evaluate(shell)
```

| Edge | When |
|---|---|
| Verify → Code | check failed and attempts ≤ `weak_attempts` (the attempt after the weak ones uses the `escalate` model) |
| Verify → Plan | the strong-model attempt failed too |
| Review → Code / Plan | `fix` (up to `max_reviews` times) / `replan` |
| Review → Code (next step) or Approve | `approve` |
| Evaluate → End / Plan | metric ≥ `target` / below it |
| Plan → End | more than `max_replans` replans |

Every loop has a counter, so a run always ends. Run `python run.py --mermaid` for the diagram.

## Layout

```
ml_graph/config.py     defaults + agent.toml loading
ml_graph/workspace.py  shell commands, snapshots, diffs
ml_graph/agents.py     planner / coder / reviewer + structured outputs + repo tools
ml_graph/graph.py      state, nodes (edges = run() return types), run()
run.py                 command-line runner: .env, models, temp copy
fake_models.py         scripted models for --offline and tests
```

## Run

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --offline --yes      # scripted models, no API key
.venv/bin/python -m pytest -q                     # success path + escalation/give-up path
cp .env.example .env   # paste your key from https://platform.claude.com/settings/keys
.venv/bin/python run.py                           # real models on example_project
```

Each run works on a temp copy of the project unless you pass `--in-place`.

## Using it on another repo

Add an `agent.toml` (see `example_project/agent.toml`) and an optional `AGENTS.md` with conventions, then run:

```sh
.venv/bin/python run.py "your task" --project path/to/repo
```

The config sets the per-step `checks`, the last-step `final_checks` (unit tests, smoke training), the `eval` command and its `metric`/`target`, the loop budgets, and which model plays each role under `[models]`.

> The planner writes acceptance commands and they run through a shell. Point this at untrusted code only inside a container.
