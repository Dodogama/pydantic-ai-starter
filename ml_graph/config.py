"""Per-project config: defaults merged with the target repo's agent.toml."""

import tomllib
from pathlib import Path

DEFAULT_CFG = {
    "conventions": "AGENTS.md",
    "checks": [],
    "final_checks": [],
    "weak_attempts": 3,
    "max_reviews": 2,
    "max_replans": 2,
    "timeout": 600,
    "models": {
        "plan": "anthropic:claude-opus-5-5",
        "code": "anthropic:claude-haiku-4-5",
        "review": "anthropic:claude-opus-5-5",
        "escalate": "anthropic:claude-opus-5-5",
    },
}


def load_cfg(root: Path, **overrides) -> dict:
    cfg = DEFAULT_CFG | tomllib.loads((root / "agent.toml").read_text()) | overrides
    cfg["models"] = DEFAULT_CFG["models"] | cfg.get("models", {})
    return cfg
