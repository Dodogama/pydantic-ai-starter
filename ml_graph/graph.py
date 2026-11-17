"""Plan -> Code -> Verify -> Review -> Approve -> Evaluate.

Edges are the return type hints of each node's run(). Shell commands, never a
model, decide whether checks and metrics pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic_ai.models import Model
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from .agents import Plan, coder, planner, reviewer
from .config import load_cfg
from .workspace import diff_since, sh, snapshot


@dataclass
class Deps:
    root: Path
    cfg: dict
    models: dict[str, Model | str]
    approve: Callable[[str], bool] = lambda msg: input(f"{msg} [y/N] ").strip().lower() == "y"

    def sh(self, cmd: str) -> tuple[bool, str]:
        return sh(self.root, cmd, self.cfg["timeout"])

    def conventions(self) -> str:
        p = self.root / self.cfg["conventions"]
        return f"Project conventions:\n{p.read_text()}" if p.is_file() else ""


@dataclass
class State:
    task: str
    plan: Plan | None = None
    plans: int = 0
    step: int = 0
    attempts: int = 0
    reviews: int = 0
    feedback: str = ""
    snapshot: dict[str, str] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    def log(self, event: str, detail: str = "") -> None:
        self.trace.append(event)
        print(f"[{len(self.trace):>3}] {event:<14} {detail}", flush=True)

    def start_step(self, root: Path) -> None:
        self.attempts, self.reviews, self.feedback, self.snapshot = 0, 0, "", snapshot(root)


@dataclass
class PlanNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> CodeNode | End[str]:
        s, d = ctx.state, ctx.deps
        if s.plans > d.cfg["max_replans"]:
            s.log("End", "replan budget exhausted")
            return End(f"FAILED after {s.plans} plans. Last failure:\n{s.feedback}")
        prompt = f"Task: {s.task}\n\n{d.conventions()}"
        if s.feedback:
            prompt += f"\n\nThe previous plan failed:\n{s.feedback}"
        s.plan = (await planner.run(prompt, deps=d.root, model=d.models["plan"])).output
        s.plans += 1
        s.step = 0
        s.log("Plan", " | ".join(st.goal for st in s.plan.steps))
        s.start_step(d.root)
        return CodeNode()


@dataclass
class CodeNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> VerifyNode:
        s, d = ctx.state, ctx.deps
        escalate = s.attempts >= d.cfg["weak_attempts"]
        step = s.plan.steps[s.step]
        prompt = (
            f"Step {s.step + 1}/{len(s.plan.steps)}: {step.goal}\nFiles: {', '.join(step.files)}\n"
            f"Acceptance command: {step.acceptance}\n\n{d.conventions()}"
        )
        if s.feedback:
            prompt += f"\n\nThe previous attempt failed:\n{s.feedback}"
        result = await coder.run(prompt, deps=d.root, model=d.models["escalate" if escalate else "code"])
        s.attempts += 1
        s.log("Code[strong]" if escalate else "Code[weak]", f"step {s.step + 1} try {s.attempts}: {result.output}")
        return VerifyNode()


@dataclass
class VerifyNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> CodeNode | ReviewNode | PlanNode:
        s, d = ctx.state, ctx.deps
        step = s.plan.steps[s.step]
        last = s.step == len(s.plan.steps) - 1
        for cmd in [*d.cfg["checks"], step.acceptance, *(d.cfg["final_checks"] if last else [])]:
            ok, out = d.sh(cmd)
            if not ok:
                s.feedback = f"$ {cmd}\n{out}"
                s.log("Verify:fail", f"{cmd}\n{'':20}-> {(out.strip().splitlines() or [''])[-1]}")
                if s.attempts <= d.cfg["weak_attempts"]:
                    return CodeNode()
                s.feedback = f"Step {s.step + 1} ({step.goal}) failed {s.attempts} attempts:\n{s.feedback}"
                return PlanNode()
        s.log("Verify:pass")
        return ReviewNode()


@dataclass
class ReviewNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> CodeNode | PlanNode | ApproveNode:
        s, d = ctx.state, ctx.deps
        step = s.plan.steps[s.step]
        prompt = f"Step: {step.goal}\n\nDiff:\n{diff_since(s.snapshot, d.root) or '(no changes)'}\n\n{d.conventions()}"
        r = (await reviewer.run(prompt, deps=d.root, model=d.models["review"])).output
        s.reviews += 1
        s.log(f"Review:{r.verdict}", r.notes)
        if r.verdict == "fix" and s.reviews <= d.cfg["max_reviews"]:
            s.feedback, s.attempts = f"Reviewer: {r.notes}", 0
            return CodeNode()
        if r.verdict != "approve":
            s.feedback = f"Reviewer rejected step {s.step + 1} ({step.goal}): {r.notes}"
            return PlanNode()
        s.step += 1
        if s.step < len(s.plan.steps):
            s.start_step(d.root)
            return CodeNode()
        return ApproveNode()


@dataclass
class ApproveNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> EvaluateNode | End[str]:
        s, d = ctx.state, ctx.deps
        if d.approve(f"All {len(s.plan.steps)} steps pass. Run full eval `{d.cfg['eval']}`?"):
            s.log("Approve:yes")
            return EvaluateNode()
        s.log("End", "eval not approved")
        return End("STOPPED: full eval not approved")


@dataclass
class EvaluateNode(BaseNode[State, Deps, str]):
    async def run(self, ctx: GraphRunContext[State, Deps]) -> PlanNode | End[str]:
        s, d = ctx.state, ctx.deps
        metric, target = d.cfg["metric"], d.cfg["target"]
        ok, out = d.sh(d.cfg["eval"])
        value = None
        if ok:
            try:
                value = float(json.loads(out.strip().splitlines()[-1])[metric])
            except (ValueError, KeyError, IndexError, TypeError):
                pass
        s.log("Evaluate", f"{metric}={value} target={target}")
        if value is not None and value >= target:
            return End(f"SUCCESS: {metric}={value:.4f} >= {target}")
        s.feedback = f"Full eval: {metric}={value}, target {target}.\n{out}"
        return PlanNode()


def build_graph():
    g = GraphBuilder(state_type=State, deps_type=Deps, input_type=PlanNode, output_type=str)
    g.add(
        g.edge_from(g.start_node).to(PlanNode),
        *(g.node(n) for n in (PlanNode, CodeNode, VerifyNode, ReviewNode, ApproveNode, EvaluateNode)),
    )
    return g.build()


graph = build_graph()


async def run(task: str, root: Path, models: dict | None = None, approve=None, **cfg_overrides) -> tuple[str, State]:
    """Run the graph on the repo at `root`. `models` overrides per-role models from agent.toml."""
    cfg = load_cfg(root, **cfg_overrides)
    deps = Deps(root=root, cfg=cfg, models=cfg["models"] | (models or {}))
    if approve:
        deps.approve = approve
    state = State(task=task)
    return await graph.run(state=state, deps=deps, inputs=PlanNode()), state
