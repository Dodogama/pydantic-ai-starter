"""Scripted FunctionModels so the example task runs end to end without an API key.

The weak coder's first attempt at step 2 has a flipped gradient sign, so the
offline run exercises the Verify -> Code retry edge before succeeding.
"""

import json
from itertools import cycle
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

EXAMPLE_TASK = "Implement LogisticRegression.loss and LogisticRegression.step in model.py so training works."

ORIGINAL = (Path(__file__).parent / "example_project" / "model.py").read_text()
LOSS = '''        eps = 1e-12
        return -sum(t * math.log(self.predict_proba(x) + eps) + (1 - t) * math.log(1 - self.predict_proba(x) + eps)
                    for x, t in zip(X, y)) / len(X)
'''
STEP = '''        loss = self.loss(X, y)
        n = len(X)
        errs = [self.predict_proba(x) - t for x, t in zip(X, y)]
        for j in range(len(self.w)):
            self.w[j] SIGN= self.lr * sum(e * x[j] for e, x in zip(errs, X)) / n
        self.b SIGN= self.lr * sum(errs) / n
        return loss
'''
TODO = "        raise NotImplementedError\n"
V1 = ORIGINAL.replace(TODO, LOSS, 1)
V2 = V1.replace(TODO, STEP.replace("SIGN", "-"), 1)
V2_BUGGY = V1.replace(TODO, STEP.replace("SIGN", "+"), 1)

PLAN = {"steps": [
    {"goal": "Implement mean binary cross-entropy in LogisticRegression.loss", "files": ["model.py"],
     "acceptance": '{python} -c "import math; from model import LogisticRegression as M; '
                   'assert abs(M(2).loss([[1, 2]], [1]) - math.log(2)) < 1e-9"'},
    {"goal": "Implement full-batch gradient descent in LogisticRegression.step", "files": ["model.py"],
     "acceptance": '{python} -c "from model import LogisticRegression as M; m = M(2); X, y = [[1, 0], [-1, 0]], [1, 0]; '
                   'a = m.step(X, y); assert m.loss(X, y) < a"'},
]}


def _structured(data: dict):
    def fn(messages, info: AgentInfo) -> ModelResponse:
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])
        return ModelResponse(parts=[TextPart(json.dumps(data))])
    return FunctionModel(fn)


def _coder(versions):
    """Each run writes the next model.py version, then replies with a summary."""
    it = iter(versions)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart("updated model.py")])
        return ModelResponse(parts=[ToolCallPart("write_file", {"path": "model.py", "content": next(it)})])
    return FunctionModel(fn)


def offline_models(always_buggy: bool = False) -> dict:
    code = cycle([V2_BUGGY]) if always_buggy else [V1, V2_BUGGY, V2]
    return {
        "plan": _structured(PLAN),
        "code": _coder(code),
        "review": _structured({"verdict": "approve", "notes": "gradient and loss look correct; no leakage"}),
        "escalate": _coder(cycle([V2_BUGGY]) if always_buggy else [V2]),
    }
