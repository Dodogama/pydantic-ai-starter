import asyncio
import shutil
from pathlib import Path

from fake_models import EXAMPLE_TASK, offline_models
from ml_graph import run

EXAMPLE = Path(__file__).parent / "example_project"


def _run(tmp_path, **kw):
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    return asyncio.run(run(EXAMPLE_TASK, tmp_path, approve=lambda _: True, **kw))


def test_retry_then_success(tmp_path):
    result, state = _run(tmp_path, models=offline_models())
    assert result.startswith("SUCCESS"), result
    assert state.trace.count("Code[weak]") == 3 and "Verify:fail" in state.trace
    assert "Code[strong]" not in state.trace


def test_escalates_then_gives_up(tmp_path):
    result, state = _run(tmp_path, models=offline_models(always_buggy=True), max_replans=0)
    assert result.startswith("FAILED"), result
    step2 = state.trace[state.trace.index("Review:approve") + 1:]
    assert [e for e in step2 if e.startswith("Code")] == ["Code[weak]"] * 3 + ["Code[strong]"]
    assert state.trace[-1] == "End"
