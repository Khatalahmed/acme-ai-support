"""The quality gate's own logic: it must pass healthy results and fail on any crossed limit."""

import copy

import pytest

from evals import gate

HEALTHY = {
    "tool": {"unwanted_actions": 0, "action_recall": 1.0, "escalation_recall": 1.0,
             "escalation_precision": 1.0, "conversation_pass_rate": 1.0},
    "rag": {"grounded_numbers": 1.0, "unanswerable_refused": 1.0, "retrieval_hit_at_5": 1.0,
            "fact_recall": 1.0},
    "router": {"accuracy": 0.984, "wrongful_cancels_per_run": 1.3, "cancel_recall": 0.92,
               "escalation_recall": 1.0},
}


def with_(path, value):
    m = copy.deepcopy(HEALTHY)
    group, name = path.split(".")
    m[group][name] = value
    return m


def test_healthy_results_pass():
    rows, failed = gate.judge(HEALTHY)
    assert failed == [] and all(ok for *_, ok in rows)


def test_every_limit_in_gate_json_is_checked():
    rows, _ = gate.judge(HEALTHY)
    checked = {f"{g}.{n}" for g, n, *_ in rows}
    configured = {f"{g}.{n}" for g, checks in gate.LIMITS.items() if not g.startswith("_")
                  for n in checks}
    assert checked == configured


@pytest.mark.parametrize("path, value", [
    ("tool.unwanted_actions", 1),           # one unwanted action is an incident
    ("rag.grounded_numbers", 40 / 41),      # one invented number
    ("rag.unanswerable_refused", 0.75),     # answered something the policies don't cover
    ("tool.action_recall", 17 / 18),
    ("router.wrongful_cancels_per_run", 3.5),
    ("router.accuracy", 0.94),
])
def test_crossing_a_limit_fails(path, value):
    _, failed = gate.judge(with_(path, value))
    assert failed == [path]


@pytest.mark.parametrize("path, value", [
    ("router.accuracy", 0.95),              # exactly at a floor passes
    ("router.wrongful_cancels_per_run", 3), # exactly at a ceiling passes
    ("rag.fact_recall", 0.97),
])
def test_boundaries_pass(path, value):
    assert gate.judge(with_(path, value))[1] == []


def test_missing_metric_fails():
    """A metric that couldn't be computed must not count as passing."""
    assert gate.judge(with_("tool.escalation_precision", None))[1] == ["tool.escalation_precision"]


def test_critical_flags():
    rows, _ = gate.judge(HEALTHY)
    critical = {f"{g}.{n}" for g, n, _, _, crit, _ in rows if crit}
    assert {"tool.unwanted_actions", "rag.grounded_numbers",
            "rag.unanswerable_refused"} <= critical
