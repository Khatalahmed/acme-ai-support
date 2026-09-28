"""Labels typed by a person in Langfuse: valid ones parse exactly, anything else is rejected."""

import pytest

from evals.import_disagreements import parse_label


@pytest.mark.parametrize("label, routes, pnr", [
    ("rag", ["rag"], None),
    ("clarify|rag", ["clarify", "rag"], None),
    ("tool:get_flight_status ACX123", ["tool:get_flight_status"], "ACX123"),
    ("tool:cancel_ticket acx456 | clarify", ["tool:cancel_ticket", "clarify"], "ACX456"),
    ({"route": "escalate"}, ["escalate"], None),
])
def test_valid_labels(label, routes, pnr):
    assert parse_label(label) == (routes, pnr)


@pytest.mark.parametrize("label", [
    "RAG please",                             # not a route name
    "tool:get_flight_status",                 # tool route without a PNR
    "tool:get_flight_status ACX123 ACX456",   # two PNRs
    "",
    "cancel",                                 # must be the full tool name
])
def test_invalid_labels_are_rejected(label):
    with pytest.raises(ValueError):
        parse_label(label)
