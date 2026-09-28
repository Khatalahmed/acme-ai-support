"""The RAG eval's style checks (deterministic, no LLM). Examples are real replies from the live
Azure deployment / baseline run, before RAG_SYSTEM stopped asking for "empathy, the policy
answer, and a next-step question"."""

import pytest

from evals.rag_eval import leaked_labels, opens_with_apology

BEFORE = ("I'm sorry for any confusion - happy to help.\n\nPolicy answer: Each passenger may carry "
          "one cabin bag of up to 7 kg.\n\nNext-step question: Would you like help checking it?")


def test_real_leaked_template_is_caught():
    assert leaked_labels(BEFORE) == ["next-step question", "policy answer"]


@pytest.mark.parametrize("reply", [
    "**Policy answer:** 23 kg per passenger.",
    "- Next step question: anything else?",
    "Empathy: we understand.\nAnswer: 23 kg.",
])
def test_heading_variants_are_caught(reply):
    assert leaked_labels(reply)


@pytest.mark.parametrize("reply", [
    "The allowance is 23 kg per passenger. Would you like to add extra weight?",
    "- Checked bags: 23 kg\n- Cabin bags: 7 kg",                   # a list of rules is fine
    "Your answer: the policy covers this in section 2.",           # label word mid-line
])
def test_normal_writing_is_not_flagged(reply):
    assert leaked_labels(reply) == []


@pytest.mark.parametrize("reply, expected", [
    (BEFORE, True),
    ("We apologise for the delay. You are entitled to a voucher.", True),
    ("The allowance is 23 kg. I'm sorry, but bags over 32 kg go as cargo.", False),  # 2nd sentence
    ("The allowance is 23 kg per passenger.", False),
])
def test_apology_opening(reply, expected):
    assert opens_with_apology(reply) is expected
