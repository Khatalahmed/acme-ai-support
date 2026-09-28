"""The airline persona is opt-in: only customer-facing replies without their own system prompt."""

import pytest

import llm_backend


@pytest.fixture
def sent(monkeypatch):
    """Capture what would be sent to a non-fine-tuned model (the persona case)."""
    captured = []
    monkeypatch.setattr(llm_backend, "MODEL", "some-general-model")
    monkeypatch.setattr(llm_backend, "_ollama_chat",
                        lambda messages, name: captured.append(messages) or "ok")
    return captured


def roles(messages):
    return [m["role"] for m in messages]


def test_no_persona_by_default(sent):
    """e.g. the router's JSON call: exactly the prompt we wrote, nothing hidden added."""
    llm_backend.chat([{"role": "user", "content": "route this"}], name="router.llm")
    assert roles(sent[0]) == ["user"]


def test_persona_when_asked(sent):
    llm_backend.chat([{"role": "user", "content": "phrase this"}], name="tool.reply", persona=True)
    assert roles(sent[0]) == ["system", "user"]
    assert sent[0][0]["content"] == llm_backend.PERSONA


def test_callers_own_system_prompt_wins(sent):
    llm_backend.chat([{"role": "system", "content": "RAG rules"}, {"role": "user", "content": "q"}],
                     persona=True)
    assert sent[0][0]["content"] == "RAG rules" and len(sent[0]) == 2


def test_fine_tuned_model_never_gets_it(monkeypatch, sent):
    monkeypatch.setattr(llm_backend, "MODEL", "acme-support")   # persona baked into Modelfile
    llm_backend.chat([{"role": "user", "content": "x"}], persona=True)
    assert roles(sent[0]) == ["user"]
