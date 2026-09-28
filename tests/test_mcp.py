"""MCP server: an AI client can read and propose, but only the human's answer changes anything."""

import json

import anyio
import pytest
from mcp import Client
from mcp.types import ElicitResult

import backend
from conftest import events
from mcp_server import create_server


def call(tool, args, human="accept", ok=True, user="asha", mode=None):
    """Call a tool the way an AI client would; `human` is what the person answers when asked."""
    asked = []

    async def human_app(context, params):
        asked.append(params.message)
        return ElicitResult(action=human, content={"ok": ok} if human == "accept" else None)

    async def run():
        kwargs = {"elicitation_callback": human_app, **({"mode": mode} if mode else {})}
        async with Client(create_server(user, "mcp-test"), **kwargs) as client:
            return await client.call_tool(tool, args)

    result = anyio.run(run)
    assert not result.is_error, result.content
    # plain-dict tools have no output schema, so the SDK sends the JSON as text content
    return json.loads(result.content[0].text), asked


def test_no_self_approval_tool_exists():
    async def run():
        async with Client(create_server("asha", "mcp-test")) as client:
            return {t.name for t in (await client.list_tools()).tools}
    tools = anyio.run(run)
    assert tools == {"get_booking", "disruption_options", "escalate_to_human",
                     "cancel_booking", "choose_disruption_option"}
    assert not any("confirm" in t for t in tools)


def test_read_tools_respect_ownership():
    mine, _ = call("get_booking", {"pnr": "ACX123"})
    theirs, _ = call("get_booking", {"pnr": "ACX789"})               # Ravi's
    assert mine["route"] == "Pune -> Delhi"
    assert theirs["reason"] == "not_found"


@pytest.mark.parametrize("mode", [None, "legacy"])                   # new and old MCP clients
def test_cancel_asks_the_human_then_cancels(mode):
    out, asked = call("cancel_booking", {"pnr": "ACX456"}, mode=mode)
    assert asked == ["Cancel booking ACX456 (Mumbai -> Kolkata, departing 2026-10-02 09:15)? "
                     "This can't be undone."]
    assert out["ok"] and backend.BOOKINGS["ACX456"]["status"] == "cancelled"
    assert events() == ["cancel_proposed", "cancel_executed"]


@pytest.mark.parametrize("human, ok", [("accept", False), ("decline", None), ("cancel", None)])
def test_human_refusal_changes_nothing(human, ok):
    out, asked = call("cancel_booking", {"pnr": "ACX456"}, human=human, ok=ok)
    assert len(asked) == 1 and out["reason"] == "not_approved"
    assert backend.BOOKINGS["ACX456"]["status"] == "confirmed"
    assert events() == []


def test_someone_elses_booking_is_never_even_asked_about():
    out, asked = call("cancel_booking", {"pnr": "ACX789"})           # Ravi's
    assert asked == [] and out["reason"] == "not_found"
    assert backend.BOOKINGS["ACX789"]["status"] == "confirmed"


def test_disruption_option_with_human_approval():
    options, _ = call("disruption_options", {"pnr": "ACX789"}, user="ravi")
    assert [o["id"] for o in options["options"]] == ["rebook:AC312", "rebook:AC314", "refund"]
    out, asked = call("choose_disruption_option", {"pnr": "ACX789", "option_id": "refund"},
                      user="ravi")
    assert asked == ["For booking ACX789: Full refund of Rs 3,200 to your original payment "
                     "method. Go ahead?"]
    assert out["result"]["refund_amount"] == 3200


def test_ai_cannot_invent_an_option():
    """An AI asking for a refund the policy doesn't offer: the human isn't even asked."""
    out, asked = call("choose_disruption_option", {"pnr": "ACX123", "option_id": "refund"})
    assert asked == [] and out["reason"] == "option_unavailable"
    assert out["offered"] == ["voucher", "rebook:AC103"]
