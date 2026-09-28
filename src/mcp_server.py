"""MCP server: ACME's booking tools for any MCP-capable assistant (Claude, Copilot, ...).

The caller here is another AI, so the one thing it must never do is approve its own changes.
There is deliberately NO `confirm` tool: every change asks the HUMAN through MCP elicitation
(the question appears in their app; the model can't answer it), and only then runs - through
the same actions.py tool layer as the API, so ownership, entitlement re-checks, idempotency and
the audit log apply unchanged.

Run (stdio):   ACME_MCP_TOKEN=demo-asha uv run python src/mcp_server.py
Claude Desktop / Claude Code config:
    {"command": "uv", "args": ["run", "python", "src/mcp_server.py"],
     "env": {"ACME_MCP_TOKEN": "demo-asha"}}
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import (  # noqa: E402
    AcceptedElicitation,
    Elicit,
    ElicitationResult,
    MCPServer,
    Resolve,
)

import actions  # noqa: E402
import policy  # noqa: E402
from auth import user_from_token  # noqa: E402


class Confirm(BaseModel):
    ok: bool


def _approved(confirm):
    return isinstance(confirm, AcceptedElicitation) and confirm.data.ok


def create_server(user_id, session_id=None):
    """An MCP server acting for one authenticated passenger."""
    session_id = session_id or f"mcp:{uuid.uuid4().hex[:8]}"
    mcp = MCPServer("ACME Bharat Airlines")

    # ---------------------------------------------------------------- read tools

    @mcp.tool()
    def get_booking(pnr: str) -> dict:
        """Status of one of the passenger's own bookings (PNR like ACX123)."""
        return actions.flight_status(user_id, session_id, pnr)

    @mcp.tool()
    def disruption_options(pnr: str) -> dict:
        """What the passenger is entitled to for a delayed or cancelled flight, per ACME policy.
        Offer ONLY these options; they are computed from policy, not negotiable."""
        out = actions.disruption_options(user_id, session_id, pnr)
        if out["ok"]:
            out["options"] = [{**o, "description": policy.describe(o)} for o in out["options"]]
        return out

    @mcp.tool()
    def escalate_to_human(pnr: str, summary: str) -> dict:
        """Hand the case to a human agent (e.g. the passenger is upset or wants an exception)."""
        return actions.escalate(user_id, session_id, pnr, "mcp_request", summary)

    # ---------------------------------------------------------------- changes: human approves

    async def ask_cancel(pnr: str) -> Confirm | Elicit[Confirm]:
        """Resolver: ask the human - only when there is actually something to cancel."""
        b = actions.flight_status(user_id, session_id, pnr)
        if not b["ok"] or b["booking_status"] == "cancelled":
            return Confirm(ok=False)          # nothing to ask; the tool body explains
        return Elicit(f"Cancel booking {b['pnr']} ({b['route']}, departing {b['departure']})? "
                      "This can't be undone.", Confirm)

    @mcp.tool()
    async def cancel_booking(
        pnr: str, confirm: Annotated[ElicitationResult[Confirm], Resolve(ask_cancel)],
    ) -> dict:
        """Cancel one of the passenger's bookings. The passenger is asked to confirm directly."""
        if not _approved(confirm):
            status = actions.flight_status(user_id, session_id, pnr)
            return status if not status["ok"] else {"ok": False, "reason": "not_approved",
                                                    "note": "Nothing was changed."}
        proposal = actions.propose_cancel(user_id, session_id, pnr, source="mcp")
        return actions.confirm(user_id, session_id) if proposal["ok"] else proposal

    async def ask_option(pnr: str, option_id: str) -> Confirm | Elicit[Confirm]:
        out = actions.disruption_options(user_id, session_id, pnr)
        option = next((o for o in out.get("options", []) if o["id"] == option_id), None)
        if not option:
            return Confirm(ok=False)          # not offered: nothing to ask
        return Elicit(f"For booking {pnr.upper()}: {policy.describe(option)}. Go ahead?", Confirm)

    @mcp.tool()
    async def choose_disruption_option(
        pnr: str, option_id: str,
        confirm: Annotated[ElicitationResult[Confirm], Resolve(ask_option)],
    ) -> dict:
        """Apply one option from disruption_options (e.g. "rebook:AC312", "refund", "voucher").
        The passenger is asked to confirm directly."""
        if not _approved(confirm):
            offered = actions.disruption_options(user_id, session_id, pnr)
            if not offered["ok"]:
                return offered
            if option_id not in {o["id"] for o in offered["options"]}:
                return {"ok": False, "reason": "option_unavailable",
                        "offered": [o["id"] for o in offered["options"]]}
            return {"ok": False, "reason": "not_approved", "note": "Nothing was changed."}
        proposal = actions.propose_option(user_id, session_id, pnr, option_id, source="mcp")
        return actions.confirm(user_id, session_id) if proposal["ok"] else proposal

    return mcp


def main():
    user_id = user_from_token(os.environ.get("ACME_MCP_TOKEN"))
    if not user_id:
        sys.exit("Set ACME_MCP_TOKEN to a valid token (e.g. demo-asha) - see src/auth.py")
    create_server(user_id).run()  # stdio


if __name__ == "__main__":
    main()
