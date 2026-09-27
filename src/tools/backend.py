"""Day 9: mock airline backend - the 'real systems' the LLM never touches directly.

Nothing outside src/actions.py should call these functions: the tool layer there adds
ownership checks, confirmation, expiry and the audit trail around them.
"""

# owner = the user_id that may see and change the booking (see src/auth.py)
FLIGHTS = {
    "ACX123": {"owner": "asha", "route": "Pune -> Delhi", "scheduled": "14:30",
               "status": "Delayed by 5 hours", "fare": 6500, "refundable": True},
    "ACX456": {"owner": "asha", "route": "Mumbai -> Kolkata", "scheduled": "09:15",
               "status": "On time", "fare": 4800, "refundable": False},
    "ACX789": {"owner": "ravi", "route": "Hyderabad -> Bengaluru", "scheduled": "18:00",
               "status": "Cancelled by airline", "fare": 3200, "refundable": True},
}


def _public(pnr, booking):
    """Booking fields safe to show the customer (no internal owner id)."""
    return {"pnr": pnr, **{k: v for k, v in booking.items() if k != "owner"}}


def get_flight_status(pnr: str) -> dict:
    booking = FLIGHTS.get(pnr.upper())
    if not booking:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    return {"ok": True, **_public(pnr.upper(), booking)}


def cancel_ticket(pnr: str) -> dict:
    booking = FLIGHTS.get(pnr.upper())
    if not booking:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    if booking["status"] == "Cancelled by passenger":
        # Idempotent: a retry or a double-clicked YES must not refund twice.
        return {"ok": True, "pnr": pnr.upper(), "cancelled": True, "already_cancelled": True,
                "refund_amount": 0, "note": "Booking was already cancelled; no further refund"}
    if booking["refundable"] or booking["status"] == "Cancelled by airline":
        refund = booking["fare"]
        note = "Refund to original payment method within 7 business days"
    else:
        refund = 0
        note = "Non-refundable fare: a travel credit voucher will be issued instead"
    booking["status"] = "Cancelled by passenger"
    return {"ok": True, "pnr": pnr.upper(), "cancelled": True,
            "refund_amount": refund, "note": note}
