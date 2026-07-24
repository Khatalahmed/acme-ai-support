"""Day 9: mock airline backend - the 'real systems' the LLM never touches directly."""

FLIGHTS = {
    "ACX123": {"route": "Pune -> Delhi", "scheduled": "14:30",
               "status": "Delayed by 5 hours", "fare": 6500, "refundable": True},
    "ACX456": {"route": "Mumbai -> Kolkata", "scheduled": "09:15",
               "status": "On time", "fare": 4800, "refundable": False},
    "ACX789": {"route": "Hyderabad -> Bengaluru", "scheduled": "18:00",
               "status": "Cancelled by airline", "fare": 3200, "refundable": True},
}


def get_flight_status(pnr: str) -> dict:
    booking = FLIGHTS.get(pnr.upper())
    if not booking:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    return {"ok": True, "pnr": pnr.upper(), **booking}


def cancel_ticket(pnr: str) -> dict:
    booking = FLIGHTS.get(pnr.upper())
    if not booking:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    if booking["refundable"] or booking["status"] == "Cancelled by airline":
        refund = booking["fare"]
        note = "Refund to original payment method within 7 business days"
    else:
        refund = 0
        note = "Non-refundable fare: a travel credit voucher will be issued instead"
    booking["status"] = "Cancelled by passenger"
    return {"ok": True, "pnr": pnr.upper(), "cancelled": True,
            "refund_amount": refund, "note": note}