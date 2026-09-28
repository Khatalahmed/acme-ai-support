"""Mock airline backend - the 'real systems' the LLM never touches directly.

Nothing outside src/actions.py may call the write functions (cancel_ticket, rebook, refund,
issue_voucher): the tool layer there adds ownership checks, confirmation, expiry and the audit
trail around them. Every write is idempotent at this level too - a booking can be cancelled
once and a disruption resolved once.
"""

CITIES = {"PNQ": "Pune", "DEL": "Delhi", "BOM": "Mumbai", "CCU": "Kolkata", "HYD": "Hyderabad",
          "BLR": "Bengaluru", "GOI": "Goa", "MAA": "Chennai", "GAU": "Guwahati"}

# status: scheduled | delayed | cancelled. cause: why a disruption happened (drives policy).
# One day of ACME operations, chosen so every tier of the disruption policy has a case.
FLIGHTS = {
    # 5 h technical delay -> 4-6 h tier: Rs 3,000 voucher OR free rebooking
    "AC101": {"origin": "PNQ", "dest": "DEL", "dep": "2026-10-02 14:30", "status": "delayed",
              "delay_min": 300, "cause": "technical", "seats": 20},
    "AC103": {"origin": "PNQ", "dest": "DEL", "dep": "2026-10-02 19:45", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 6},
    "AC105": {"origin": "PNQ", "dest": "DEL", "dep": "2026-10-03 07:10", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 0},            # full: never offered
    # on time
    "AC205": {"origin": "BOM", "dest": "CCU", "dep": "2026-10-02 09:15", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 15},
    # airline-initiated cancellation -> full refund OR free rebooking
    "AC310": {"origin": "HYD", "dest": "BLR", "dep": "2026-10-02 18:00", "status": "cancelled",
              "delay_min": 0, "cause": "technical", "seats": 0},
    "AC312": {"origin": "HYD", "dest": "BLR", "dep": "2026-10-02 20:30", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 4},
    "AC316": {"origin": "HYD", "dest": "BLR", "dep": "2026-10-02 21:45", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 0},            # full: never offered
    "AC314": {"origin": "HYD", "dest": "BLR", "dep": "2026-10-03 07:00", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 12},
    # weather cancellation -> rebooking or refund, but no meals/hotel
    "AC420": {"origin": "DEL", "dest": "GOI", "dep": "2026-10-02 11:00", "status": "cancelled",
              "delay_min": 0, "cause": "weather", "seats": 0},
    "AC422": {"origin": "DEL", "dest": "GOI", "dep": "2026-10-02 16:00", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 3},
    # 2.5 h delay -> Rs 400 meal voucher, nothing to choose
    "AC501": {"origin": "BLR", "dest": "MAA", "dep": "2026-10-02 10:00", "status": "delayed",
              "delay_min": 150, "cause": "technical", "seats": 9},
    # 7 h weather delay -> excluded from compensation
    "AC610": {"origin": "CCU", "dest": "GAU", "dep": "2026-10-02 12:00", "status": "delayed",
              "delay_min": 420, "cause": "weather", "seats": 11},
    # 6 h 40 m technical delay -> over-6 h tier: full refund OR rebooking + Rs 5,000 voucher
    "AC702": {"origin": "MAA", "dest": "DEL", "dep": "2026-10-02 08:00", "status": "delayed",
              "delay_min": 400, "cause": "technical", "seats": 30},
    "AC704": {"origin": "MAA", "dest": "DEL", "dep": "2026-10-02 15:00", "status": "scheduled",
              "delay_min": 0, "cause": None, "seats": 5},
}

# owner = the user_id that may see and change the booking (see src/auth.py).
# status: confirmed | cancelled. resolution: set once a disruption option has been applied.
BOOKINGS = {
    "ACX123": {"owner": "asha", "flight": "AC101", "fare": 6500, "refundable": True},
    "ACX456": {"owner": "asha", "flight": "AC205", "fare": 4800, "refundable": False},
    "ACX789": {"owner": "ravi", "flight": "AC310", "fare": 3200, "refundable": True},
    "ACX654": {"owner": "ravi", "flight": "AC420", "fare": 5400, "refundable": False},
    "ACX321": {"owner": "asha", "flight": "AC501", "fare": 3900, "refundable": True},
    "ACX987": {"owner": "ravi", "flight": "AC610", "fare": 4100, "refundable": False},
    "ACX246": {"owner": "asha", "flight": "AC702", "fare": 7200, "refundable": False},
}
for _b in BOOKINGS.values():
    _b.update(status="confirmed", resolution=None, vouchers=[])


def display_status(flight):
    if flight["status"] == "cancelled":
        return "Cancelled by airline"
    if flight["status"] == "delayed":
        h, m = divmod(flight["delay_min"], 60)
        return f"Delayed by {h} hours" + (f" {m} minutes" if m else "")
    return "On time"


def flight_view(flight_no):
    f = FLIGHTS[flight_no]
    return {"flight": flight_no, "route": f"{CITIES[f['origin']]} -> {CITIES[f['dest']]}",
            "departure": f["dep"], "scheduled": f["dep"][-5:], "status": display_status(f),
            "seats_available": f["seats"],
            # raw disruption facts: which policy applies depends on these, not on the wording
            "flight_status": f["status"], "delay_min": f["delay_min"], "cause": f["cause"]}


def get_flight_status(pnr: str) -> dict:
    """Customer-safe view of a booking and its flight (no owner id)."""
    b = BOOKINGS.get(pnr.upper())
    if not b:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    view = flight_view(b["flight"])
    if b["status"] == "cancelled":
        view["status"] = "Cancelled by passenger"
    return {"ok": True, "pnr": pnr.upper(), **view, "fare": b["fare"],
            "refundable": b["refundable"], "booking_status": b["status"],
            "resolution": b["resolution"], "vouchers": list(b["vouchers"])}


def find_alternatives(flight_no, limit=3):
    """Later flights on the same route that are operating and have a free seat."""
    f = FLIGHTS[flight_no]
    alts = [(no, x) for no, x in FLIGHTS.items()
            if no != flight_no and x["origin"] == f["origin"] and x["dest"] == f["dest"]
            and x["status"] == "scheduled" and x["seats"] > 0 and x["dep"] > f["dep"]]
    return [flight_view(no) for no, _ in sorted(alts, key=lambda a: a[1]["dep"])[:limit]]


# ---------------------------------------------------------------- writes (via actions.py only)

def cancel_ticket(pnr: str) -> dict:
    """Voluntary cancellation by the passenger."""
    b = BOOKINGS.get(pnr.upper())
    if not b:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    if b["status"] == "cancelled":
        # Idempotent: a retry or a double-clicked YES must not refund twice.
        return {"ok": True, "pnr": pnr.upper(), "cancelled": True, "already_cancelled": True,
                "refund_amount": 0, "note": "Booking was already cancelled; no further refund"}
    if b["refundable"] or FLIGHTS[b["flight"]]["status"] == "cancelled":
        refund, note = b["fare"], "Refund to original payment method within 7 business days"
    else:
        refund, note = 0, "Non-refundable fare: a travel credit voucher will be issued instead"
    b["status"] = "cancelled"
    return {"ok": True, "pnr": pnr.upper(), "cancelled": True, "refund_amount": refund, "note": note}


def _resolve(pnr, resolution):
    """Mark a disruption as resolved; returns an error dict if it already was."""
    b = BOOKINGS.get(pnr.upper())
    if not b:
        return {"ok": False, "error": f"No booking found for PNR {pnr.upper()}"}
    if b["resolution"] or b["status"] == "cancelled":
        return {"ok": False, "error": f"Booking {pnr.upper()} is already resolved",
                "already_resolved": True}
    b["resolution"] = resolution
    return None


def rebook(pnr: str, flight_no: str, voucher: int = 0) -> dict:
    """Move a booking to another flight, taking one seat; optionally add a travel voucher."""
    target = FLIGHTS.get(flight_no)
    if not target or target["seats"] <= 0 or target["status"] != "scheduled":
        return {"ok": False, "error": f"No seat available on {flight_no}", "no_seat": True}
    err = _resolve(pnr, {"kind": "rebook", "flight": flight_no, "voucher": voucher})
    if err:
        return err
    b = BOOKINGS[pnr.upper()]
    old, b["flight"] = b["flight"], flight_no
    target["seats"] -= 1
    if voucher:
        b["vouchers"].append(voucher)
    return {"ok": True, "pnr": pnr.upper(), "rebooked_from": old, **flight_view(flight_no),
            "voucher_amount": voucher}


def refund(pnr: str, amount: int) -> dict:
    """Refund the fare for a disrupted flight; the booking is then cancelled."""
    err = _resolve(pnr, {"kind": "refund", "amount": amount})
    if err:
        return err
    BOOKINGS[pnr.upper()]["status"] = "cancelled"
    return {"ok": True, "pnr": pnr.upper(), "refund_amount": amount,
            "note": "Refund to original payment method within 7 business days"}


def issue_voucher(pnr: str, amount: int) -> dict:
    """Travel voucher instead of rebooking; the passenger keeps their (delayed) flight."""
    err = _resolve(pnr, {"kind": "voucher", "amount": amount})
    if err:
        return err
    BOOKINGS[pnr.upper()]["vouchers"].append(amount)
    return {"ok": True, "pnr": pnr.upper(), "voucher_amount": amount,
            "note": "Travel voucher valid for future travel with ACME Bharat Airlines"}
