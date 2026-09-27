"""Disruption entitlements as code: what a passenger is OWED, decided without any LLM.

Pure functions over plain data - no database, no model, no network - so every rule is unit-
tested and a reviewer can check each line against the policy documents it cites:
  - refund-cancellation-policy.md  §4 airline-initiated, §5 weather/force-majeure cancellations
  - delay-compensation-policy.md   §2 delay tiers, §4 weather/ATC exclusion

The LLM only EXPLAINS these results to the passenger. It never computes or changes them.
"""

# The policy text overlaps at the edges ("2-4 hours" and "4-6 hours" both contain 4 h).
# Code has to choose; the choice lives here, in one place, for the business to confirm.
# Decision: lower bound inclusive -> exactly 4 h is the 4-6 h tier, exactly 6 h too.
MEAL_TIER_MIN = 120        # 2 h  <= delay < 4 h  -> meal voucher
CHOICE_TIER_MIN = 240      # 4 h  <= delay <= 6 h -> voucher OR rebooking
REFUND_TIER_MIN = 360      # delay > 6 h          -> refund OR rebooking + voucher

MEAL_VOUCHER = 400
DELAY_VOUCHER = 3000
LONG_DELAY_VOUCHER = 5000
NO_COMPENSATION_CAUSES = {"weather", "atc"}


def _rebook_options(alternatives, voucher=0):
    return [{"id": f"rebook:{a['flight']}", "kind": "rebook", "flight": a["flight"],
             "departure": a["departure"], "voucher": voucher} for a in alternatives]


def _refund(fare):
    return {"id": "refund", "kind": "refund", "amount": fare}


def entitlements(booking, flight, alternatives):
    """What this booking is owed for its flight's disruption.

    booking:      {"fare", "status", "resolution", ...}  (backend BOOKINGS row)
    flight:       {"status", "delay_min", "cause", ...}  (backend FLIGHTS row)
    alternatives: flight_view dicts of later same-route flights with free seats

    Returns {"disruption", "options": [...choose one...], "automatic": [...no choice needed...],
             "policy", "note"}.
    """
    result = {"disruption": None, "options": [], "automatic": [], "policy": None, "note": ""}

    if booking["status"] != "confirmed" or booking["resolution"]:
        result["note"] = "This booking's disruption has already been resolved."
        return result

    if flight["status"] == "cancelled":
        result["disruption"] = "cancelled"
        result["options"] = _rebook_options(alternatives) + [_refund(booking["fare"])]
        if flight["cause"] in NO_COMPENSATION_CAUSES:
            result["policy"] = "refund-cancellation-policy.md §5"
            result["note"] = ("Weather cancellation: free rebooking or a full refund; no meals "
                              "or hotel are provided.")
        else:
            result["policy"] = "refund-cancellation-policy.md §4"
            result["note"] = "Airline-initiated cancellation: free rebooking or a full refund."
        return result

    if flight["status"] != "delayed":
        return result

    delay = flight["delay_min"]
    result["disruption"] = "delayed"
    result["policy"] = "delay-compensation-policy.md §2"
    if flight["cause"] in NO_COMPENSATION_CAUSES:
        result["policy"] = "delay-compensation-policy.md §4"
        result["note"] = "Delays caused by weather or air traffic control are not compensated."
    elif delay < MEAL_TIER_MIN:
        result["note"] = "Delays under 2 hours are not compensated."
    elif delay < CHOICE_TIER_MIN:
        result["automatic"] = [{"id": "meal_voucher", "kind": "meal_voucher",
                                "amount": MEAL_VOUCHER}]
        result["note"] = "2-4 hour delay: a meal voucher at the departure airport."
    elif delay <= REFUND_TIER_MIN:
        result["options"] = ([{"id": "voucher", "kind": "voucher", "amount": DELAY_VOUCHER}]
                             + _rebook_options(alternatives))
        result["note"] = "4-6 hour delay: a travel voucher OR free rebooking."
    else:
        result["options"] = ([_refund(booking["fare"])]
                             + _rebook_options(alternatives, voucher=LONG_DELAY_VOUCHER))
        result["note"] = "Over 6 hours: a full refund OR free rebooking plus a travel voucher."
    return result


def describe(option):
    """One-line, customer-facing description of an option (deterministic, no LLM)."""
    if option["kind"] == "rebook":
        extra = f" plus a Rs {option['voucher']:,} travel voucher" if option["voucher"] else ""
        return f"Free rebooking onto {option['flight']} departing {option['departure']}{extra}"
    if option["kind"] == "refund":
        return f"Full refund of Rs {option['amount']:,} to your original payment method"
    if option["kind"] == "voucher":
        return f"A Rs {option['amount']:,} travel voucher (you keep your current flight)"
    if option["kind"] == "meal_voucher":
        return f"A Rs {option['amount']:,} meal voucher at the departure airport"
    raise ValueError(f"unknown option kind {option['kind']!r}")
