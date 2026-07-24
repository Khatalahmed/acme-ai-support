"""Day 8 - Stage 1: generate ACME's official policy documents (the RAG canon)."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "policies"

# filename -> (title, facts that MUST appear verbatim-ish; these numbers ARE the canon)
POLICIES = {
    "refund-cancellation-policy.md": (
        "Refund and Cancellation Policy",
        [
            "Fully-flexible fares: 100% refund, processed within 7 business days",
            "Non-refundable fares: taxes refunded; travel credit voucher valid 12 months",
            "Airline-initiated cancellation: full refund OR free rebooking on next available flight",
            "Weather cancellations: no monetary compensation, but free rebooking or full refund",
            "Cancellation within 24 hours of booking: full refund regardless of fare type",
        ],
    ),
    "delay-compensation-policy.md": (
        "Flight Delay Compensation Policy",
        [
            "Delay 2-4 hours: meal vouchers worth Rs 400",
            "Delay 4-6 hours: Rs 3,000 travel voucher OR free rebooking",
            "Delay over 6 hours: full refund OR rebooking plus Rs 5,000 travel voucher",
            "Overnight delay: hotel accommodation provided by the airline",
            "Compensation payouts are processed within 7 business days",
            "No compensation for delays caused by weather or air traffic control",
        ],
    ),
    "baggage-policy.md": (
        "Baggage Policy",
        [
            "Checked baggage allowance: 23 kg per passenger (domestic)",
            "Cabin baggage: 7 kg, one piece",
            "Excess baggage: Rs 500 per kg",
            "Delayed baggage: report within 4 hours of arrival; Rs 1,000 per day interim compensation up to 5 days",
            "Lost baggage (untraced after 21 days): compensation up to Rs 20,000",
            "Damaged baggage: file report at airport desk within 7 days with photos",
        ],
    ),
    "loyalty-program-policy.md": (
        "AcmeMiles Loyalty Program Policy",
        [
            "Earn 5 AcmeMiles per Rs 100 spent on base fare",
            "Miles expire after 24 months of account inactivity",
            "Missing miles: claim within 90 days of travel with boarding pass",
            "Tier levels: Silver (20,000 miles/yr), Gold (50,000), Platinum (100,000)",
        ],
    ),
    "special-assistance-policy.md": (
        "Special Assistance Policy",
        [
            "Wheelchair assistance: free, request at least 48 hours before departure",
            "Unaccompanied minors (5-12 years): allowed with Rs 3,000 service fee",
            "Pregnant passengers: fit-to-fly certificate required after 28 weeks",
            "Medical equipment: carried free, does not count toward baggage allowance",
        ],
    ),
}

PROMPT = """Write the official policy document for ACME Bharat Airlines: "{title}".

It MUST contain ALL of these provisions, stated clearly (keep every number exactly as given):
{facts}

Format: Markdown with a # title, ## section headings, and short clear paragraphs or bullet
lists under each. Formal but plain airline-policy language. 400-700 words.
Output ONLY the markdown document, nothing else.
"""


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    model = os.environ.get("VERTEX_MODEL", "gemini-3.5-flash")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, (title, facts) in POLICIES.items():
        prompt = PROMPT.format(title=title, facts="\n".join(f"- {f}" for f in facts))
        resp = client.models.generate_content(model=model, contents=prompt)
        (OUT_DIR / filename).write_text(resp.text.strip(), encoding="utf-8")
        print(f"wrote {filename} ({len(resp.text)} chars)")

    print(f"\nCanon complete -> {OUT_DIR}")


if __name__ == "__main__":
    main()