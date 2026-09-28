"""Simulated authentication: bearer token -> user_id.

A real deployment would verify a signed token (OIDC/JWT) from an identity provider. The rest
of the code only needs `user_id`, so swapping this for real auth changes nothing else.
"""

# Demo users. Asha owns ACX123 and ACX456; Ravi owns ACX789 (see tools/backend.py).
DEMO_TOKENS = {
    "demo-asha": "asha",
    "demo-ravi": "ravi",
}


def user_from_token(token):
    """user_id for a valid token, else None."""
    return DEMO_TOKENS.get(token)
