"""Abuse controls for the public demo: a per-visitor rate limit and a global daily cap.

Every chat message can spend real Azure/Jev credit, so a public URL needs both:
  RATE_LIMIT_PER_IP   messages per visitor per 10 minutes   (default 30)
  DAILY_REQUEST_CAP   messages per day across all visitors  (default 1000)

In memory, so the app must run as ONE replica (Container Apps: --max-replicas 1).
"""

import os
import threading
import time
from collections import defaultdict, deque

WINDOW_S = 600
_lock = threading.Lock()
_hits = defaultdict(deque)          # visitor -> timestamps within the window
_day = {"date": None, "count": 0}


def _limit(name, default):
    return int(os.environ.get(name) or default)


def client_ip(request):
    """The visitor's IP. Behind Azure's ingress, X-Forwarded-For is '<client-sent...>, <real>':
    a client can prepend fake entries, but the proxy APPENDS the address it saw - so trust the
    last entry only. Locally (no proxy) it's the socket address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def check(visitor, now=None):
    """None if allowed (and counted), else (reason, retry_after_seconds)."""
    now = now if now is not None else time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    with _lock:
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if _day["count"] >= _limit("DAILY_REQUEST_CAP", 1000):
            tomorrow = (int(now // 86400) + 1) * 86400
            return "daily demo limit reached", int(tomorrow - now)
        hits = _hits[visitor]
        while hits and hits[0] <= now - WINDOW_S:
            hits.popleft()
        if len(hits) >= _limit("RATE_LIMIT_PER_IP", 30):
            return "too many messages", int(hits[0] + WINDOW_S - now) + 1
        hits.append(now)
        _day["count"] += 1
        return None


def reset():
    """Forget all counts (tests)."""
    with _lock:
        _hits.clear()
        _day.update(date=None, count=0)
