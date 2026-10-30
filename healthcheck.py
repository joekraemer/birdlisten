#!/usr/bin/env python3
"""Container healthcheck (Dockerfile HEALTHCHECK, issue #27). Exit 0 healthy,
1 unhealthy; Docker then marks the container `unhealthy` and the fleet's
bin/health.sh reports it.

What it checks: HTTP liveness of the collage server. GET /api/recent?hours=1
must answer 200 within the timeout. That route reads SQLite, so a wedged
server, a dead server thread or a missing / corrupt database all fail it.

What it does not check: the capture pipeline. There is no cheap, existing
signal for it (no heartbeat file; the timing summary is a log line). A dead
pipeline thread already makes main() return 1 and loop.py restart it. A
heartbeat file the worker touches per clip would be the follow-up.

With SERVE_PORT unset the server is off by design, so there is nothing to
probe and the check passes.

Stdlib only and no app imports, so each run costs one short Python start.
"""

from __future__ import annotations

import os
import sys
import urllib.request

TIMEOUT_S = 8      # under the HEALTHCHECK --timeout (10 s)


def check(env=os.environ, timeout: float = TIMEOUT_S) -> tuple[bool, str]:
    port = env.get("SERVE_PORT", "").strip()
    if not port:
        return True, "SERVE_PORT unset, collage server off: nothing to check"
    url = f"http://127.0.0.1:{port}/api/recent?hours=1"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read()
            if resp.status != 200:
                return False, f"{url}: HTTP {resp.status}"
    except Exception as exc:  # noqa: BLE001 -- any failure is unhealthy
        return False, f"{url}: {type(exc).__name__}: {exc}"
    return True, f"{url}: ok"


def main() -> int:
    ok, msg = check()
    print(msg)          # Docker keeps the last few outputs in `docker inspect`
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
