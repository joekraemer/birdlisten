#!/usr/bin/env python3
"""Container healthcheck (Dockerfile HEALTHCHECK, issues #27 and #32). Exit 0
healthy, 1 unhealthy; Docker then marks the container `unhealthy` and the
fleet's bin/health.sh reports it.

It runs two checks, and both must pass:

1. HTTP liveness of the collage server. GET /api/recent?hours=1 must answer
   200 within the timeout. That route reads SQLite, so a wedged server, a dead
   server thread or a missing / corrupt database all fail it. With SERVE_PORT
   unset the server is off by design, so this check is skipped.

2. Liveness of the analysis worker (stream mode). The worker touches
   DATA_DIR/heartbeat on every loop iteration: at least every 0.5 s when
   idle, and after each clip when busy (a clip takes a second or two). That
   includes the time the cameras are down or retrying, because the worker
   just idles then. So a camera outage never fails this check; the timing
   lines and `cameras_up` report that. What fails it is a worker that is hung
   (stuck in an analysis or a db write) or gone, with no beat for
   HEARTBEAT_MAX_AGE_S seconds (default 300, 0 turns the check off).

   Start-up grace: a beat older than the container's main process (PID 1)
   is from an earlier run. Until the first beat of this run, the check passes
   for HEARTBEAT_MAX_AGE_S seconds after PID 1 started (model load and db
   open come first). After that, a missing beat fails. When the start time
   cannot be read (no /proc, i.e. not Linux), a missing beat passes.

   Skipped with CAPTURE_MODE=roundrobin (loop.py sleeps LOOP_INTERVAL between
   passes, and there is no worker thread) and when DATA_DIR does not exist
   (no pipeline here to check).

Stdlib only and no app imports, so each run costs one short Python start, a
few stat() calls and two small /proc reads.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

TIMEOUT_S = 8                  # under the HEALTHCHECK --timeout (10 s)
HEARTBEAT_FILE = "heartbeat"   # in DATA_DIR; stream.Worker touches it
HEARTBEAT_MAX_AGE_S = 300      # default; the worker beats every <= 0.5 s idle, ~1-2 s per clip


def check_http(env=os.environ, timeout: float = TIMEOUT_S) -> tuple[bool, str]:
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


def main_started_at(proc: str = "/proc") -> float | None:
    """Wall-clock start of PID 1 (the container's loop.py), or None off Linux."""
    try:
        with open(f"{proc}/1/stat") as f:
            stat = f.read()
        with open(f"{proc}/uptime") as f:
            uptime = float(f.read().split()[0])
        # Field 22 (starttime, clock ticks after boot). The command name in
        # field 2 may contain spaces, so count from its closing paren.
        ticks = int(stat.rsplit(")", 1)[1].split()[19])
        return time.time() - uptime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _max_age(env) -> tuple[float, str]:
    raw = env.get("HEARTBEAT_MAX_AGE_S", "").strip()
    if not raw:
        return HEARTBEAT_MAX_AGE_S, ""
    try:
        v = float(raw)
        if v >= 0:
            return v, ""
    except ValueError:
        pass
    return HEARTBEAT_MAX_AGE_S, f" (bad HEARTBEAT_MAX_AGE_S={raw!r}, using {HEARTBEAT_MAX_AGE_S})"


def check_heartbeat(env=os.environ, now: float | None = None,
                    started_at: float | None = None) -> tuple[bool, str]:
    max_age, note = _max_age(env)
    if max_age == 0:
        return True, "heartbeat check off (HEARTBEAT_MAX_AGE_S=0)"
    if env.get("CAPTURE_MODE", "").strip().lower() == "roundrobin":
        return True, "roundrobin mode: no worker heartbeat to check"
    data_dir = Path(env.get("DATA_DIR", "/data"))
    if not data_dir.is_dir():
        return True, f"{data_dir} missing: no pipeline to check"
    now = time.time() if now is None else now
    path = data_dir / HEARTBEAT_FILE
    try:
        beat = path.stat().st_mtime
    except OSError:
        beat = None
    if beat is not None and started_at is not None and beat < started_at:
        beat = None                               # left by an earlier run
    if beat is None:
        if started_at is None:
            return True, f"no heartbeat yet at {path}, start time unknown{note}"
        up = now - started_at
        if up <= max_age:
            return True, f"starting: no heartbeat yet, {up:.0f}s since start (grace {max_age:g}s){note}"
        return False, f"no worker heartbeat at {path} {up:.0f}s after start (limit {max_age:g}s){note}"
    age = now - beat
    if age > max_age:
        return False, f"worker heartbeat {age:.0f}s old (limit {max_age:g}s): worker hung or dead{note}"
    return True, f"worker heartbeat {max(age, 0):.0f}s old{note}"


def check(env=os.environ, timeout: float = TIMEOUT_S, now: float | None = None,
          started_at: float | None = None, proc: str = "/proc") -> tuple[bool, str]:
    ok_http, msg_http = check_http(env, timeout)
    if started_at is None:
        started_at = main_started_at(proc)
    ok_beat, msg_beat = check_heartbeat(env, now, started_at)
    return ok_http and ok_beat, f"{msg_http}; {msg_beat}"


def main() -> int:
    ok, msg = check()
    print(msg)          # Docker keeps the last few outputs in `docker inspect`
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
