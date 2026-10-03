#!/usr/bin/env python3
"""Run the app on a fixed interval, forever. This is the container's PID 1.

Why a loop and not cron: the container is then an ordinary long-running
service. Docker's `restart: unless-stopped` restarts it if it dies, `docker
compose logs` has everything, and there is no host-side schedule to maintain
per app. One process, one job, one log stream.

Environment:
  LOOP_INTERVAL   seconds between runs (default 3600)
  LOOP_JITTER     random extra 0..N seconds per run so several apps do not
                  hammer the same API at the same instant (default 60)
  LOOP_ONCE       set to 1 to run a single pass and exit (handy for testing)
  RUN_ARGS        extra CLI args passed to the app, space separated

A failing run is logged with its traceback and the loop continues. Only a
failure to even import the app is fatal (that is a packaging bug, and a
crash-loop is the right signal for it).
"""

from __future__ import annotations

import logging
import os
import random
import shlex
import signal
import sys
import time
import traceback

# ---- change this one line per app ------------------------------------------
from birdlisten import main as app_main  # noqa: E402
# ----------------------------------------------------------------------------
# birdlisten-specific: optional HTTP collage server, on only when SERVE_PORT
# is set. Daemon thread, so it never keeps the process alive or changes rc.
from serve import start_from_env as app_start_server  # noqa: E402

log = logging.getLogger("loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

_stop = False


def _on_signal(signum, _frame):
    global _stop
    log.info("received %s, will stop after this run", signal.Signals(signum).name)
    _stop = True


def _report(rc, started: float) -> None:
    """Non-zero is logged at ERROR so the fleet health check's grep sees it."""
    took = time.monotonic() - started
    if rc in (0, None):
        log.info("run finished rc=%s in %.1fs", rc, took)
    else:
        log.error("run finished with ERROR rc=%s in %.1fs", rc, took)


def main() -> int:
    interval = max(int(os.environ.get("LOOP_INTERVAL", "3600")), 1)
    jitter = max(int(os.environ.get("LOOP_JITTER", "60")), 0)
    once = os.environ.get("LOOP_ONCE", "") == "1"
    run_args = shlex.split(os.environ.get("RUN_ARGS", ""))

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info("starting: interval=%ss jitter=%ss once=%s args=%s", interval, jitter, once, run_args)
    app_start_server()  # no-op when SERVE_PORT is unset
    while True:
        started = time.monotonic()
        # The app's main() parses sys.argv like a CLI would, so hand it the args
        # the same way a shell would.
        sys.argv = [sys.argv[0], *run_args]
        try:
            rc = app_main()
            _report(rc, started)
        except SystemExit as exc:  # argparse / sys.exit inside the app
            _report(exc.code, started)
        except Exception:  # noqa: BLE001 -- log and keep the loop alive
            log.error("run failed after %.1fs:\n%s", time.monotonic() - started, traceback.format_exc())

        if once or _stop:
            return 0

        delay = interval + random.randint(0, max(jitter, 0))
        log.info("next run in %ss", delay)
        # Sleep in small steps so SIGTERM stops us within a second, not an hour.
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not _stop:
            time.sleep(1)
        if _stop:
            return 0


if __name__ == "__main__":
    sys.exit(main())
