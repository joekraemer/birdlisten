#!/usr/bin/env python3
"""Shared audio listener (issue #40): one camera connection, many consumers.

The listener owns the RTSP connections. It reuses stream.py's CameraSupervisor
(one long-lived ffmpeg per camera, backoff, stall detection, segment
validation) and hands every finished segment to each consumer's inbox with a
hard link:

  AUDIO_DIR/segments/<NN-cam>/            ffmpeg writes here (stream.py layout)
  AUDIO_DIR/segments/queued/              a finished segment is renamed here,
                                          then linked into every inbox and unlinked
  AUDIO_DIR/inbox/<consumer>/<NN-cam>_r<run>_<UTC start>Z.wav
  AUDIO_DIR/status.json                   cameras and inboxes, rewritten every 10 s

A consumer reads its own inbox and deletes each file when it is done with it.
The audio is freed by the filesystem when the last consumer deletes its link,
so "delete once every app is done" needs no acknowledgement protocol. A link
appears atomically and points at a complete segment, so a consumer never sees
a partial file. Every inbox is capped by age and size (oldest dropped first),
so a stopped consumer cannot fill the volume or hold up the others.

All of AUDIO_DIR must be one filesystem (hard links), i.e. one volume shared
by the listener and its consumers, and the consumers must be able to delete
from their inbox (same uid as this image's `app` user, 10001).

Environment:
  CAMERAS          required. name=rtsp-url pairs, as for birdlisten. The
                   listener does not warn about main-stream URLs: a camera
                   whose sub-stream is broken can use its main stream here,
                   and only this one connection pays for the video.
  CONSUMERS        required. Comma-separated inbox names, e.g.
                   birdlisten,transcriber ([a-z][a-z0-9_-]*, at most 32 chars)
  AUDIO_DIR        shared volume root (default /audio), absolute
  CLIP_SECONDS     seconds per segment (default 30, 3..600)
  INBOX_MAX_AGE_S  drop inbox files whose audio started longer ago (default 3600)
  INBOX_MAX_MB     per-inbox size cap, oldest dropped first (default 512)
  CAMERA_STALE_S   healthcheck: a camera with no segment for this long makes
                   the container unhealthy (default 300)

Segments are 48 kHz mono s16 WAV, exactly what stream mode analyzes today
(about 2.9 MB per 30 s clip).

Usage:
  python listener.py                 run until SIGTERM/SIGINT
  python listener.py --healthcheck   exit 0 healthy, 1 unhealthy (compose healthcheck)
  python listener.py --selftest      real ffmpeg + hard-link fan-out against a test tone
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import birdlisten as bl
import stream

log = logging.getLogger("birdlisten.listener")

STATUS_FILE = "status.json"
STATUS_VERSION = 1
CONSUMER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
# An inbox file: <camera dir>_<segment name>, the same name stream.py queues
# (and that AUDIO_SOURCE=inbox reads back).
INBOX_RE = stream.QUEUED_SPLIT_RE


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class ListenerConfig:
    cameras: tuple[bl.Camera, ...]
    consumers: tuple[str, ...]
    audio_dir: Path = Path("/audio")
    clip_seconds: int = 30
    inbox_max_age_s: int = 3600
    inbox_max_bytes: int = 512 * 1024 * 1024
    camera_stale_s: int = 300

    @property
    def segment_dir(self) -> Path:
        return self.audio_dir / "segments"

    def inbox(self, consumer: str) -> Path:
        return self.audio_dir / "inbox" / consumer


def _int(env, name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise bl.ConfigError(f"{name} must be a whole number, got {raw!r}") from None
    if not lo <= value <= hi:
        raise bl.ConfigError(f"{name} must be {lo}..{hi}, got {value}")
    return value


def parse_consumers(raw: str) -> tuple[str, ...]:
    names = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not names:
        raise bl.ConfigError("CONSUMERS is empty")
    for n in names:
        if not CONSUMER_RE.match(n):
            raise bl.ConfigError(f"CONSUMERS entry {n!r} must match {CONSUMER_RE.pattern}")
    if len(set(names)) != len(names):
        raise bl.ConfigError("CONSUMERS names must be unique")
    return names


def load_config(env=os.environ) -> ListenerConfig:
    cams = env.get("CAMERAS", "").strip()
    if not cams:
        raise bl.ConfigError("CAMERAS is required")
    cons = env.get("CONSUMERS", "").strip()
    if not cons:
        raise bl.ConfigError("CONSUMERS is required")
    raw_dir = env.get("AUDIO_DIR", "").strip() or "/audio"
    audio_dir = Path(raw_dir)
    if not audio_dir.is_absolute():
        raise bl.ConfigError("AUDIO_DIR must be an absolute path")
    if "%" in raw_dir:
        raise bl.ConfigError("AUDIO_DIR must not contain '%'")    # ffmpeg -strftime pattern
    return ListenerConfig(
        cameras=bl.parse_cameras(cams),
        consumers=parse_consumers(cons),
        audio_dir=audio_dir,
        clip_seconds=_int(env, "CLIP_SECONDS", 30, 3, 600),
        inbox_max_age_s=_int(env, "INBOX_MAX_AGE_S", 3600, 60, 7 * 86400),
        inbox_max_bytes=_int(env, "INBOX_MAX_MB", 512, 1, 100_000) * 1024 * 1024,
        camera_stale_s=_int(env, "CAMERA_STALE_S", 300, 30, 86400),
    )


# ----------------------------------------------------------------- inbox names
def inbox_name(camera_dir: str, segment_name: str) -> str:
    return f"{camera_dir}_{segment_name}"


def parse_inbox_name(name: str) -> tuple[str, dt.datetime] | None:
    """'01-garage_r0001_20261117T000238Z.wav' -> ('01-garage', aware UTC start), else None."""
    m = INBOX_RE.match(name)
    if not m:
        return None
    return m.group(1), stream.parse_start_utc(m.group(2))


def camera_of(camera_dir: str) -> str:
    """'01-west-rooftop' -> 'west-rooftop' (the sanitised camera name)."""
    return camera_dir.split("-", 1)[1]


def _iso(t: dt.datetime | None) -> str | None:
    return None if t is None else t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


# ----------------------------------------------------------------- fan-out
@dataclass
class CameraState:
    dir: str
    segments: int = 0
    last_segment_utc: dt.datetime | None = None


class Fanout:
    """Stands in for stream.ClipQueue: CameraSupervisor only calls put(clip).
    Links the queued segment into every inbox, then unlinks the queued name.
    Called from every supervisor thread at once."""

    def __init__(self, inboxes: dict[str, Path], camera_dirs: dict[str, str]):
        self.inboxes = inboxes
        self._lock = threading.Lock()
        self.cameras = {name: CameraState(d) for name, d in camera_dirs.items()}
        self.link_failures = 0

    def put(self, clip: stream.Clip) -> None:
        src = clip.path
        linked = 0
        for consumer, inbox in self.inboxes.items():
            try:
                os.link(src, inbox / src.name)
                linked += 1
            except FileExistsError:                   # same name already waiting: keep the first
                log.warning("%s: %s already in inbox %s, not replaced", clip.camera, src.name, consumer)
            except OSError as exc:
                log.warning("%s: could not link %s into inbox %s: %s",
                            clip.camera, src.name, consumer, bl.scrub(str(exc)))
                with self._lock:
                    self.link_failures += 1
        stream.safe_unlink(src)
        with self._lock:
            st = self.cameras[clip.camera]
            st.segments += 1
            st.last_segment_utc = clip.ready_utc
        if not linked:
            log.warning("%s: segment %s reached no inbox", clip.camera, src.name)

    def snapshot(self) -> dict[str, CameraState]:
        with self._lock:
            return {n: CameraState(s.dir, s.segments, s.last_segment_utc) for n, s in self.cameras.items()}


# ----------------------------------------------------------------- inbox caps
@dataclass
class InboxState:
    files: int = 0
    bytes: int = 0
    oldest_utc: dt.datetime | None = None
    pruned_age: int = 0                   # totals since start
    pruned_size: int = 0
    _unreported: int = field(default=0, repr=False)
    _last_warn: float | None = field(default=None, repr=False)


def prune_inbox(inbox: Path, *, max_age_s: float, max_bytes: int, now_utc: dt.datetime) -> tuple[int, int, int, int, dt.datetime | None]:
    """Drop files older than max_age_s, then the oldest until the inbox fits in
    max_bytes. Only inbox-pattern files count or are touched. Returns
    (pruned_by_age, pruned_by_size, files_left, bytes_left, oldest_left)."""
    entries = []
    for name in os.listdir(inbox):
        parsed = parse_inbox_name(name)
        if parsed is None:
            continue
        path = inbox / name
        try:
            size = path.stat().st_size
        except FileNotFoundError:                     # the consumer just took it
            continue
        entries.append((parsed[1], name, path, size))
    entries.sort()
    cutoff = now_utc - dt.timedelta(seconds=max_age_s)
    by_age = by_size = 0
    kept = []
    for e in entries:
        if e[0] < cutoff:
            by_age += stream.safe_unlink(e[2])
        else:
            kept.append(e)
    total = sum(e[3] for e in kept)
    while kept and total > max_bytes:
        e = kept.pop(0)
        by_size += stream.safe_unlink(e[2])
        total -= e[3]
    return by_age, by_size, len(kept), total, (kept[0][0] if kept else None)


# ----------------------------------------------------------------- status
def build_status(*, started_utc: dt.datetime, now_utc: dt.datetime, cfg: ListenerConfig,
                 cameras: dict[str, CameraState], supervisors: dict, inboxes: dict[str, InboxState]) -> dict:
    return {
        "version": STATUS_VERSION,
        "written_utc": _iso(now_utc),
        "started_utc": _iso(started_utc),
        "clip_seconds": cfg.clip_seconds,
        "cameras": {
            name: {"dir": st.dir,
                   "up": bool(getattr(supervisors.get(name), "up", False)),
                   "fails": int(getattr(supervisors.get(name), "fails", 0)),
                   "segments": st.segments,
                   "last_segment_utc": _iso(st.last_segment_utc)}
            for name, st in cameras.items()},
        "inboxes": {
            name: {"files": s.files, "bytes": s.bytes, "oldest_utc": _iso(s.oldest_utc),
                   "pruned_age": s.pruned_age, "pruned_size": s.pruned_size}
            for name, s in inboxes.items()},
    }


def write_status(audio_dir: Path, status: dict) -> None:
    """Atomic: consumers and the healthcheck never read a half-written file."""
    fd, tmp = tempfile.mkstemp(prefix=".status-", dir=audio_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=1, sort_keys=False)
        os.replace(tmp, audio_dir / STATUS_FILE)
    except BaseException:
        stream.safe_unlink(tmp)
        raise


def evaluate(status: dict, now_utc: dt.datetime, *, camera_stale_s: float, status_max_age_s: float = 60) -> list[str]:
    """Problems that make the listener unhealthy; [] = healthy. A camera that
    has never produced a segment is measured from the listener's start."""
    problems = []
    written = _from_iso(status.get("written_utc"))
    if written is None or (now_utc - written).total_seconds() > status_max_age_s:
        problems.append(f"status not updated since {status.get('written_utc')}")
    started = _from_iso(status.get("started_utc"))
    for name, cam in sorted(status.get("cameras", {}).items()):
        last = _from_iso(cam.get("last_segment_utc")) or started
        if last is None:
            continue
        age = (now_utc - last).total_seconds()
        if age > camera_stale_s:
            what = "no segment" if cam.get("last_segment_utc") else "no segment since start"
            problems.append(f"{name}: {what} for {age:.0f}s (fails={cam.get('fails', 0)})")
    return problems


def healthcheck(env=os.environ, now_utc: dt.datetime | None = None, started_at: float | None = None) -> tuple[bool, str]:
    """Reads only AUDIO_DIR and CAMERA_STALE_S, so a config problem elsewhere
    shows up in the logs, not as a healthcheck crash."""
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    audio_dir = Path(env.get("AUDIO_DIR", "").strip() or "/audio")
    try:
        stale = _int(env, "CAMERA_STALE_S", 300, 30, 86400)
    except bl.ConfigError as exc:
        return False, str(exc)
    path = audio_dir / STATUS_FILE
    try:
        status = json.loads(path.read_text())
    except FileNotFoundError:
        if started_at is None:
            import healthcheck as hc                  # stdlib only
            started_at = hc.main_started_at()
        if started_at is not None and now_utc.timestamp() - started_at < stale:
            return True, f"{path} not written yet (start-up grace)"
        return False, f"{path} missing"
    except (OSError, ValueError) as exc:
        return False, f"{path}: {type(exc).__name__}: {exc}"
    problems = evaluate(status, now_utc, camera_stale_s=stale)
    if problems:
        return False, "; ".join(problems)
    cams = status.get("cameras", {})
    return True, f"{len(cams)} cameras fresh"


# ----------------------------------------------------------------- run
def prepare(cfg: ListenerConfig) -> dict[str, Path]:
    """Create the volume layout, clear segments a crashed run left behind, and
    return the inboxes. Inbox files are kept: consumers pick them up after a
    restart. OSError is left to the caller."""
    cfg.audio_dir.mkdir(parents=True, exist_ok=True)
    peak = (len(cfg.cameras) * 2 + 1) * (cfg.clip_seconds * 96000 + 44)
    stream.prepare_segment_dir(cfg.segment_dir, peak_bytes=peak)
    stream.cleanup_stale(cfg.segment_dir)
    inboxes = {}
    for c in cfg.consumers:
        p = cfg.inbox(c)
        p.mkdir(parents=True, exist_ok=True)
        inboxes[c] = p
    root = cfg.audio_dir / "inbox"
    extra = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name not in inboxes)
    if extra:
        log.warning("inboxes not in CONSUMERS are left alone and not capped: %s", ", ".join(extra))
    if os.stat(cfg.segment_dir).st_dev != os.stat(root).st_dev:
        raise OSError(f"{cfg.segment_dir} and {root} are on different filesystems; hard links need one volume")
    return inboxes


def run_listener(cfg: ListenerConfig, *, stop: threading.Event, signalled=lambda: False, spawn=None,
                 input_args: list[str] | None = None, tuning: stream.StreamTuning = stream.StreamTuning(),
                 clock=time.monotonic, now_utc=stream._utcnow, rng: random.Random | None = None,
                 status_every_s: float = 10.0, summary_s: float = 300.0,
                 inbox_warn_s: float = 600.0) -> int:
    """Run until stop/signal. 0 after a signal, 1 after a startup error or a dead
    capture thread. Every ffmpeg is gone when this returns."""
    rng = rng or random.Random()
    started = now_utc()
    log.info("listener: %d cameras, consumers %s, audio %s, inbox cap %d s / %d MB",
             len(cfg.cameras), ",".join(cfg.consumers), cfg.audio_dir,
             cfg.inbox_max_age_s, cfg.inbox_max_bytes // (1024 * 1024))
    try:
        inboxes = prepare(cfg)
    except OSError as exc:
        log.error("audio dir %s not usable: %s", cfg.audio_dir, bl.scrub(str(exc)))
        return 1
    dirs = {cam.name: stream.camera_dirname(i, cam.name) for i, cam in enumerate(cfg.cameras)}
    fan = Fanout(inboxes, dirs)
    stats = stream.Stats(clock)
    registry = stream.ProcRegistry()
    inbox_state = {c: InboxState() for c in cfg.consumers}
    sups: dict[str, stream.CameraSupervisor] = {}
    prev_hook = threading.excepthook
    threading.excepthook = stream._scrubbed_excepthook
    rc = 0
    try:
        for i, cam in enumerate(cfg.cameras):
            sup = stream.CameraSupervisor(
                i, cam, seg_dir=cfg.segment_dir, clip_seconds=cfg.clip_seconds, queue=fan, stats=stats,
                registry=registry, stop=stop, signalled=signalled, once=False, spawn=spawn, tuning=tuning,
                clock=clock, now_utc=now_utc, rng=rng, input_args=input_args)
            sups[cam.name] = sup
            stream._track(sup)
            sup.start()

        def refresh(final: bool = False) -> None:
            now = now_utc()
            for c, path in inboxes.items():
                st = inbox_state[c]
                try:
                    by_age, by_size, st.files, st.bytes, st.oldest_utc = prune_inbox(
                        path, max_age_s=cfg.inbox_max_age_s, max_bytes=cfg.inbox_max_bytes, now_utc=now)
                except OSError as exc:
                    log.warning("inbox %s: %s", c, bl.scrub(str(exc)))
                    continue
                st.pruned_age += by_age; st.pruned_size += by_size
                st._unreported += by_age + by_size
                m = clock()
                if st._unreported and (st._last_warn is None or m - st._last_warn >= inbox_warn_s):
                    log.warning("inbox %s: dropped %d unread segments (consumer stopped or behind); "
                                "%d files, %.0f MB waiting", c, st._unreported, st.files, st.bytes / 1e6)
                    st._unreported, st._last_warn = 0, m
            status = build_status(started_utc=started, now_utc=now, cfg=cfg, cameras=fan.snapshot(),
                                  supervisors={} if final else sups, inboxes=inbox_state)
            try:
                write_status(cfg.audio_dir, status)
            except OSError as exc:
                log.warning("could not write status: %s", bl.scrub(str(exc)))

        def summary() -> None:
            cams = fan.snapshot()
            up = sum(1 for s in sups.values() if s.up)
            parts = [f"cameras_up={up}/{len(sups)}"]
            parts += [f"{n}={c.segments}" for n, c in cams.items()]
            parts += [f"inbox_{n}={s.files}" for n, s in inbox_state.items()]
            log.info("listener summary %s", " ".join(parts))

        next_status = clock()
        next_summary = clock() + summary_s
        while True:
            if signalled() or stop.is_set():
                break
            dead = next((s for s in sups.values() if not s.is_alive()), None)
            if dead is not None:
                log.error("capture thread %s died", dead.cam.name)
                rc = 1
                break
            now = clock()
            if now >= next_status:
                refresh()
                next_status = now + status_every_s
            if now >= next_summary:
                summary()
                next_summary = now + summary_s
            time.sleep(tuning.tick_s)
    except Exception:
        log.error("listener crashed:\n%s", bl.scrub(traceback.format_exc()))
        rc = 1
    finally:
        stop.set()
        deadline = clock() + tuning.shutdown_budget_s
        for s in sups.values():
            s.join(timeout=max(0.0, deadline - clock()))
        stream._sweep(registry, clock)
        try:
            refresh(final=True)
        except Exception:                             # noqa: BLE001 -- best effort on the way out
            pass
        if not any(s.is_alive() for s in sups.values()):
            threading.excepthook = prev_hook
        log.info("listener stopped")
    return rc


def main(argv=None, env=os.environ) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--healthcheck"]:
        ok, msg = healthcheck(env)
        print(("healthy: " if ok else "unhealthy: ") + msg)
        return 0 if ok else 1
    if argv == ["--selftest"]:
        return _selftest_main()
    if argv:
        print(__doc__)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    try:
        cfg = load_config(env)
    except bl.ConfigError as exc:
        log.error("config: %s", bl.scrub(str(exc)))
        return 2
    flag = stream._SignalFlag()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stream._make_handler(flag, {}))
    return run_listener(cfg, stop=threading.Event(), signalled=lambda: flag.value)


# ----------------------------------------------------------------- CI self-test
def selftest(deadline_s: float = 20.0) -> list[str]:
    """Real ffmpeg (the 7 s lavfi tone from stream.py) through the listener into
    two inboxes. Returns problems; [] = pass."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="listener-selftest-") as tmp:
        cfg = ListenerConfig(cameras=(bl.Camera("selftest", "rtsp://selftest.invalid/"),),
                             consumers=("a", "b"), audio_dir=Path(tmp), clip_seconds=3)
        stop = threading.Event()
        tuning = stream.StreamTuning(min_segment_s=0.5, stagger_s=0, backoff_base_s=30, backoff_cap_s=30, poll_s=0.1)
        out: list[int] = []
        t = threading.Thread(target=lambda: out.append(run_listener(
            cfg, stop=stop, input_args=list(stream.SELFTEST_INPUT), tuning=tuning, status_every_s=0.2)),
            name="listener-selftest", daemon=True)
        t.start()
        a, b = cfg.inbox("a"), cfg.inbox("b")
        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            if a.is_dir() and len([n for n in os.listdir(a) if parse_inbox_name(n)]) >= 3:
                break
            time.sleep(0.1)
        time.sleep(0.5)                               # one more status write
        stop.set()
        t.join(timeout=10)
        if t.is_alive():
            return ["listener still running 10 s after stop"]
        names_a = sorted(n for n in os.listdir(a) if parse_inbox_name(n))
        names_b = sorted(n for n in os.listdir(b) if parse_inbox_name(n))
        if len(names_a) != 3:
            problems.append(f"inbox a has {len(names_a)} segments, expected 3: {names_a}")
        if names_a != names_b:
            problems.append(f"inboxes differ: {names_a} vs {names_b}")
        for n in names_a:
            st = os.stat(a / n)
            if st.st_nlink != 2:
                problems.append(f"{n} has {st.st_nlink} links, expected 2 (one per inbox)")
            if not n.startswith("00-selftest_r0001_"):
                problems.append(f"unexpected inbox name {n}")
        left = [n for d in (cfg.segment_dir / "queued", cfg.segment_dir / "00-selftest") if d.is_dir()
                for n in os.listdir(d)]
        if left:
            problems.append(f"segments left outside the inboxes: {left}")
        try:
            status = json.loads((Path(tmp) / STATUS_FILE).read_text())
            cam = status["cameras"]["selftest"]
            if cam["segments"] != 3 or not cam["last_segment_utc"]:
                problems.append(f"status camera entry wrong: {cam}")
            if status["inboxes"]["a"]["files"] != 3:
                problems.append(f"status inbox entry wrong: {status['inboxes']['a']}")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"status.json unreadable: {type(exc).__name__}: {exc}")
        if out != [0]:
            problems.append(f"run_listener returned {out}, expected [0]")
    return problems


def _selftest_main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    problems = selftest()
    if problems:
        print("listener selftest FAILED:")
        for p in problems:
            print(f"  {bl.scrub(p)}")
        return 1
    print("listener selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
