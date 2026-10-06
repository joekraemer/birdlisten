"""Stream capture: every camera at once (CAPTURE_MODE=stream, issue #4).

Each camera gets one supervisor thread that keeps one long-lived ffmpeg
running. ffmpeg writes back-to-back CLIP_SECONDS WAV segments with its
segment muxer into SEGMENT_DIR/<NN>-<name>/. A finished segment is validated,
renamed into SEGMENT_DIR/queued/ and put on a bounded drop-oldest queue. One
analysis worker thread owns the BirdNET analyzer, the SQLite write connection
and ntfy.

This module holds the building blocks (tuning constants, the ffmpeg command,
the queue, the process registry, the summary statistics, /proc sums, the
process-wide run guard, signal and excepthook helpers, segment file helpers),
the per-camera supervisor thread, the analysis worker, and run_stream /
main_stream, which run the pipeline until a signal. Everything is stdlib only.
"""

from __future__ import annotations

import collections
import datetime as dt
import logging
import os
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path

import birdlisten as bl

log = logging.getLogger("birdlisten")


# ----------------------------------------------------------------- tuning
@dataclass(frozen=True)
class StreamTuning:
    """Internal timings in seconds. Constants, not env vars; tests shrink them."""
    poll_s: float = 0.5
    tick_s: float = 0.25
    stall_s: float = 30.0
    term_grace_s: float = 2.0
    supervisor_join_s: float = 4.0
    shutdown_budget_s: float = 8.0
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 60.0
    healthy_run_s: float = 120.0
    stagger_s: float = 1.5
    still_failing_log_s: float = 600.0
    min_segment_s: float = 3.0
    fatal_wait_s: float = 60.0
    worker_ready_s: float = 10.0
    once_worker_per_clip_s: float = 120.0


# ----------------------------------------------------------------- names
# A segment ffmpeg writes into a camera directory: r<run>_<UTC start>Z.wav.
SEG_RE = re.compile(r"^r(\d{4,})_(\d{8}T\d{6})Z\.wav$")
# A camera directory: <index:02d>-<sanitised name>.
CAMDIR_RE = re.compile(r"^\d{2,}-[A-Za-z0-9_-]+$")
# A segment handed to the queue: <camera dir>_<segment name>.
QUEUED_RE = re.compile(r"^\d{2,}-[A-Za-z0-9_-]+_r\d{4,}_\d{8}T\d{6}Z\.wav$")
# An ffmpeg exit reason that means the NVR refused the session.
REFUSED_RE = re.compile(r"\b(453|503)\b|Not Enough Bandwidth|Service Unavailable|Connection refused", re.I)


def camera_dirname(index: int, name: str) -> str:
    """Safe, unique path component for a camera: '00-front_door_' for 'front door!'."""
    return f"{index:02d}-" + re.sub(r"[^A-Za-z0-9_-]", "_", name)


def segment_pattern(seg_dir: Path, index: int, name: str, run: int) -> str:
    """ffmpeg -strftime output pattern for one run of one camera."""
    return str(Path(seg_dir) / camera_dirname(index, name) / f"r{run:04d}_%Y%m%dT%H%M%SZ.wav")


# ----------------------------------------------------------------- ffmpeg
def rtsp_input(cam: bl.Camera) -> list[str]:
    return ["-rtsp_transport", "tcp", "-i", cam.rtsp]


def segment_cmd(input_args: list[str], clip_seconds: int, pattern: str) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            *input_args,
            "-vn", "-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le",
            "-f", "segment", "-segment_time", str(clip_seconds), "-segment_format", "wav",
            "-reset_timestamps", "1", "-strftime", "1", pattern]


def spawn_env() -> dict[str, str]:
    """TZ=UTC makes ffmpeg's strftime segment names UTC."""
    return {**os.environ, "TZ": "UTC"}


def popen_spawn(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, errors="replace", env=env)


# ----------------------------------------------------------------- queue
@dataclass(frozen=True)
class Clip:
    camera: str               # camera name; the worker maps it back to the Camera
    path: Path                # in SEGMENT_DIR/queued/
    start_utc: dt.datetime    # aware UTC, from the segment name
    duration_s: float         # from the WAV header
    ready_utc: dt.datetime    # when the supervisor handed it off
    ready_mono: float         # same moment on the monotonic clock


CLOSED = object()


class ClipQueue:
    """Bounded FIFO of Clips. A put into a full queue drops (and deletes) the
    oldest clip; put never blocks."""

    def __init__(self, maxsize: int, stats: Stats, now_mono):
        self.maxsize = maxsize
        self.cond = threading.Condition()
        self._dq: collections.deque[Clip] = collections.deque()
        self._closed = False
        self._stats = stats
        self._now = now_mono

    def put(self, clip: Clip) -> None:
        rejected = dropped = None
        first, total = False, 0
        with self.cond:
            if self._closed:
                rejected = clip
            else:
                if len(self._dq) >= self.maxsize:
                    dropped = self._dq.popleft()
                self._dq.append(clip)
                n = len(self._dq)
                self.cond.notify()
                # Lock order is always queue -> stats; Stats never calls back.
                self._stats.clip_queued(n)
                if dropped is not None:
                    first, total = self._stats.note_drop()
        # File and log I/O only after the lock is released.
        if rejected is not None:
            safe_unlink(rejected.path)
            return
        if dropped is not None:
            safe_unlink(dropped.path)
            log.info("%s", bl._fmt_timing(f"camera={dropped.camera}", [
                ("clip_s", dropped.duration_s, "{:.1f}"),
                ("queue_wait_s", self._now() - dropped.ready_mono, "{:.2f}"),
                ("dropped", "queue_full", "{}"),
                ("rss_mb", bl._max_rss_mb(), "{:.0f}"),
            ]))
            if first:
                log.warning("analysis behind, dropped oldest clip %s %s (%d dropped since start)",
                            dropped.camera, dropped.start_utc.isoformat(timespec="seconds"), total)

    def get(self, timeout: float):
        """Next Clip; CLOSED once closed and empty; None on timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self._dq or self._closed, timeout)
            if self._dq:
                return self._dq.popleft()
            return CLOSED if self._closed else None

    def close(self) -> None:
        with self.cond:
            self._closed = True
            self.cond.notify_all()

    def drain(self) -> int:
        """Remove and delete every queued clip. Returns how many there were."""
        with self.cond:
            clips = list(self._dq)
            self._dq.clear()
        for clip in clips:
            safe_unlink(clip.path)
        return len(clips)

    def __len__(self) -> int:
        with self.cond:
            return len(self._dq)


# ----------------------------------------------------------------- registry
class ProcRegistry:
    def __init__(self): self._lock = threading.Lock(); self._procs = {}; self._closed = False

    def add(self, cam: str, proc) -> bool:        # False after close(): caller kills + waits
        with self._lock:
            if self._closed: return False
            self._procs[cam] = proc; return True

    def remove(self, cam: str, proc) -> None:     # after the supervisor's wait() returned
        with self._lock:
            if self._procs.get(cam) is proc: del self._procs[cam]

    def snapshot(self) -> list[tuple[str, object]]:
        with self._lock: return list(self._procs.items())

    def close(self) -> list[tuple[str, object]]:  # main thread, final sweep
        with self._lock: self._closed = True; return list(self._procs.items())


# ----------------------------------------------------------------- stats
COUNTERS = ("clips", "analyzed", "dropped", "short", "failed", "restarts", "refused")


class Stats:
    """Window counters and running totals for the summary line. Every read or
    write of the state goes through a method that takes the lock."""

    def __init__(self, clock):
        self._lock = threading.Lock(); self._clock = clock
        self._win = dict.fromkeys(COUNTERS, 0); self._tot = dict.fromkeys(COUNTERS, 0)
        self._drop_warned = False; self._queue_max = 0
        self._analyze: list[float] = []; self._notify_s = 0.0
        self._busy_s = 0.0; self._busy_since: float | None = None
        self._win_start = clock()

    def add(self, **counts: int) -> None:
        """stats.add(failed=1); stats.add(restarts=1, refused=0). Unknown key -> KeyError."""
        unknown = set(counts) - set(COUNTERS)
        if unknown:
            raise KeyError(", ".join(sorted(unknown)))
        with self._lock:
            for k, n in counts.items():
                self._win[k] += n; self._tot[k] += n

    def note_drop(self) -> tuple[bool, int]:
        """Count a drop. Returns (first drop in this window?, total drops)."""
        with self._lock:
            self._win["dropped"] += 1; self._tot["dropped"] += 1
            first = not self._drop_warned; self._drop_warned = True
            return first, self._tot["dropped"]

    def clip_queued(self, qlen: int) -> None:
        with self._lock:
            self._win["clips"] += 1; self._tot["clips"] += 1
            self._queue_max = max(self._queue_max, qlen)

    def analyzed(self, analyze_s: float, notify_s: float) -> None:
        with self._lock:
            self._win["analyzed"] += 1; self._tot["analyzed"] += 1
            self._analyze.append(analyze_s); self._notify_s += notify_s

    def busy_begin(self) -> None:
        with self._lock:
            self._busy_since = self._clock()

    def busy_end(self) -> None:
        with self._lock:
            if self._busy_since is not None:
                self._busy_s += self._clock() - self._busy_since
            self._busy_since = None

    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tot)

    def snapshot_and_reset(self, now: float, *, queue_len: int, cameras: int, cameras_up: int,
                           cpu_s: float, rss_mb: float, ffmpeg_cpu_s: float | None = None,
                           ffmpeg_rss_mb: float | None = None) -> dict:
        """The summary fields in line order, then start a new window at `now`.
        analyze_mean_s/analyze_p95_s are absent when nothing was analyzed;
        ffmpeg_* are passed through and may be None (the formatter skips None)."""
        with self._lock:
            if self._busy_since is not None:          # split an in-progress clip at the boundary
                self._busy_s += now - self._busy_since
                self._busy_since = now
            window = now - self._win_start
            duty = min(1.0, max(0.0, self._busy_s / window)) if window > 0 else 0.0
            out: dict = {"window_s": window, "cameras": cameras, "cameras_up": cameras_up}
            for k in COUNTERS:
                out[k] = self._win[k]
            if self._analyze:
                v = sorted(self._analyze)
                out["analyze_mean_s"] = sum(v) / len(v)
                out["analyze_p95_s"] = v[int(.95 * (len(v) - 1))]
            out.update(duty=duty, notify_s=self._notify_s, queue_max=self._queue_max,
                       queue_len=queue_len, cpu_s=cpu_s, ffmpeg_cpu_s=ffmpeg_cpu_s,
                       rss_mb=rss_mb, ffmpeg_rss_mb=ffmpeg_rss_mb)
            self._win = dict.fromkeys(COUNTERS, 0)
            self._analyze = []; self._notify_s = 0.0; self._busy_s = 0.0
            self._drop_warned = False
            self._queue_max = queue_len
            self._win_start = now
            return out


# ----------------------------------------------------------------- resources
def _self_cpu_seconds() -> float:
    """User+sys CPU of this process, all threads. No RUSAGE_CHILDREN: reaping a
    restarted ffmpeg would add its whole lifetime to one clip."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    return s.ru_utime + s.ru_stime


def _proc_sums(snapshot: list[tuple[str, object]], prev_cpu: dict[int, float], *,
               proc_root: Path = Path("/proc"), clk_tck: int | None = None
               ) -> tuple[float | None, float | None, dict[int, float]]:
    """Return (ffmpeg_cpu_s delta, ffmpeg_rss_mb, new prev_cpu). None = nothing readable."""
    cpu = rss = 0.0; read = 0; new_prev: dict[int, float] = {}
    if not proc_root.is_dir(): return None, None, {}
    tck = clk_tck or os.sysconf("SC_CLK_TCK")
    for _cam, proc in snapshot:
        pid = proc.pid
        try:
            stat = (proc_root / str(pid) / "stat").read_text()
            fields = stat.rsplit(")", 1)[1].split()          # comm may contain spaces
            total = (int(fields[11]) + int(fields[12])) / tck  # utime, stime (fields 14, 15)
            status = (proc_root / str(pid) / "status").read_text()
            kb = next(int(l.split()[1]) for l in status.splitlines() if l.startswith("VmRSS:"))
        except (OSError, ValueError, IndexError, StopIteration):
            continue                                          # pid gone or unreadable: skip it
        cpu += total - prev_cpu.get(pid, 0.0); rss += kb / 1024; read += 1
        new_prev[pid] = total
    return (cpu, rss, new_prev) if read else (None, None, new_prev)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ----------------------------------------------------------------- run guard
# One pipeline per process: a second run_stream is refused while any thread of
# an earlier one is alive.
_GUARD = threading.Lock()
_RUN = {"active": False, "threads": []}      # threads started by the current or last run


def _claim_run() -> int | None:
    """Return None if claimed, else the number of earlier threads still alive."""
    with _GUARD:
        alive = [t for t in _RUN["threads"] if t.is_alive()]
        if _RUN["active"] or alive:
            return len(alive) or 1
        _RUN["active"] = True; _RUN["threads"] = []
        return None


def _track(t: threading.Thread) -> None:      # called just before every t.start()
    with _GUARD: _RUN["threads"].append(t)


def _release_run() -> None:                   # outermost finally of run_stream
    with _GUARD: _RUN["active"] = False       # the thread list is kept on purpose


def _guard_threads_alive() -> bool:
    """True while any thread the current or last run started is alive."""
    with _GUARD:
        return any(t.is_alive() for t in _RUN["threads"])


# ----------------------------------------------------------------- crash + signals
def _scrubbed_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:          # same as the default hook
        return
    tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    name = args.thread.name if args.thread is not None else "?"
    log.error("thread %s crashed:\n%s", name, bl.scrub(tb))


class _SignalFlag:
    def __init__(self): self.value = False


def _make_handler(flag, prev):
    def _handler(signum, frame):
        flag.value = True                          # plain attribute store, nothing else
        p = prev.get(signum)
        if callable(p) and p is not signal.default_int_handler:
            p(signum, frame)                       # loop.py's _on_signal: logs, sets loop._stop
    return _handler


# ----------------------------------------------------------------- segment files
def validate_segment(path: Path, min_s: float) -> tuple[str, float, str]:
    """('ok', duration, ''), ('short', duration, '') or ('bad', 0.0, reason)."""
    try:
        size = Path(path).stat().st_size
        with wave.open(str(path), "rb") as w:
            rate, channels, width, nframes = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        return "bad", 0.0, bl.scrub(str(exc)) or type(exc).__name__
    if rate != 48000:
        return "bad", 0.0, f"rate {rate}"
    if channels != 1:
        return "bad", 0.0, f"{channels} channels"
    if width != 2:
        return "bad", 0.0, f"sample width {width}"
    if 44 + nframes * 2 > size:
        return "bad", 0.0, f"truncated, {nframes} frames in {size} bytes"
    duration = nframes / 48000
    if duration < min_s:
        return "short", duration, ""
    return "ok", duration, ""


def parse_start_utc(name: str) -> dt.datetime:
    """Segment start from 'r0001_20261005T231500Z.wav', as an aware UTC datetime."""
    m = SEG_RE.match(name)
    if not m:
        raise ValueError(f"not a segment name: {name!r}")
    return dt.datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def safe_unlink(path) -> bool:
    """Delete a file; a missing file is fine. Returns False if it could not be deleted."""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.warning("could not delete %s: %s", path, bl.scrub(str(exc)))
        return False


def cleanup_stale(seg_dir: Path) -> int:
    """Delete segments a crashed run left behind. Only pattern-matching files in
    camera directories and queued/ are touched. Returns how many were removed."""
    seg_dir = Path(seg_dir)
    targets = []
    for d in sorted(seg_dir.iterdir()):
        if d.is_dir() and not d.is_symlink() and CAMDIR_RE.match(d.name):
            targets += [f for f in d.iterdir() if SEG_RE.match(f.name) and f.is_file()]
    queued = seg_dir / "queued"
    if queued.is_dir() and not queued.is_symlink():
        targets += [f for f in queued.iterdir() if QUEUED_RE.match(f.name) and f.is_file()]
    n = sum(1 for f in targets if safe_unlink(f))
    if n:
        log.info("removed %d stale segments", n)
    return n


def prepare_segment_dir(seg_dir: Path, peak_bytes: int) -> None:
    """Create SEGMENT_DIR and queued/, prove it is writable, and warn if free
    space is below the expected peak. OSError is left to the caller."""
    seg_dir = Path(seg_dir)
    (seg_dir / "queued").mkdir(parents=True, exist_ok=True)
    probe = seg_dir / f".probe-{os.getpid()}"
    probe.write_bytes(b"")
    probe.unlink()
    free = shutil.disk_usage(seg_dir).free
    if free < peak_bytes:
        log.warning("segment dir %s has %.0f MB free, peak use is about %.0f MB",
                    seg_dir, free / 1e6, peak_bytes / 1e6)


# ----------------------------------------------------------------- supervisor
def _read_stderr(proc, lines: collections.deque) -> None:
    """Keep the last scrubbed stderr lines of one ffmpeg; ends at EOF."""
    for line in proc.stderr:
        lines.append(bl.scrub(line.rstrip()))


class CameraSupervisor(threading.Thread):
    """Keeps one ffmpeg segmenter running for one camera and hands each
    finished segment to the queue. A segment is finished once a successor
    exists or ffmpeg has exited. Restarts with jittered exponential backoff."""

    def __init__(self, index: int, cam: bl.Camera, *, seg_dir: Path, clip_seconds: int,
                 queue: ClipQueue, stats: Stats, registry: ProcRegistry,
                 stop: threading.Event, signalled, once: bool, spawn, tuning: StreamTuning,
                 clock, now_utc, rng, input_args: list[str] | None = None):
        super().__init__(name=f"cam-{cam.name}", daemon=True)
        self.index = index
        self.cam = cam
        self.seg_dir = Path(seg_dir)
        self.dirname = camera_dirname(index, cam.name)
        self.cam_dir = self.seg_dir / self.dirname
        self.queued_dir = self.seg_dir / "queued"
        self.clip_seconds = clip_seconds
        self.up = False                               # read by the main thread for cameras_up
        self.fails = 0                                # failures since the last healthy run
        self.backoffs: collections.deque[float] = collections.deque(maxlen=32)  # recent delays
        self._queue = queue
        self._stats = stats
        self._registry = registry
        self._stop_event = stop                       # not _stop: Thread uses that name
        self._signalled = signalled
        self._once = once
        self._spawn_fn = spawn                        # None -> popen_spawn, looked up at call time
        self._tuning = tuning
        self._clock = clock
        self._now_utc = now_utc
        self._rng = rng
        self._input_args = input_args
        self._run_no = 0
        self._proc = None
        self._seen: dict[str, int] = {}
        self._seq = 0
        self._recent: collections.deque[str] = collections.deque(maxlen=8)
        self._streak_key: str | None = None           # reason key of the last ERROR; None = no streak
        self._streak_since: dt.datetime | None = None
        self._last_warn = 0.0

    # ------------------------------------------------------------- thread body
    def run(self) -> None:
        if self._stop_event.wait(self.index * self._tuning.stagger_s):
            return
        while True:
            try:
                delay = self._spawn_cycle()
            except Exception:
                log.error("%s: supervisor crashed: %s", self.cam.name, bl.scrub(traceback.format_exc()))
                self._reap_after_crash()
                if self._stop_event.is_set() or self._signalled() or self._once:
                    return
                self.fails += 1
                self._stats.add(restarts=1)
                delay = self._backoff()
            if delay is None or self._stop_event.wait(delay):
                return

    def _spawn_cycle(self) -> float | None:
        """One ffmpeg run. Returns the backoff before the next one, or None to end."""
        t = self._tuning
        name = self.cam.name
        self._run_no += 1
        run = self._run_no
        self._seen, self._seq = {}, 0
        self._recent = collections.deque(maxlen=8)
        self._proc = None
        self.cam_dir.mkdir(parents=True, exist_ok=True)
        self._clear_dir()
        args = rtsp_input(self.cam) if self._input_args is None else self._input_args
        cmd = segment_cmd(args, self.clip_seconds, segment_pattern(self.seg_dir, self.index, name, run))
        spawn = self._spawn_fn or popen_spawn
        try:
            proc = spawn(cmd, spawn_env())
        except OSError as exc:
            if self._stop_event.is_set() or self._signalled():
                return None
            return self._fail(bl.scrub(str(exc)) or type(exc).__name__, 0.0, 0, stalled=False)
        self._proc = proc
        if not self._registry.add(name, proc):        # shutdown has begun
            proc.kill(); proc.wait()
            self._proc = None
            return None
        lines: collections.deque[str] = collections.deque(maxlen=20)
        reader = threading.Thread(target=_read_stderr, args=(proc, lines),
                                  name=f"cam-{name}-stderr", daemon=True)
        _track(reader)
        reader.start()
        spawn_mono = self._clock()
        last_progress = spawn_mono
        prev = None
        handed = 0
        stall = None
        ended = None                                  # "stop" | "once" | "deadline" | None (exit/stall)
        deadline = spawn_mono + 2 * self.clip_seconds + t.stall_s
        while True:
            rc = proc.poll()
            if rc is not None:
                self._registry.remove(name, proc)     # right after the reaping poll()
                break
            if self._stop_event.is_set():
                rc = self._terminate(proc); ended = "stop"
                break
            n, newest = self._poll_once(run, alive=True)
            handed += n
            size = None
            if newest is not None:
                try:
                    size = (self.cam_dir / newest).stat().st_size
                except OSError:
                    pass
            now = self._clock()
            if (newest, size) != prev:
                prev = (newest, size); last_progress = now
            self.up = bool(size)
            if self.fails and handed and now - spawn_mono >= t.healthy_run_s:
                log.info("%s: recovered after %d attempts", name, self.fails)
                self.fails = 0; self._streak_key = None
            if self._once and handed:
                rc = self._terminate(proc); ended = "once"
                break
            if self._once and now >= deadline:
                rc = self._terminate(proc); ended = "deadline"
                break
            if now - last_progress >= t.stall_s:
                stall = f"stalled: no audio for {t.stall_s:g}s"
                rc = self._terminate(proc)
                break
            self._stop_event.wait(t.poll_s)
        up_s = self._clock() - spawn_mono
        self.up = False
        self._proc = None
        reader.join(timeout=1.0)
        if ended is None and not self._stop_event.is_set():
            handed += self._poll_once(run, alive=False)[0]   # the final segment(s)
        self._clear_dir()
        if self._stop_event.is_set() or self._signalled() or ended in ("stop", "once"):
            return None
        if self._once and handed:
            return None
        if ended == "deadline":
            reason = f"no complete segment within {deadline - spawn_mono:g}s"
        else:
            reason = stall or next((l for l in reversed(lines) if l), None) or f"exit {rc}"
        return self._fail(reason, up_s, handed, stalled=stall is not None)

    # ------------------------------------------------------------- segments
    def _poll_once(self, run: int, alive: bool) -> tuple[int, str | None]:
        """Hand off every complete segment of this run. Returns (handed, newest)."""
        names = []
        for n in os.listdir(self.cam_dir):
            m = SEG_RE.match(n)
            if m and int(m.group(1)) == run:
                names.append(n)
        for n in sorted(names):                      # sorted only so a same-poll tie is stable
            if n not in self._seen:
                self._seq += 1; self._seen[n] = self._seq
        order = sorted(names, key=lambda n: (self._seen[n], n))
        newest = order[-1] if order else None
        complete = order[:-1] if alive else order
        handed = 0
        for n in complete:
            handed += self._hand_off(n)
            self._seen.pop(n, None)                  # gone from the camera dir either way
        return handed, (newest if alive else None)

    def _hand_off(self, name: str) -> int:
        cam = self.cam.name
        src = self.cam_dir / name
        if name in self._recent:                     # strftime same-second name reuse
            safe_unlink(src)
            log.info("%s: duplicate segment name %s, dropped", cam, name)
            self._stats.add(failed=1)
            return 0
        status, duration, why = validate_segment(src, self._tuning.min_segment_s)
        if status == "bad":
            log.warning("%s: bad segment %s (%s), deleted", cam, name, why)
            safe_unlink(src)
            self._stats.add(failed=1)
            return 0
        if status == "short":
            log.info("%s: short segment %s (%.1fs), deleted", cam, name, duration)
            safe_unlink(src)
            self._stats.add(short=1)
            return 0
        start = parse_start_utc(name)
        dst = self.queued_dir / f"{self.dirname}_{name}"
        try:
            os.rename(src, dst)
        except OSError as exc:
            log.warning("%s: could not queue %s: %s", cam, name, bl.scrub(str(exc)))
            safe_unlink(src)
            self._stats.add(failed=1)
            return 0
        self._recent.append(name)
        self._queue.put(Clip(camera=cam, path=dst, start_utc=start, duration_s=duration,
                             ready_utc=self._now_utc(), ready_mono=self._clock()))
        return 1

    def _clear_dir(self) -> None:
        """Delete this camera's segment files (any run); leave anything else."""
        for n in os.listdir(self.cam_dir):
            if SEG_RE.match(n):
                safe_unlink(self.cam_dir / n)

    # ------------------------------------------------------------- process
    def _terminate(self, proc):
        """terminate, wait term_grace_s, then kill + wait. Returns the exit code."""
        proc.terminate()
        try:
            rc = proc.wait(timeout=self._tuning.term_grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        self._registry.remove(self.cam.name, proc)   # right after the reaping wait()
        return rc

    def _reap_after_crash(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill(); proc.wait()
        except Exception:                            # already logged the crash; keep the thread alive
            pass
        self._registry.remove(self.cam.name, proc)

    # ------------------------------------------------------------- failures
    def _backoff(self) -> float:
        t = self._tuning
        d = min(t.backoff_cap_s, t.backoff_base_s * 2 ** (self.fails - 1)) * self._rng.uniform(0.8, 1.2)
        self.backoffs.append(d)
        return d

    def _fail(self, reason: str, up_s: float, segments: int, *, stalled: bool) -> float | None:
        """Count, log and time one failed run. Returns the backoff, or None in once mode."""
        name = self.cam.name
        self.fails += 1
        refused = bool(REFUSED_RE.search(reason))
        self._stats.add(restarts=1, refused=int(refused))
        d = None if self._once else self._backoff()
        log.info("%s", bl._fmt_timing(f"camera={name}", [
            ("up_s", up_s, "{:.1f}"),
            ("segments", segments, "{}"),
            ("backoff_s", d, "{:.1f}"),
            ("refused", 1 if refused else None, "{}"),
            ("error", reason if stalled else f"ffmpeg exited: {reason}", "{}"),
        ]))
        if d is None:
            log.error("%s: ffmpeg exited after %.1fs, %d segments: %s", name, up_s, segments, reason)
            return None
        key = re.sub(r"\d+", "#", reason)
        now = self._clock()
        if self._streak_key is None or key != self._streak_key:
            if self._streak_key is None:
                self._streak_since = self._now_utc()
            self._streak_key = key
            self._last_warn = now
            log.error("%s: ffmpeg exited after %.1fs, %d segments: %s; retry in %.1fs",
                      name, up_s, segments, reason, d)
        elif now - self._last_warn >= self._tuning.still_failing_log_s:
            self._last_warn = now
            log.warning("%s: still failing (%d attempts since %s): %s; retry in %.1fs",
                        name, self.fails, self._streak_since.strftime("%H:%M:%SZ"), reason, d)
        return d


# ----------------------------------------------------------------- worker
class Worker(threading.Thread):
    """The one analysis thread. It makes every BirdNET call, owns the only
    SQLite write connection (opened in this thread) and is the only ntfy sender."""

    def __init__(self, cfg: bl.Config, *, queue: ClipQueue, stats: Stats, stop: threading.Event,
                 cameras: dict[str, bl.Camera], analyze_fn, clock):
        super().__init__(name="birdlisten-worker", daemon=True)
        self.cfg = cfg
        self.ready = threading.Event()                # set once open_db returned or failed
        self.open_error: str | None = None
        self._queue = queue
        self._stats = stats
        self._stop_event = stop                       # not _stop: Thread uses that name
        self._cameras = cameras
        self._analyze = analyze_fn
        self._clock = clock

    def run(self) -> None:
        try:
            conn = bl.open_db(self.cfg.data_dir)      # thread-bound sqlite3 connection
        except Exception as exc:
            self.open_error = bl.scrub(str(exc)) or type(exc).__name__
            self.ready.set()
            return
        self.ready.set()
        try:
            while True:
                if self._stop_event.is_set():         # shutdown: take no other clip
                    break
                clip = self._queue.get(timeout=0.5)
                if clip is CLOSED:                    # once mode: closed and drained
                    break
                if clip is None:
                    continue
                if self._stop_event.is_set():         # woken by close() after stop
                    safe_unlink(clip.path)            # shutdown deletion, not counted
                    break
                self._process(conn, clip)
        finally:
            conn.close()

    def _process(self, conn, clip: Clip) -> None:
        self._stats.busy_begin()
        queue_wait = self._clock() - clip.ready_mono
        lag = (clip.ready_utc - (clip.start_utc + dt.timedelta(seconds=clip.duration_s))).total_seconds()
        analyze_s = cpu_s = None
        try:
            c0, t0 = _self_cpu_seconds(), self._clock()
            try:
                dets = self._analyze(clip.path, self.cfg, clip.start_utc)
            finally:
                analyze_s, cpu_s = self._clock() - t0, _self_cpu_seconds() - c0
            best = bl.best_per_species(dets)
            log.info("%s", self._timing(clip, lag, queue_wait, analyze_s, len(best), cpu_s))
            notify_s = bl.store_clip(self.cfg, conn, self._cameras[clip.camera], clip.start_utc,
                                     clip.path, best, clip_s=clip.duration_s, dry_run=False)
            self._stats.analyzed(analyze_s, notify_s)
        except Exception as exc:                      # one bad clip never stops the worker
            self._stats.add(failed=1)
            reason = bl.scrub(str(exc)) or type(exc).__name__
            log.error("%s: %s", clip.camera, reason)
            log.info("%s", self._timing(clip, lag, queue_wait, analyze_s, None, cpu_s, error=reason))
        finally:
            safe_unlink(clip.path)
            self._stats.busy_end()

    @staticmethod
    def _timing(clip: Clip, lag, queue_wait, analyze_s, detections, cpu_s, error=None) -> str:
        return bl._fmt_timing(f"camera={clip.camera}", [
            ("clip_s", clip.duration_s, "{:.1f}"),
            ("segment_lag_s", lag, "{:.2f}"),
            ("queue_wait_s", queue_wait, "{:.2f}"),
            ("analyze_s", analyze_s, "{:.2f}"),
            ("detections", detections, "{}"),
            ("cpu_s", cpu_s, "{:.2f}"),
            ("rss_mb", bl._max_rss_mb(), "{:.0f}"),
            ("error", error, "{}"),
        ])


# ----------------------------------------------------------------- pipeline
@dataclass
class _Pipeline:
    """What run_stream has built so far. Fields stay None/empty until created,
    so the crash path can clean up after a failure at any point."""
    cameras: int = 0
    queue: ClipQueue | None = None
    registry: ProcRegistry | None = None
    stats: Stats | None = None
    worker: Worker | None = None
    supervisors: list = field(default_factory=list)
    started: list = field(default_factory=list)   # every thread run_stream started itself
    prev_cpu: dict = field(default_factory=dict)  # /proc CPU per ffmpeg pid at the last summary
    cpu_mark: float = 0.0                         # _self_cpu_seconds() at the last summary
    final_logged: bool = False


def _start(p: _Pipeline, t: threading.Thread) -> None:
    _track(t)
    p.started.append(t)
    t.start()


def _wait(seconds: float, signalled, tick_s: float, clock) -> None:
    """Sleep in ticks for `seconds`, returning early on a signal."""
    end = clock() + seconds
    while clock() < end and not signalled():
        time.sleep(tick_s)


def _join_until(threads, deadline: float, tick_s: float, clock) -> None:
    for t in threads:
        while t.is_alive() and clock() < deadline:
            time.sleep(tick_s)


_SUMMARY_FMT = {"window_s": "{:.0f}", "analyze_mean_s": "{:.2f}", "analyze_p95_s": "{:.2f}",
                "duty": "{:.2f}", "notify_s": "{:.1f}", "cpu_s": "{:.1f}", "ffmpeg_cpu_s": "{:.1f}",
                "rss_mb": "{:.0f}", "ffmpeg_rss_mb": "{:.0f}"}


def _format_summary(s: dict, final: bool) -> str:
    """'timing summary window_s=... [final=1] cameras=...' in snapshot order; None skipped."""
    pairs = []
    for k, v in s.items():
        pairs.append((k, v, _SUMMARY_FMT.get(k, "{}")))
        if k == "window_s" and final:
            pairs.append(("final", 1, "{}"))
    return bl._fmt_timing("summary", pairs)


def _summary(p: _Pipeline, clock, *, final: bool) -> None:
    now = clock()
    cpu = _self_cpu_seconds()
    ffmpeg_cpu, ffmpeg_rss, p.prev_cpu = _proc_sums(p.registry.snapshot(), p.prev_cpu)
    s = p.stats.snapshot_and_reset(now, queue_len=len(p.queue), cameras=p.cameras,
                                   cameras_up=sum(1 for sup in p.supervisors if sup.up),
                                   cpu_s=cpu - p.cpu_mark, rss_mb=bl._max_rss_mb(),
                                   ffmpeg_cpu_s=ffmpeg_cpu, ffmpeg_rss_mb=ffmpeg_rss)
    p.cpu_mark = cpu
    log.info("%s", _format_summary(s, final))


def _final_summary(p: _Pipeline, clock) -> None:
    if p.stats is None or p.queue is None or p.registry is None or p.final_logged:
        return
    _summary(p, clock, final=True)
    p.final_logged = True


def _sweep(registry: ProcRegistry | None, clock) -> None:
    """Close the registry, kill every ffmpeg still listed, then reap them all
    against one shared 0.5 s deadline."""
    if registry is None:
        return
    procs = registry.close()
    for _cam, proc in procs:
        try:
            proc.kill()
        except OSError:
            pass
    deadline = clock() + 0.5
    for _cam, proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - clock()))
        except subprocess.TimeoutExpired:
            pass


def _refuse(n: int, signalled, once: bool, tuning: StreamTuning, clock) -> int:
    log.error("stream pipeline already running (%d threads alive), not starting another", n)
    if not once:
        _wait(tuning.fatal_wait_s, signalled, tuning.tick_s, clock)
    return 1


def _shutdown(p: _Pipeline, stop: threading.Event, tuning: StreamTuning, clock) -> int:
    """Normal (signal) shutdown inside the shutdown budget. Returns 0."""
    stop.set()
    p.queue.close()
    log.info("stopping capture (signal)")
    start = clock()
    _join_until(p.supervisors, start + tuning.supervisor_join_s, tuning.tick_s, clock)
    if p.worker is not None:
        _join_until([p.worker], start + tuning.shutdown_budget_s, tuning.tick_s, clock)
    p.queue.drain()
    _final_summary(p, clock)
    _sweep(p.registry, clock)
    log.info("stream pipeline stopped")
    return 0


def _crash_shutdown(p: _Pipeline, stop: threading.Event, signalled, once: bool, tuning: StreamTuning,
                    clock, is_base_exc: bool) -> int:
    """The rc 1 path: stop everything run_stream started and wait for it. It
    never raises, every step is idempotent, and it can follow a normal shutdown
    that failed half way. Returns 1."""
    t = tuning

    def step(name, fn):
        try:
            fn()
        except Exception:
            log.error("stream shutdown step %s failed:\n%s", name, bl.scrub(traceback.format_exc()))

    def stop_all():
        stop.set()
        if p.queue is not None:
            p.queue.close()

    # A BaseException means the process is about to exit: signal-path bounds from now.
    sig_start = [clock() if is_base_exc else None]

    def join_all():
        for th in list(p.started):
            bound = t.shutdown_budget_s if th is p.worker else t.supervisor_join_s
            # A worker still inside open_db is joined for at most worker_ready_s;
            # the run guard keeps tracking it.
            ready_limit = clock() + t.worker_ready_s if th is p.worker and not th.ready.is_set() else None
            while th.is_alive():
                now = clock()
                if sig_start[0] is None and signalled():
                    sig_start[0] = now
                if sig_start[0] is not None and now >= sig_start[0] + bound:
                    break
                if ready_limit is not None and now >= ready_limit:
                    break
                time.sleep(t.tick_s)

    def drain_and_summary():
        if p.queue is not None:
            p.queue.drain()
        _final_summary(p, clock)

    step("stop", stop_all)
    step("join", join_all)
    step("summary", drain_and_summary)
    step("sweep", lambda: _sweep(p.registry, clock))
    if sig_start[0] is None and not once:
        step("wait", lambda: _wait(t.fatal_wait_s, signalled, t.tick_s, clock))
    return 1


def run_stream(cfg: bl.Config, *, stop: threading.Event, signalled=lambda: False, once: bool = False,
               spawn=None, analyze_fn=None, load_model=None, tuning: StreamTuning = StreamTuning(),
               clock=time.monotonic, now_utc=_utcnow, rng: random.Random | None = None,
               summary_s: float | None = None, input_args: list[str] | None = None) -> int:
    """Run the stream pipeline until stop/signal (or one window in once mode).
    Returns 0 or 1. Never raises an Exception while a thread it started is
    alive, and refuses to start while an earlier pipeline's threads live."""
    n = _claim_run()
    if n is not None:
        return _refuse(n, signalled, once, tuning, clock)
    p = _Pipeline()
    prev_hook = threading.excepthook
    try:
        threading.excepthook = _scrubbed_excepthook
        try:
            return _run(cfg, p, stop=stop, signalled=signalled, once=once, spawn=spawn,
                        analyze_fn=analyze_fn, load_model=load_model, tuning=tuning, clock=clock,
                        now_utc=now_utc, rng=rng, summary_s=summary_s, input_args=input_args)
        except BaseException as exc:
            log.error("stream main loop crashed:\n%s", bl.scrub(traceback.format_exc()))
            base = not isinstance(exc, Exception)
            rc = _crash_shutdown(p, stop, signalled, once, tuning, clock, base)
            if base:
                raise                                 # KeyboardInterrupt/SystemExit, after cleanup
            return rc
    finally:
        if not _guard_threads_alive():
            threading.excepthook = prev_hook
        _release_run()


def _run(cfg: bl.Config, p: _Pipeline, *, stop, signalled, once, spawn, analyze_fn, load_model,
         tuning: StreamTuning, clock, now_utc, rng, summary_s, input_args) -> int:
    """Startup steps 1-6, the tick loop and the shutdown. Runs inside run_stream's handler."""
    t = tuning
    if analyze_fn is None:
        analyze_fn = lambda w, c, when: bl.analyze(Path(w), c, when)   # noqa: E731
    if load_model is None:
        load_model = lambda: bl.analyzer()                             # noqa: E731
    if rng is None:
        rng = random.Random()
    seg_dir = Path(cfg.segment_dir)
    p.cameras = len(cfg.cameras)
    qsize = cfg.queue_size or 2 * p.cameras
    interval = summary_s if summary_s is not None else cfg.summary_minutes * 60

    def fatal(msg: str) -> int:
        log.error("%s", msg)
        if not once:
            _wait(t.fatal_wait_s, signalled, t.tick_s, clock)
        return 1

    log.info("stream mode: %d cameras, queue %d, summary every %d min, segments %s",
             p.cameras, qsize, cfg.summary_minutes, seg_dir)
    for name in bl.main_stream_urls(cfg):
        log.warning("%s: main-stream URL; use the sub-stream (same audio, less NVR bandwidth)", name)

    # 1. segment dir, writability probe, startup cleanup
    try:
        prepare_segment_dir(seg_dir, peak_bytes=(p.cameras + qsize + 1) * (cfg.clip_seconds * 96000 + 44))
        cleanup_stale(seg_dir)
    except OSError as exc:
        return fatal(f"segment dir {seg_dir} not writable: {bl.scrub(str(exc))}")
    # 2-4. the model, once, in this thread, between two signal checks
    if signalled():
        return 0
    try:
        load_model()
    except Exception as exc:
        return fatal(f"birdnet model failed: {bl.scrub(str(exc)) or type(exc).__name__}")
    if signalled():
        return 0

    # 5. the worker, which must open the database before any ffmpeg starts
    p.stats = Stats(clock)
    p.queue = ClipQueue(qsize, p.stats, clock)
    p.registry = ProcRegistry()
    p.cpu_mark = _self_cpu_seconds()
    p.worker = Worker(cfg, queue=p.queue, stats=p.stats, stop=stop,
                      cameras={c.name: c for c in cfg.cameras}, analyze_fn=analyze_fn, clock=clock)
    _start(p, p.worker)
    end = clock() + t.worker_ready_s
    while not p.worker.ready.is_set() and clock() < end:
        if signalled() or stop.is_set():
            return _shutdown(p, stop, t, clock)
        time.sleep(t.tick_s)
    if p.worker.open_error is not None or not p.worker.ready.is_set():
        timed_out = not p.worker.ready.is_set()
        stop.set()                                    # the worker checks it before its first get()
        _join_until([p.worker], clock() + t.worker_ready_s, t.tick_s, clock)
        return fatal("database open failed: " + ("timed out" if timed_out else p.worker.open_error))

    # 6. one supervisor per camera; each staggers its own first spawn
    for i, cam in enumerate(cfg.cameras):
        sup = CameraSupervisor(i, cam, seg_dir=seg_dir, clip_seconds=cfg.clip_seconds, queue=p.queue,
                               stats=p.stats, registry=p.registry, stop=stop, signalled=signalled,
                               once=once, spawn=spawn, tuning=t, clock=clock, now_utc=now_utc,
                               rng=rng, input_args=input_args)
        p.supervisors.append(sup)
        _start(p, sup)

    next_summary = [clock() + interval]

    def maybe_summary():
        if clock() >= next_summary[0]:
            _summary(p, clock, final=False)
            next_summary[0] = clock() + interval

    def worker_died() -> int:
        log.error("analysis worker died")
        return _crash_shutdown(p, stop, signalled, once, t, clock, False)

    if not once:
        while True:
            if signalled() or stop.is_set():
                return _shutdown(p, stop, t, clock)
            if not p.worker.is_alive():
                return worker_died()
            dead = next((s for s in p.supervisors if not s.is_alive()), None)
            if dead is not None:
                log.error("capture thread %s died", dead.cam.name)
                return _crash_shutdown(p, stop, signalled, once, t, clock, False)
            maybe_summary()
            time.sleep(t.tick_s)

    # once mode: every camera hands off one segment (or fails once), then the
    # worker analyzes everything queued and exits on CLOSED.
    while any(s.is_alive() for s in p.supervisors):
        if signalled() or stop.is_set():
            return _shutdown(p, stop, t, clock)
        if not p.worker.is_alive():
            return worker_died()
        maybe_summary()
        time.sleep(t.tick_s)
    p.queue.close()
    end = clock() + p.cameras * t.once_worker_per_clip_s
    while p.worker.is_alive() and clock() < end:
        if signalled() or stop.is_set():
            return _shutdown(p, stop, t, clock)
        maybe_summary()
        time.sleep(t.tick_s)
    rc = 0 if p.stats.totals()["analyzed"] else 1
    if p.worker.is_alive():
        log.error("analysis worker did not finish")
        stop.set()
        rc = 1
    p.queue.drain()
    _final_summary(p, clock)
    _sweep(p.registry, clock)
    log.info("stream pipeline stopped")
    return rc


def main_stream(cfg: bl.Config, once: bool = False, **seams) -> int:
    """Install the signal handlers, then run the pipeline until a signal (or
    one window with once=True). Called from birdlisten.main() in the main thread."""
    flag = _SignalFlag()
    saved: dict = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            saved[sig] = signal.getsignal(sig)
        handler = _make_handler(flag, dict(saved))
        for sig in saved:
            signal.signal(sig, handler)
    try:
        # A signal that landed while main() parsed its config only reached
        # loop.py's handler, which set loop._stop.
        if any(getattr(sys.modules.get(m), "_stop", False) is True for m in ("__main__", "loop")):
            flag.value = True
        stop = threading.Event()
        return run_stream(cfg, stop=stop, signalled=lambda: flag.value, once=once, **seams)
    finally:
        for sig, prev in saved.items():
            if prev is not None:
                signal.signal(sig, prev)


# ----------------------------------------------------------------- CI self-test
class SelftestError(RuntimeError):
    """One selftest() check failed; the message says which."""


class _ListHandler(logging.Handler):
    def __init__(self, records: list):
        super().__init__(logging.DEBUG)
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


SELFTEST_INPUT = ["-re", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=16000:duration=7"]


def selftest(spawn=None, deadline_s: float = 20.0, records: list | None = None) -> None:
    """Run one real CameraSupervisor against a 7 s lavfi sine, through the exact
    segment_cmd flags, and check the three clips it hands off (3 s, 3 s, 1 s
    tail). Raises SelftestError on any mismatch. spawn defaults to the real
    Popen; tests pass a fake. Log records are collected into `records` and kept
    off stderr."""
    real_spawn = spawn or popen_spawn
    spawns = [0]

    def counting_spawn(cmd, env):
        spawns[0] += 1
        return real_spawn(cmd, env)

    records = [] if records is None else records
    handler = _ListHandler(records)
    logger = logging.getLogger("birdlisten")
    prev_propagate, prev_level = logger.propagate, logger.level
    logger.addHandler(handler)
    logger.propagate = False
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        with tempfile.TemporaryDirectory(prefix="birdlisten-selftest-") as tmp:
            _selftest_run(Path(tmp), counting_spawn, spawns, deadline_s, records)
    finally:
        logger.removeHandler(handler)
        logger.propagate = prev_propagate
        logger.setLevel(prev_level)


def _selftest_run(seg_dir: Path, spawn, spawns: list, deadline_s: float, records: list) -> None:
    queued = seg_dir / "queued"
    queued.mkdir()
    stats = Stats(time.monotonic)
    queue = ClipQueue(8, stats, time.monotonic)
    registry = ProcRegistry()
    stop = threading.Event()
    tuning = StreamTuning(min_segment_s=0.5, stagger_s=0, backoff_base_s=30, backoff_cap_s=30, poll_s=0.1)
    sup = CameraSupervisor(0, bl.Camera("selftest", "rtsp://selftest.invalid/"), seg_dir=seg_dir,
                           clip_seconds=3, queue=queue, stats=stats, registry=registry, stop=stop,
                           signalled=lambda: False, once=False, spawn=spawn, tuning=tuning,
                           clock=time.monotonic, now_utc=_utcnow, rng=random.Random(0),
                           input_args=list(SELFTEST_INPUT))
    clips: list[Clip] = []
    try:
        sup.start()
        start = time.monotonic()
        while len(clips) < 3 and time.monotonic() < start + deadline_s:
            c = queue.get(timeout=0.5)
            if isinstance(c, Clip):
                clips.append(c)
        # The tail is handed off just before the supervisor logs the exit and
        # enters its backoff; stopping earlier would skip that log line.
        while (len(clips) == 3 and time.monotonic() < start + deadline_s
               and not any(r.levelno == logging.ERROR for r in list(records))):
            time.sleep(0.05)
    finally:
        stop.set()
        sup.join(timeout=5)
        if sup.is_alive():                            # never leave an ffmpeg behind
            for _cam, proc in registry.snapshot():
                try:
                    proc.kill()
                except OSError:
                    pass
            sup.join(timeout=2)

    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise SelftestError(msg)

    check(not sup.is_alive(), "supervisor still running 5 s after stop")
    check(spawns[0] == 1, f"expected 1 ffmpeg spawn, got {spawns[0]}")
    check(len(clips) == 3, f"expected 3 clips, got {len(clips)}")
    for i, c in enumerate(clips[:2]):
        check(abs(c.duration_s - 3.0) <= 0.1, f"clip {i + 1} is {c.duration_s:.2f}s, expected 3.0s")
    check(0.5 <= clips[2].duration_s <= 1.5, f"tail clip is {clips[2].duration_s:.2f}s, expected 0.5-1.5s")
    for i, c in enumerate(clips):
        check(c.path.parent == queued, f"clip {i + 1} is in {c.path.parent}, not queued/")
        check(c.path.name.startswith("00-selftest_r0001_"), f"clip {i + 1} is named {c.path.name}")
        try:
            with wave.open(str(c.path), "rb") as w:
                fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        except (wave.Error, EOFError, OSError) as exc:
            raise SelftestError(f"clip {i + 1} does not open as WAV: {exc}") from exc
        check(fmt == (48000, 1, 2), f"clip {i + 1} is {fmt}, expected (48000, 1, 2)")
    for a, b in zip(clips, clips[1:]):
        gap = (b.start_utc - a.start_utc).total_seconds()
        check(2 <= gap <= 4, f"clip starts {a.start_utc} and {b.start_utc} are {gap:g}s apart, expected 2-4s")
    errors = [r.getMessage() for r in records if r.levelno == logging.ERROR]
    check(len(errors) == 1 and "ffmpeg exited" in errors[0] and "exit 0" in errors[0],
          f"expected one 'ffmpeg exited ... exit 0' ERROR, got {errors}")
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    check(not warnings, f"unexpected WARNING records: {warnings}")
    for c in clips:
        c.path.unlink()
    check(registry.snapshot() == [], f"process registry not empty: {registry.snapshot()}")
    for d in (seg_dir / sup.dirname, queued):
        left = sorted(os.listdir(d)) if d.is_dir() else []
        check(not left, f"{d.name}/ not empty: {left}")


def _selftest_main() -> int:
    records: list = []
    try:
        selftest(records=records)
    except Exception as exc:  # noqa: BLE001 -- any failure fails the CI step
        print(f"selftest FAILED: {bl.scrub(str(exc)) or type(exc).__name__}")
        for r in records:
            print(f"  {r.levelname} {r.threadName}: {bl.scrub(r.getMessage())}")
        return 1
    print("selftest ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(_selftest_main())
    print("usage: python stream.py --selftest", file=sys.stderr)
    sys.exit(2)
