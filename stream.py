"""Stream capture: every camera at once (CAPTURE_MODE=stream, issue #4).

Each camera gets one supervisor thread that keeps one long-lived ffmpeg
running. ffmpeg writes back-to-back CLIP_SECONDS WAV segments with its
segment muxer into SEGMENT_DIR/<NN>-<name>/. A finished segment is validated,
renamed into SEGMENT_DIR/queued/ and put on a bounded drop-oldest queue. One
analysis worker thread owns the BirdNET analyzer, the SQLite write connection
and ntfy.

This module holds the building blocks: tuning constants, the ffmpeg command,
the queue, the process registry, the summary statistics, /proc sums, the
process-wide run guard, signal and excepthook helpers, segment file helpers,
and the per-camera supervisor thread. Everything is stdlib only.
"""

from __future__ import annotations

import collections
import datetime as dt
import logging
import os
import re
import resource
import shutil
import signal
import subprocess
import threading
import traceback
import wave
from dataclasses import dataclass
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
