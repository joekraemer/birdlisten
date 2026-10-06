"""Tests for stream.py (CAPTURE_MODE=stream). Network-free, no real ffmpeg.
Run: uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q test_stream.py"""

from __future__ import annotations

import collections
import datetime as dt
import itertools
import logging
import os
import queue as queue_mod
import random
import re
import signal
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import birdlisten as bl
import stream

UTC = dt.timezone.utc


# ----------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _run_guard():
    """A thread a test leaks fails that test, not a later one."""
    yield
    deadline = time.monotonic() + 2.0
    for t in list(stream._RUN["threads"]):
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in stream._RUN["threads"] if t.is_alive()]
    active = stream._RUN["active"]
    stream._RUN.update(active=False, threads=[])
    if alive or active:
        pytest.fail(f"stream run guard not clean: alive={alive} active={active}")


def make_wav(path: Path, seconds: float, rate: int = 48000, channels: int = 1, width: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\0" * (int(seconds * rate) * channels * width))
    return path


def make_truncated_wav(path: Path, seconds: float, keep_data_bytes: int = 100) -> Path:
    """A WAV whose header promises more frames than the file holds."""
    make_wav(path, seconds)
    with open(path, "r+b") as f:
        f.truncate(44 + keep_data_bytes)
    return path


def make_clip(path: Path, camera="front", duration_s=30.0, ready_mono=0.0,
              start=dt.datetime(2026, 10, 5, 23, 15, tzinfo=UTC)) -> stream.Clip:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return stream.Clip(camera=camera, path=path, start_utc=start, duration_s=duration_s,
                       ready_utc=start, ready_mono=ready_mono)


class FakeClock:
    def __init__(self, t=0.0): self.t = t
    def __call__(self): return self.t


def snap(stats: stream.Stats, now: float, queue_len: int = 0) -> dict:
    return stats.snapshot_and_reset(now, queue_len=queue_len, cameras=4, cameras_up=4, cpu_s=0.0, rss_mb=1.0)


# ----------------------------------------------------------------- 1: ffmpeg command
def test_segment_cmd_exact_argv():
    cam = bl.Camera("front door!", "rtsp://10.0.0.2:554/h264Preview_01_sub")
    p = "/seg/00-front_door_/r0001_%Y%m%dT%H%M%SZ.wav"
    assert stream.segment_cmd(stream.rtsp_input(cam), 30, p) == [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-rtsp_transport", "tcp", "-i", "rtsp://10.0.0.2:554/h264Preview_01_sub",
        "-vn", "-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le",
        "-f", "segment", "-segment_time", "30", "-segment_format", "wav",
        "-reset_timestamps", "1", "-strftime", "1", p,
    ]


def test_spawn_env_is_utc(monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")
    env = stream.spawn_env()
    assert env["TZ"] == "UTC"
    assert env["PATH"]


def test_segment_pattern_sanitises_name(tmp_path):
    assert stream.camera_dirname(0, "front door!") == "00-front_door_"
    assert stream.camera_dirname(12, "back") == "12-back"
    assert stream.segment_pattern(tmp_path, 0, "front door!", 1) == \
        f"{tmp_path}/00-front_door_/r0001_%Y%m%dT%H%M%SZ.wav"


def test_name_regexes():
    assert stream.SEG_RE.match("r0001_20261005T231500Z.wav").groups() == ("0001", "20261005T231500")
    assert not stream.SEG_RE.match("r001_20261005T231500Z.wav")
    assert stream.CAMDIR_RE.match("00-front_door_") and not stream.CAMDIR_RE.match("queued")
    assert stream.QUEUED_RE.match("00-front_r0001_20261005T231500Z.wav")
    assert not stream.QUEUED_RE.match("notes.txt")
    for reason in ("method DESCRIBE failed: 453 Not Enough Bandwidth", "503 Service Unavailable",
                   "Connection refused"):
        assert stream.REFUSED_RE.search(reason)
    assert not stream.REFUSED_RE.search("exit 1")


def test_parse_start_utc_is_aware_utc():
    t = stream.parse_start_utc("r0001_20261005T231500Z.wav")
    assert t.isoformat(timespec="seconds") == "2026-10-05T23:15:00+00:00"
    with pytest.raises(ValueError):
        stream.parse_start_utc("notes.txt")


def test_stream_tuning_defaults():
    t = stream.StreamTuning()
    assert (t.poll_s, t.tick_s, t.stall_s, t.term_grace_s, t.supervisor_join_s, t.shutdown_budget_s,
            t.backoff_base_s, t.backoff_cap_s, t.healthy_run_s, t.stagger_s, t.still_failing_log_s,
            t.min_segment_s, t.fatal_wait_s, t.worker_ready_s, t.once_worker_per_clip_s) == (
        0.5, 0.25, 30, 2, 4, 8, 1, 60, 120, 1.5, 600, 3, 60, 10, 120)
    with pytest.raises(Exception):
        t.poll_s = 1  # frozen


# ----------------------------------------------------------------- 3: validation (pure part)
def test_validate_segment(tmp_path):
    assert stream.validate_segment(make_wav(tmp_path / "ok.wav", 3), 3) == ("ok", 3.0, "")
    status, dur, _ = stream.validate_segment(make_wav(tmp_path / "short.wav", 2), 3)
    assert (status, dur) == ("short", 2.0)
    for path in (make_wav(tmp_path / "rate.wav", 3, rate=16000),
                 make_wav(tmp_path / "stereo.wav", 3, channels=2),
                 make_wav(tmp_path / "width.wav", 3, width=1),
                 make_truncated_wav(tmp_path / "trunc.wav", 3)):
        status, dur, reason = stream.validate_segment(path, 3)
        assert (status, dur) == ("bad", 0.0) and reason, path.name
    (tmp_path / "head.wav").write_bytes(b"RIFF\x00\x00")
    (tmp_path / "junk.wav").write_bytes(b"not a wav file at all, no RIFF header here....")
    for name in ("head.wav", "junk.wav", "missing.wav"):
        status, _, reason = stream.validate_segment(tmp_path / name, 3)
        assert status == "bad" and reason, name


def test_safe_unlink(tmp_path, caplog):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert stream.safe_unlink(f) and not f.exists()
    assert stream.safe_unlink(f)            # already gone: fine, no warning
    d = tmp_path / "dir"
    d.mkdir()
    with caplog.at_level(logging.WARNING, logger="birdlisten"):
        assert stream.safe_unlink(d) is False
    assert "could not delete" in caplog.text


# ----------------------------------------------------------------- 4: queue
def test_queue_drops_oldest_and_warns_once_per_window(tmp_path, caplog):
    stats = stream.Stats(FakeClock())
    q = stream.ClipQueue(2, stats, now_mono=lambda: 100.0)
    clips = [make_clip(tmp_path / f"q{i}.wav", duration_s=12.4 + i, ready_mono=90.0 + i) for i in range(5)]
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        q.put(clips[0]); q.put(clips[1])
        assert not caplog.records
        q.put(clips[2])                       # drops clips[0]
    assert not clips[0].path.exists() and clips[1].path.exists()
    assert stats.totals()["dropped"] == 1 and stats.totals()["clips"] == 3
    timing = [r.getMessage() for r in caplog.records if r.getMessage().startswith("timing ")]
    assert len(timing) == 1
    assert re.fullmatch(r"timing camera=front clip_s=12\.4 queue_wait_s=10\.00 dropped=queue_full rss_mb=\d+",
                        timing[0])
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["analysis behind, dropped oldest clip front 2026-10-05T23:15:00+00:00 (1 dropped since start)"]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        q.put(clips[3])                       # second drop, same window: no WARNING
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "clip_s=13.4" in caplog.records[0].getMessage()

    snap(stats, 10.0, queue_len=len(q))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        q.put(clips[4])                       # new window: WARNING again
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["analysis behind, dropped oldest clip front 2026-10-05T23:15:00+00:00 (3 dropped since start)"]
    assert [c.path.exists() for c in clips] == [False, False, False, True, True]


def test_queue_close_get_and_drain(tmp_path):
    stats = stream.Stats(FakeClock())
    q = stream.ClipQueue(4, stats, now_mono=lambda: 0.0)
    a, b = make_clip(tmp_path / "a.wav"), make_clip(tmp_path / "b.wav")
    assert q.get(timeout=0.01) is None
    q.put(a); q.put(b)
    assert len(q) == 2
    q.close()
    late = make_clip(tmp_path / "late.wav")
    q.put(late)                               # closed: deleted, not queued
    assert not late.path.exists() and len(q) == 2
    assert q.get(timeout=0.01) is a
    assert q.get(timeout=0.01) is b
    assert q.get(timeout=0.01) is stream.CLOSED
    assert stats.totals()["clips"] == 2

    q2 = stream.ClipQueue(4, stats, now_mono=lambda: 0.0)
    c, d = make_clip(tmp_path / "c.wav"), make_clip(tmp_path / "d.wav")
    q2.put(c); q2.put(d)
    assert q2.drain() == 2 and len(q2) == 0
    assert not c.path.exists() and not d.path.exists()
    assert q2.drain() == 0


def test_queue_close_wakes_waiting_get(tmp_path):
    q = stream.ClipQueue(2, stream.Stats(FakeClock()), now_mono=lambda: 0.0)
    got = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=5)))
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(timeout=2)
    assert got == [stream.CLOSED]


# ----------------------------------------------------------------- registry
def test_proc_registry():
    reg = stream.ProcRegistry()
    p1, p2 = object(), object()
    assert reg.add("a", p1) and reg.add("b", p2)
    reg.remove("a", object())                 # not the listed proc: kept
    assert reg.snapshot() == [("a", p1), ("b", p2)]
    reg.remove("a", p1)
    assert reg.close() == [("b", p2)]
    assert reg.add("c", object()) is False


# ----------------------------------------------------------------- 12: Stats
def test_stats_duty_and_window():
    clock = FakeClock(0.0)
    stats = stream.Stats(clock)
    clock.t = 3.0; stats.busy_begin()
    clock.t = 5.5; stats.busy_end()
    s = snap(stats, 10.0)
    assert s["window_s"] == 10.0 and s["duty"] == pytest.approx(0.25)


def test_stats_busy_straddles_window():
    clock = FakeClock(0.0)
    stats = stream.Stats(clock)
    clock.t = 8.0; stats.busy_begin()
    assert snap(stats, 10.0)["duty"] == pytest.approx(0.2)
    clock.t = 13.0; stats.busy_end()
    assert snap(stats, 20.0)["duty"] == pytest.approx(0.3)
    assert snap(stats, 30.0)["duty"] == 0.0


def test_stats_duty_clamped():
    clock = FakeClock(0.0)
    stats = stream.Stats(clock)
    stats.busy_begin()
    clock.t = 20.0; stats.busy_end()
    assert snap(stats, 10.0)["duty"] == 1.0


def test_stats_mean_p95_and_omission():
    stats = stream.Stats(FakeClock(0.0))
    s = snap(stats, 5.0)
    assert "analyze_mean_s" not in s and "analyze_p95_s" not in s
    for v in range(20, 0, -1):                # 20 values, unsorted
        stats.analyzed(float(v), 0.1)
    s = snap(stats, 10.0)
    assert s["analyzed"] == 20
    assert s["analyze_mean_s"] == pytest.approx(10.5)
    assert s["analyze_p95_s"] == 19.0         # sorted(v)[int(.95 * 19)] = sorted(v)[18]
    assert s["notify_s"] == pytest.approx(2.0)
    s = snap(stats, 15.0)
    assert s["analyzed"] == 0 and s["notify_s"] == 0.0 and "analyze_p95_s" not in s


def test_stats_summary_key_order_and_counters():
    stats = stream.Stats(FakeClock(0.0))
    stats.add(failed=2, short=1)
    stats.add(restarts=1, refused=1)
    stats.clip_queued(3)
    stats.analyzed(1.0, 0.0)
    s = stats.snapshot_and_reset(5.0, queue_len=1, cameras=4, cameras_up=3, cpu_s=2.0, rss_mb=600.0,
                                 ffmpeg_cpu_s=1.5, ffmpeg_rss_mb=90.0)
    assert list(s) == ["window_s", "cameras", "cameras_up", "clips", "analyzed", "dropped", "short",
                       "failed", "restarts", "refused", "analyze_mean_s", "analyze_p95_s", "duty",
                       "notify_s", "queue_max", "queue_len", "cpu_s", "ffmpeg_cpu_s", "rss_mb",
                       "ffmpeg_rss_mb"]
    assert (s["cameras"], s["cameras_up"], s["clips"], s["failed"], s["short"], s["restarts"],
            s["refused"], s["queue_max"], s["queue_len"]) == (4, 3, 1, 2, 1, 1, 1, 3, 1)
    assert "errors" not in s
    with pytest.raises(KeyError):
        stats.add(errors=1)
    with pytest.raises(KeyError):
        stats.add(clips=1, bogus=1)
    assert stats.totals()["clips"] == 1       # nothing applied from the rejected call
    s = snap(stats, 10.0, queue_len=2)
    assert s["queue_max"] == 1                # reset to the queue length at the last snapshot
    assert s["ffmpeg_cpu_s"] is None and s["ffmpeg_rss_mb"] is None
    assert stats.totals() == {"clips": 1, "analyzed": 1, "dropped": 0, "short": 1, "failed": 2,
                              "restarts": 1, "refused": 1}


def test_stats_note_drop_first_once_per_window():
    stats = stream.Stats(FakeClock(0.0))
    assert stats.note_drop() == (True, 1)
    assert stats.note_drop() == (False, 2)
    assert snap(stats, 1.0)["dropped"] == 2
    assert stats.note_drop() == (True, 3)
    assert stats.note_drop() == (False, 4)


def test_stats_counter_race_totals_40000():
    stats = stream.Stats(time.monotonic)
    go = threading.Event()

    def work():
        go.wait()
        for _ in range(10_000):
            stats.add(clips=1)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    go.set()
    total = 0
    while any(t.is_alive() for t in threads):
        total += snap(stats, time.monotonic())["clips"]
    for t in threads:
        t.join()
    total += snap(stats, time.monotonic())["clips"]
    assert total == 40_000
    assert stats.totals()["clips"] == 40_000


# ----------------------------------------------------------------- 14: startup cleanup
def test_cleanup_stale_only_touches_pattern_files(tmp_path, caplog):
    seg = tmp_path / "seg"
    stale_cam = make_wav(seg / "00-front" / "r0003_20260101T000000Z.wav", 0.1)
    stale_q = make_wav(seg / "queued" / "00-front_r0003_20260101T000000Z.wav", 0.1)
    keep = [seg / "00-front" / "notes.txt", seg / "queued" / "notes.txt",
            seg / "other" / "r0001_20260101T000000Z.wav"]
    for k in keep:
        k.parent.mkdir(parents=True, exist_ok=True)
        k.write_text("keep")
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert stream.cleanup_stale(seg) == 2
    assert not stale_cam.exists() and not stale_q.exists()
    assert all(k.exists() for k in keep)
    assert "removed 2 stale segments" in caplog.text
    assert stream.cleanup_stale(seg) == 0


def test_prepare_segment_dir(tmp_path, caplog):
    seg = tmp_path / "a" / "seg"
    with caplog.at_level(logging.WARNING, logger="birdlisten"):
        stream.prepare_segment_dir(seg, peak_bytes=1)
    assert (seg / "queued").is_dir()
    assert not list(seg.glob(".probe-*"))
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger="birdlisten"):
        stream.prepare_segment_dir(seg, peak_bytes=10**18)
    assert "MB free" in caplog.text


def test_prepare_segment_dir_unwritable_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        stream.prepare_segment_dir(blocker / "seg", peak_bytes=1)


# ----------------------------------------------------------------- 18: handler + excepthook
def test_signal_handler_sets_only_flag_and_chains():
    class SpyStop:
        touched = 0

        def set(self):
            SpyStop.touched += 1
            raise AssertionError("stop.set called in the signal handler")

    flag = stream._SignalFlag()
    stop = SpyStop()
    calls = []
    handler = stream._make_handler(flag, {signal.SIGTERM: lambda s, f: calls.append(s),
                                          signal.SIGINT: signal.default_int_handler})
    handler(signal.SIGTERM, None)
    assert flag.value is True and calls == [signal.SIGTERM]
    flag.value = False
    handler(signal.SIGINT, None)              # default_int_handler would raise KeyboardInterrupt
    assert flag.value is True and calls == [signal.SIGTERM]
    assert SpyStop.touched == 0 and isinstance(stop, SpyStop)


def test_signal_handler_skips_non_callable_prev():
    flag = stream._SignalFlag()
    handler = stream._make_handler(flag, {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_IGN})
    handler(signal.SIGTERM, None)
    handler(signal.SIGINT, None)
    assert flag.value is True


def test_scrubbed_excepthook_logs_scrubbed_traceback(monkeypatch, caplog):
    monkeypatch.setattr(threading, "excepthook", stream._scrubbed_excepthook)

    def boom():
        raise RuntimeError("rtsp://admin:s3cret@h/x")

    with caplog.at_level(logging.ERROR, logger="birdlisten"):
        t = threading.Thread(target=boom, name="cam-garage")
        t.start(); t.join()
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert msgs[0].startswith("thread cam-garage crashed:\n")
    assert "Traceback" in msgs[0] and "RuntimeError" in msgs[0] and "rtsp://admin:***@h/x" in msgs[0]
    assert "s3cret" not in caplog.text


def test_scrubbed_excepthook_ignores_systemexit(monkeypatch, caplog):
    monkeypatch.setattr(threading, "excepthook", stream._scrubbed_excepthook)

    def bye():
        raise SystemExit(3)

    with caplog.at_level(logging.DEBUG, logger="birdlisten"):
        t = threading.Thread(target=bye)
        t.start(); t.join()
    assert not caplog.records


# ----------------------------------------------------------------- run guard
def test_run_guard_claim_release_and_threads():
    assert stream._claim_run() is None
    assert stream._RUN["active"] is True
    assert stream._claim_run() == 1           # active, no threads alive yet
    release = threading.Event()
    t = threading.Thread(target=release.wait, name="cam-x")
    stream._track(t)
    t.start()
    assert stream._guard_threads_alive()
    stream._release_run()
    assert stream._RUN["active"] is False
    assert stream._RUN["threads"] == [t]      # kept on purpose
    assert stream._claim_run() == 1           # one earlier thread still alive
    release.set(); t.join(timeout=2)
    assert not stream._guard_threads_alive()
    assert stream._claim_run() is None        # frees itself once the thread ended
    assert stream._RUN["threads"] == []
    stream._release_run()


# ----------------------------------------------------------------- 20: /proc sums
def _stat(pid: int, comm: str, utime: int, stime: int) -> str:
    rest = ["S"] + ["0"] * 10 + [str(utime), str(stime)] + ["0"] * 10   # rest[11], rest[12]
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


def _proc(root: Path, pid: int, stat: str | None, status: str | None) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    if stat is not None:
        (d / "stat").write_text(stat)
    if status is not None:
        (d / "status").write_text(status)


def test_proc_sums_fake_proc(tmp_path):
    root = tmp_path / "proc"
    _proc(root, 101, _stat(101, "ffmpeg", 200, 100), "Name:\tffmpeg\nVmRSS:\t    2048 kB\n")
    _proc(root, 102, _stat(102, "(ff mpeg) x)", 50, 50), "Name:\tffmpeg\nVmRSS:\t1024 kB\n")
    snapshot = [(c, SimpleNamespace(pid=p)) for c, p in (("a", 101), ("b", 102), ("c", 103))]
    cpu, rss, prev = stream._proc_sums(snapshot, {101: 1.0, 999: 5.0}, proc_root=root, clk_tck=100)
    assert cpu == pytest.approx((3.0 - 1.0) + 1.0)
    assert rss == pytest.approx(3.0)
    assert prev == {101: 3.0, 102: 1.0}


def test_proc_sums_skips_odd_files(tmp_path):
    root = tmp_path / "proc"
    _proc(root, 104, "104 (ffmpeg) S 1 2\n", "VmRSS:\t1024 kB\n")              # truncated stat
    _proc(root, 105, _stat(105, "ffmpeg", 10, 10), "Name:\tffmpeg\nState:\tZ\n")  # zombie, no VmRSS
    _proc(root, 106, None, None)                                                  # nothing readable
    _proc(root, 107, _stat(107, "ffmpeg", 100, 0), "VmRSS:\t512 kB\n")
    procs = [(str(p), SimpleNamespace(pid=p)) for p in (104, 105, 106)]
    assert stream._proc_sums(procs, {}, proc_root=root, clk_tck=100) == (None, None, {})
    cpu, rss, prev = stream._proc_sums(procs + [("ok", SimpleNamespace(pid=107))], {},
                                       proc_root=root, clk_tck=100)
    assert (cpu, rss, prev) == (pytest.approx(1.0), pytest.approx(0.5), {107: 1.0})


def test_proc_sums_missing_root(tmp_path):
    assert stream._proc_sums([("a", SimpleNamespace(pid=1))], {1: 2.0},
                             proc_root=tmp_path / "nope", clk_tck=100) == (None, None, {})


# ----------------------------------------------------------------- credentials
def test_no_password_in_log_format_strings():
    src = Path(stream.__file__).read_text()
    assert not re.search(r"rtsps?://[^:/@\s\"']+:[^*@\s\"']+@", src)


# ----------------------------------------------------------------- FakeFfmpeg + supervisor rig
# Shrunk timings for every thread-level test (FEAT-004/005 reuse this).
FAST = stream.StreamTuning(poll_s=0.02, tick_s=0.01, stall_s=0.5, term_grace_s=0.2, supervisor_join_s=0.5,
                           shutdown_budget_s=1.0, backoff_base_s=0.01, backoff_cap_s=0.04, healthy_run_s=0.6,
                           stagger_s=0.05, still_failing_log_s=0.3, min_segment_s=0.1, fatal_wait_s=0.2,
                           worker_ready_s=0.5, once_worker_per_clip_s=5)
BASE = dt.datetime(2026, 10, 5, 23, 15, 0, tzinfo=UTC)
SECRET_URL = "rtsp://admin:s3cret@10.0.0.2/x"


def _wav_header(nframes: int) -> bytes:
    """Canonical 44-byte header, 48 kHz mono s16."""
    data = nframes * 2
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data, b"WAVE", b"fmt ", 16, 1, 1,
                       48000, 96000, 2, 16, b"data", data)


class FakeFfmpeg:
    """Stands in for the ffmpeg segmenter (design 'Test seams'). A writer thread
    expands the -strftime pattern in cmd[-1] with a synthetic UTC clock
    (BASE + n * -segment_time, or `starts`) and writes valid 48 kHz mono WAVs.

    n_segments     segments to write, None = forever
    seg_s          audio seconds per segment (0.2 = 9600 frames)
    chunks/chunk_s write each segment in `chunks` pieces, chunk_s apart (a growing newest file)
    gap_s          pause before each segment after the first
    delay_next_s   extra pause before the second segment
    exit_after_s   pause after the last segment before exiting with exit_code
    exit_code      rc of the normal end; stderr_lines are emitted just before any exit
    stall_after    after this many segments stay alive and write nothing (until terminate/kill)
    ignore_terminate  terminate() does nothing; kill() always works
    duplicate_name the last segment reuses the first segment's name
    self_exit_rc   rc used by interrupt() (a terminal Ctrl-C reaching ffmpeg first)
    reap_delay_s   poll()/wait() report the exit only this long after it
    blocking_stderr  stderr iteration blocks before EOF until stderr_release is set
    starts         explicit start datetimes, one per segment
    """

    _pids = itertools.count(1000)

    def __init__(self, cmd, env, *, n_segments=2, seg_s=0.2, chunks=1, chunk_s=0.0, gap_s=0.05,
                 delay_next_s=0.0, exit_after_s=0.0, exit_code=0, stderr_lines=(), stall_after=None,
                 ignore_terminate=False, duplicate_name=False, self_exit_rc=255, reap_delay_s=0.0,
                 blocking_stderr=False, starts=None):
        self.cmd, self.env, self.pid = cmd, env, next(self._pids)
        self.pattern = cmd[-1]
        self.clip_seconds = int(cmd[cmd.index("-segment_time") + 1])
        self.n_segments, self.seg_s, self.chunks, self.chunk_s = n_segments, seg_s, chunks, chunk_s
        self.gap_s, self.delay_next_s, self.exit_after_s = gap_s, delay_next_s, exit_after_s
        self.exit_code, self.stderr_lines, self.stall_after = exit_code, list(stderr_lines), stall_after
        self.ignore_terminate, self.duplicate_name = ignore_terminate, duplicate_name
        self.self_exit_rc, self.reap_delay_s = self_exit_rc, reap_delay_s
        self.blocking_stderr, self.starts = blocking_stderr, starts
        self.calls: list[tuple[str, float]] = []      # ("terminate"|"kill"|"wait"|"reaped", monotonic)
        self.created: list[tuple[Path, float]] = []   # recorded just before the file is opened
        self.closed: list[tuple[Path, float]] = []
        self.returncode = None
        self.exited_mono = None
        self.stderr_release = threading.Event()
        self._reap_at = 0.0
        self._exited = threading.Event()
        self._io = threading.Lock()                   # no file write after the exit is visible
        self._lines: queue_mod.Queue = queue_mod.Queue()
        self.stderr = self._iter_stderr()
        self._writer = threading.Thread(target=self._write, daemon=True,
                                        name=f"fake-ffmpeg-{Path(self.pattern).parent.name}")
        self._writer.start()

    # ProcLike
    def poll(self):
        if self._exited.is_set() and time.monotonic() >= self._reap_at:
            return self.returncode
        return None

    def wait(self, timeout=None):
        self.calls.append(("wait", time.monotonic()))
        end = None if timeout is None else time.monotonic() + timeout
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        left = self._reap_at - time.monotonic()
        if left > 0:
            if end is not None and time.monotonic() + left > end:
                time.sleep(max(0.0, end - time.monotonic()))
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            time.sleep(left)
        self.calls.append(("reaped", time.monotonic()))
        return self.returncode

    def terminate(self):
        self.calls.append(("terminate", time.monotonic()))
        if not self.ignore_terminate:
            self._exit(255)

    def kill(self):
        self.calls.append(("kill", time.monotonic()))
        self._exit(-9)

    # test controls
    def interrupt(self):
        """SIGINT from the terminal reached ffmpeg before the supervisor saw stop."""
        self._exit(self.self_exit_rc)

    def alive(self) -> bool:
        return not self._exited.is_set()

    def names(self) -> list[str]:
        return [p.name for p, _ in self.created]

    # internals
    def _iter_stderr(self):
        while True:
            line = self._lines.get()
            if line is None:
                if self.blocking_stderr:
                    self.stderr_release.wait()
                return
            yield line

    def _exit(self, rc):
        with self._io:
            if self._exited.is_set():
                return
            self.exited_mono = time.monotonic()
            self.returncode = rc
            self._reap_at = self.exited_mono + self.reap_delay_s
            for line in self.stderr_lines:
                self._lines.put(line + "\n")
            self._lines.put(None)
            self._exited.set()

    def _start_of(self, i: int) -> dt.datetime:
        if self.starts:
            return self.starts[i]
        if self.duplicate_name and i and i == (self.n_segments or 0) - 1:
            i = 0
        return BASE + dt.timedelta(seconds=i * self.clip_seconds)

    def _segment(self, i: int) -> bool:
        path = Path(self._start_of(i).strftime(self.pattern))
        frames = int(self.seg_s * 48000)
        fill = bytes([i % 250 + 1, 0])                # distinct content per segment
        with self._io:
            if self._exited.is_set():
                return False
            self.created.append((path, time.monotonic()))
            f = open(path, "wb", buffering=0)
            f.write(_wav_header(0))
        written = 0
        try:
            for k in range(self.chunks):
                if k and self._exited.wait(self.chunk_s):
                    return False
                n = frames // self.chunks + (frames % self.chunks if k == self.chunks - 1 else 0)
                with self._io:
                    if self._exited.is_set():
                        return False
                    f.write(fill * n)
                    written += n
                    f.seek(0); f.write(_wav_header(written)); f.seek(0, 2)
        finally:
            f.close()
        self.closed.append((path, time.monotonic()))
        return True

    def _write(self):
        i = 0
        while self.n_segments is None or i < self.n_segments:
            if self.stall_after is not None and i >= self.stall_after:
                self._exited.wait()                   # alive and silent
                return
            if i and self._exited.wait(self.gap_s + (self.delay_next_s if i == 1 else 0.0)):
                return
            if not self._segment(i):
                return
            i += 1
        if self.stall_after is not None:
            self._exited.wait()
            return
        if not self._exited.wait(self.exit_after_s):
            self._exit(self.exit_code)


HOLD = dict(n_segments=0, stall_after=0)              # alive and silent: what a spawn gets once its plan runs out


class Spawner:
    """spawn seam keyed by the camera dir name in cmd[-1]. plans maps a dir name to
    a list of FakeFfmpeg option dicts (spawn k uses item k, HOLD after the list)
    or exceptions (raised from spawn)."""

    def __init__(self, plans: dict[str, list]):
        self.plans = plans
        self.fakes: dict[str, list[FakeFfmpeg]] = collections.defaultdict(list)
        self.spawns: dict[str, list[tuple[float, list[str], dict, list[str]]]] = collections.defaultdict(list)
        self._lock = threading.Lock()

    def __call__(self, cmd, env):
        d = Path(cmd[-1]).parent
        with self._lock:
            k = len(self.spawns[d.name])
            self.spawns[d.name].append((time.monotonic(), cmd, env, sorted(os.listdir(d))))
            plan = self.plans.get(d.name, [])
            item = plan[k] if k < len(plan) else HOLD
            if isinstance(item, BaseException):
                raise item
            fake = FakeFfmpeg(cmd, env, **item)
            self.fakes[d.name].append(fake)
            return fake

    def count(self, camdir: str) -> int:
        with self._lock:
            return len(self.spawns[camdir])

    def all_fakes(self) -> list[FakeFfmpeg]:
        with self._lock:
            return [f for fs in self.fakes.values() for f in fs]


def wait_for(pred, timeout=3.0, step=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    raise AssertionError(f"timed out waiting for {getattr(pred, '__name__', pred)}")


class Rig:
    """Supervisors standalone against a real ClipQueue, Stats and ProcRegistry.
    Use as a context manager: exit sets stop, joins every supervisor and makes
    sure no fake is left running."""

    def __init__(self, tmp_path, spawner, names=("front",), *, once=False, tuning=FAST,
                 signalled=lambda: False, clip_seconds=30):
        self.seg = tmp_path / "seg"
        stream.prepare_segment_dir(self.seg, peak_bytes=1)
        self.spawner = spawner
        self.stats = stream.Stats(time.monotonic)
        self.queue = stream.ClipQueue(1000, self.stats, time.monotonic)
        self.registry = stream.ProcRegistry()
        self.stop = threading.Event()
        self.sups = [stream.CameraSupervisor(
            i, bl.Camera(n, f"rtsp://admin:s3cret@10.0.0.{i + 2}/x"), seg_dir=self.seg,
            clip_seconds=clip_seconds, queue=self.queue, stats=self.stats, registry=self.registry,
            stop=self.stop, signalled=signalled, once=once, spawn=spawner, tuning=tuning,
            clock=time.monotonic, now_utc=stream._utcnow, rng=random.Random(i)) for i, n in enumerate(names)]
        self.t0 = None

    def __enter__(self):
        self.t0 = time.monotonic()
        for s in self.sups:
            stream._track(s)
            s.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for f in self.spawner.all_fakes():
            f.stderr_release.set()
        for s in self.sups:
            s.join(timeout=3)
        for f in self.spawner.all_fakes():
            if f.alive():
                f.kill()
            f._writer.join(timeout=1)
        assert not [s.name for s in self.sups if s.is_alive()]

    def take(self, n: int, timeout: float = 3.0) -> list[stream.Clip]:
        out, end = [], time.monotonic() + timeout
        while len(out) < n and time.monotonic() < end:
            c = self.queue.get(timeout=0.05)
            if isinstance(c, stream.Clip):
                out.append(c)
        assert len(out) == n, f"got {len(out)} clips, wanted {n}"
        return out


def _msgs(caplog, level=None, start=None) -> list[str]:
    return [r.getMessage() for r in list(caplog.records)
            if (level is None or r.levelno == level) and (start is None or r.getMessage().startswith(start))]


def _no_secret(caplog):
    assert not [m for m in _msgs(caplog) if "s3cret" in m]


def _approx_backoffs(got, want):
    assert len(got) >= len(want), got
    for g, w in zip(got, want):
        assert 0.8 * w - 1e-9 <= g <= 1.2 * w + 1e-9, (list(got), want)


# ----------------------------------------------------------------- 2: completion detection
def test_segment_queued_only_after_successor_or_exit(tmp_path, caplog):
    sp = Spawner({"00-front": [dict(n_segments=2, delay_next_s=0.3, exit_after_s=0.3, exit_code=0)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: sp.fakes["00-front"] and sp.fakes["00-front"][0].closed)
        fake = sp.fakes["00-front"][0]
        first = fake.created[0][0]
        time.sleep(0.15)                              # several polls while N is the newest file
        assert len(rig.queue) == 0 and first.exists()
        c0, = rig.take(1)
        assert len(fake.created) == 2 and c0.ready_mono >= fake.created[1][1]
        assert c0.path == rig.seg / "queued" / "00-front_r0001_20261005T231500Z.wav"
        assert c0.path.exists() and not first.exists()
        assert c0.camera == "front" and c0.start_utc == BASE and c0.duration_s == pytest.approx(0.2)
        c1, = rig.take(1)
        assert fake.exited_mono is not None and c1.ready_mono >= fake.exited_mono
        assert c1.path.name == "00-front_r0001_20261005T231530Z.wav"
        assert not list((rig.seg / "00-front").glob("r0001_*"))
    assert rig.stats.totals()["failed"] == 0
    assert "exit 0" in "\n".join(_msgs(caplog, logging.ERROR))


def test_backward_clock_step_uses_first_seen_order(tmp_path, caplog):
    starts = [dt.datetime(2026, 10, 5, 23, 15, 30, tzinfo=UTC), dt.datetime(2026, 10, 5, 23, 10, 0, tzinfo=UTC)]
    sp = Spawner({"00-front": [dict(n_segments=2, starts=starts, gap_s=0.1, exit_after_s=0.2)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        c0, c1 = rig.take(2)
        fake = sp.fakes["00-front"][0]
        assert fake.created[1][1] <= c0.ready_mono < fake.exited_mono   # queued once the successor appeared
        assert c1.ready_mono >= fake.exited_mono                        # the newest only after exit
        spawns_at_second = sp.count("00-front")
    assert [c.start_utc.isoformat() for c in (c0, c1)] == ["2026-10-05T23:15:30+00:00", "2026-10-05T23:10:00+00:00"]
    assert [c.path.name for c in (c0, c1)] == ["00-front_r0001_20261005T231530Z.wav",
                                               "00-front_r0001_20261005T231000Z.wav"]
    assert spawns_at_second == 1
    assert rig.stats.totals()["failed"] == 0
    text = "\n".join(_msgs(caplog))
    assert "bad segment" not in text and "stalled" not in text


# ----------------------------------------------------------------- 3: duplicate name
def test_duplicate_segment_name_dropped_and_first_kept(tmp_path, caplog):
    sp = Spawner({"00-front": [dict(n_segments=3, duplicate_name=True, gap_s=0.1, exit_after_s=0.1, exit_code=1)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        c0, c1 = rig.take(2)
        wait_for(lambda: rig.stats.totals()["failed"] == 1)
    fake = sp.fakes["00-front"][0]
    assert fake.names() == ["r0001_20261005T231500Z.wav", "r0001_20261005T231530Z.wav", "r0001_20261005T231500Z.wav"]
    assert c0.path.name == "00-front_r0001_20261005T231500Z.wav"
    assert c0.path.read_bytes()[44:46] == bytes([1, 0])      # segment 0's audio, not overwritten
    assert "front: duplicate segment name r0001_20261005T231500Z.wav, dropped" in _msgs(caplog, logging.INFO)
    assert rig.queue.get(timeout=0.05) is None
    assert not list((rig.seg / "00-front").glob("r*.wav"))


# ----------------------------------------------------------------- 5: restart, backoff, stagger
def test_restart_backoff_stagger_scrubbed(tmp_path, caplog):
    refused = dict(n_segments=0, exit_code=1, stderr_lines=[f"{SECRET_URL}: Connection refused"])
    sp = Spawner({"00-a": [refused] * 5, "01-b": [dict(n_segments=None, gap_s=0.03)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp, ("a", "b")) as rig:
        wait_for(lambda: sp.count("00-a") >= 5)
        n = rig.stats.totals()["clips"]
        wait_for(lambda: rig.stats.totals()["clips"] >= n + 2)       # B keeps producing
        a = rig.sups[0]
    _approx_backoffs(a.backoffs, [0.01, 0.02, 0.04, 0.04])
    runs = [re.search(r"/r(\d{4})_", cmd[-1]).group(1) for _, cmd, _, _ in sp.spawns["00-a"][:5]]
    assert runs == ["0001", "0002", "0003", "0004", "0005"]
    assert sp.spawns["00-a"][0][0] >= rig.t0
    assert sp.spawns["01-b"][0][0] >= rig.t0 + FAST.stagger_s
    assert sp.count("01-b") == 1
    _no_secret(caplog)
    timing = _msgs(caplog, logging.INFO, "timing camera=a ")
    assert len(timing) >= 4
    assert all("refused=1" in m and "Connection_refused" in m and "rtsp://admin:***@" in m for m in timing)
    assert re.fullmatch(r"timing camera=a up_s=\d+\.\d segments=0 backoff_s=0\.0 refused=1 "
                        r"error=ffmpeg_exited:_rtsp://admin:\*\*\*@10\.0\.0\.2/x:_Connection_refused", timing[0])
    errors = _msgs(caplog, logging.ERROR)
    assert len(errors) == 1
    assert re.fullmatch(r"a: ffmpeg exited after \d+\.\ds, 0 segments: rtsp://admin:\*\*\*@10\.0\.0\.2/x: "
                        r"Connection refused; retry in 0\.0s", errors[0])
    assert rig.stats.totals()["refused"] >= 4


def test_ctrl_c_exit_is_silent_and_cleanup_keeps_other_files(tmp_path, caplog):
    flag = SimpleNamespace(value=False)
    sp = Spawner({"00-front": [dict(n_segments=1, stall_after=1, self_exit_rc=255)]})
    rig = Rig(tmp_path, sp, signalled=lambda: flag.value)
    camdir = rig.seg / "00-front"
    leftover = make_wav(camdir / "r0001_20250101T000000Z.wav", 0.2)
    (camdir / "notes.txt").write_text("keep")
    with caplog.at_level(logging.INFO, logger="birdlisten"), rig:
        wait_for(lambda: sp.fakes["00-front"] and sp.fakes["00-front"][0].closed)
        flag.value = True                             # handler ran; main has not set stop yet
        sp.fakes["00-front"][0].interrupt()           # ffmpeg got the SIGINT too: rc 255
        rig.sups[0].join(timeout=2)
        assert not rig.sups[0].is_alive() and not rig.stop.is_set()
    assert sp.spawns["00-front"][0][3] == ["notes.txt"]      # per-spawn cleanup ran before the spawn
    assert not leftover.exists() and (camdir / "notes.txt").exists()
    assert sp.count("00-front") == 1
    assert not _msgs(caplog, logging.ERROR) and not _msgs(caplog, logging.WARNING)
    assert not [m for m in _msgs(caplog, start="timing ") if "error=" in m]
    assert rig.stats.totals()["restarts"] == 0


def test_spawn_oserror_backs_off_and_crash_is_scrubbed(tmp_path, caplog):
    sp = Spawner({"00-front": [FileNotFoundError(2, "No such file or directory", "ffmpeg"),
                               RuntimeError(f"boom {SECRET_URL}")]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: sp.count("00-front") >= 3)
    timing = _msgs(caplog, logging.INFO, "timing camera=front ")
    assert timing[0].startswith("timing camera=front up_s=0.0 segments=0 backoff_s=0.0 error=ffmpeg_exited:_[Errno_2]")
    errors = _msgs(caplog, logging.ERROR)
    assert errors[0].startswith("front: ffmpeg exited after 0.0s, 0 segments: [Errno 2] No such file")
    assert errors[1].startswith("front: supervisor crashed: Traceback")
    assert "RuntimeError: boom rtsp://admin:***@10.0.0.2/x" in errors[1]
    _no_secret(caplog)
    assert rig.stats.totals()["restarts"] == 2
    _approx_backoffs(rig.sups[0].backoffs, [0.01, 0.02])


# ----------------------------------------------------------------- 6: healthy-run reset
def test_healthy_run_resets_backoff(tmp_path, caplog):
    quick = dict(n_segments=2, gap_s=0.03, exit_after_s=0.03, exit_code=1, stderr_lines=["Connection timed out"])
    healthy = dict(n_segments=9, gap_s=0.1, exit_code=1, stderr_lines=["Connection timed out"])
    sp = Spawner({"00-front": [quick, quick, quick, healthy]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: sp.count("00-front") >= 5)
        backoffs = list(rig.sups[0].backoffs)
    _approx_backoffs(backoffs, [0.01, 0.02, 0.04, 0.01])    # hand-offs alone did not reset; the healthy run did
    assert "front: recovered after 3 attempts" in _msgs(caplog, logging.INFO)
    errors = _msgs(caplog, logging.ERROR)
    assert len(errors) == 2                           # a new streak after the recovery
    assert rig.stats.totals()["restarts"] == 4


# ----------------------------------------------------------------- 8: stall watchdog
def test_silent_ffmpeg_is_stalled_and_restarted(tmp_path, caplog):
    sp = Spawner({"00-front": [dict(n_segments=1, stall_after=1)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: sp.count("00-front") >= 2)
        c0, = rig.take(1)                             # the stalled run's last valid segment
    fake = sp.fakes["00-front"][0]
    term = [t for c, t in fake.calls if c == "terminate"]
    assert term and term[0] - fake.closed[0][1] >= FAST.stall_s - FAST.poll_s
    assert c0.path.name == "00-front_r0001_20261005T231500Z.wav"
    timing = _msgs(caplog, logging.INFO, "timing camera=front ")
    assert any(m.endswith("error=stalled:_no_audio_for_0.5s") for m in timing)
    assert any("stalled: no audio for 0.5s" in m for m in _msgs(caplog, logging.ERROR))
    assert rig.stats.totals()["restarts"] == 1


def test_growing_newest_is_not_stalled_when_older_files_vanish(tmp_path, caplog):
    sp = Spawner({"00-front": [dict(n_segments=2, chunks=8, chunk_s=0.08, gap_s=0.02, exit_after_s=0.1)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        c0, = rig.take(1)
        c0.path.unlink()                              # the worker deleted it mid-run
        c1, = rig.take(1, timeout=4)
    fake = sp.fakes["00-front"][0]
    assert not [c for c, _ in fake.calls if c == "terminate"]
    assert fake.closed[1][1] - fake.created[0][1] > 2 * FAST.stall_s   # ran well past stall_s
    assert c1.duration_s == pytest.approx(0.2)
    assert "stalled" not in "\n".join(_msgs(caplog))


# ----------------------------------------------------------------- 9: log rate limit
def test_refusals_rate_limit_logs(tmp_path, caplog):
    bw = dict(n_segments=0, exit_code=1, exit_after_s=0.05, stderr_lines=["method DESCRIBE failed: 453 Not Enough Bandwidth"])
    other = dict(n_segments=0, exit_code=1, exit_after_s=0.05, stderr_lines=["Connection timed out"])
    sp = Spawner({"00-front": [bw] * 10 + [other]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: len(_msgs(caplog, logging.INFO, "timing camera=front ")) >= 11, timeout=5)
    recs = [r for r in caplog.records if r.name == "birdlisten"]
    timing = [r.getMessage() for r in recs if r.getMessage().startswith("timing camera=front ")]
    assert all("refused=1" in m and "453_Not_Enough_Bandwidth" in m for m in timing[:10])
    assert "refused=1" not in timing[10]
    errors = [r for r in recs if r.levelno == logging.ERROR]
    assert len(errors) == 2
    assert "453 Not Enough Bandwidth; retry in" in errors[0].getMessage()
    assert "Connection timed out" in errors[1].getMessage()
    warns = [r for r in recs if r.levelno == logging.WARNING]
    assert warns, "expected at least one still-failing WARNING"
    times = [errors[0].created] + [w.created for w in warns]
    assert all(b - a >= FAST.still_failing_log_s - 0.02 for a, b in zip(times, times[1:]))
    assert all(errors[0].created < w.created < errors[1].created for w in warns)
    for w in warns:
        assert re.fullmatch(r"front: still failing \(\d+ attempts since \d\d:\d\d:\d\dZ\): method DESCRIBE failed: "
                            r"453 Not Enough Bandwidth; retry in 0\.0s", w.getMessage())
    assert rig.stats.totals()["refused"] == 10


# ----------------------------------------------------------------- 13: registry before reader.join
def test_registry_removed_before_reader_join(tmp_path, caplog):
    sp = Spawner({"00-front": [dict(n_segments=1, exit_after_s=0.2, exit_code=1, blocking_stderr=True,
                                    stderr_lines=["boom"])]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp) as rig:
        wait_for(lambda: sp.fakes["00-front"])
        fake = sp.fakes["00-front"][0]
        wait_for(lambda: rig.registry.snapshot() == [("front", fake)])
        wait_for(lambda: not fake.alive())
        wait_for(lambda: rig.registry.snapshot() == [], timeout=0.5)
        reader = next(t for t in stream._RUN["threads"] if t.name == "cam-front-stderr")
        assert reader.is_alive() and rig.sups[0].is_alive()   # supervisor is inside reader.join
        assert not _msgs(caplog, start="timing camera=front ")
        fake.stderr_release.set()
        wait_for(lambda: _msgs(caplog, start="timing camera=front "))
        reader.join(timeout=1)
    assert "error=ffmpeg_exited:_boom" in _msgs(caplog, start="timing camera=front ")[0]


# ----------------------------------------------------------------- once mode (supervisor part)
def test_once_mode_one_clip_or_one_error(tmp_path, caplog):
    sp = Spawner({"00-ok": [dict(n_segments=None, gap_s=0.05)],
                  "01-bad": [dict(n_segments=0, exit_code=1, stderr_lines=["Connection refused"])]})
    with caplog.at_level(logging.INFO, logger="birdlisten"), Rig(tmp_path, sp, ("ok", "bad"), once=True) as rig:
        for s in rig.sups:
            s.join(timeout=3)
            assert not s.is_alive()                   # both ended without stop
        assert not rig.stop.is_set()
    clips = [rig.queue.get(timeout=0.05) for _ in range(2)]
    assert clips[0].camera == "ok" and clips[1] is None
    assert sp.count("00-ok") == 1 and sp.count("01-bad") == 1
    assert [c for c, _ in sp.fakes["00-ok"][0].calls][:1] == ["terminate"]
    assert not list((rig.seg / "00-ok").glob("r*.wav"))
    errors = _msgs(caplog, logging.ERROR)
    assert len(errors) == 1
    assert re.fullmatch(r"bad: ffmpeg exited after \d\.\ds, 0 segments: Connection refused", errors[0])
    assert not [m for m in _msgs(caplog, start="timing camera=bad ") if "backoff_s" in m]
