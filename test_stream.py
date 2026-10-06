"""Tests for stream.py (CAPTURE_MODE=stream). Network-free, no real ffmpeg.
Run: uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q test_stream.py"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import itertools
import logging
import os
import queue as queue_mod
import random
import re
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
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
    for r in BgRun.live:                      # a background run_stream a failed test left behind
        r.flag.value = True
        if r.thread.is_alive():
            r.thread.join(timeout=5)
    BgRun.live.clear()
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


# ----------------------------------------------------------------- run_stream helpers
BUSHTIT = bl.Detection("Bushtit", "P. minimus", 0.9, 3.0, 6.0)
HEALTH_RE = re.compile(r"(?i)traceback|error|exception|fatal|panic")      # fleet/bin/health.sh
SUMMARY_KEYS = ["window_s", "final", "cameras", "cameras_up", "clips", "analyzed", "dropped", "short", "failed",
                "restarts", "refused", "analyze_mean_s", "analyze_p95_s", "duty", "notify_s", "queue_max",
                "queue_len", "cpu_s", "ffmpeg_cpu_s", "rss_mb", "ffmpeg_rss_mb"]
OPTIONAL_KEYS = {"final", "analyze_mean_s", "analyze_p95_s", "ffmpeg_cpu_s", "ffmpeg_rss_mb"}


class Die(BaseException):
    """Escapes every `except Exception`, so it kills the thread it is raised in."""


def stream_cfg(tmp_path, names=("front",), **env) -> bl.Config:
    cams = ",".join(f"{n}=rtsp://admin:s3cret@10.0.0.{i + 2}/x" for i, n in enumerate(names))
    return bl.load_config({"CAMERAS": cams, "LATITUDE": "0", "LONGITUDE": "0",
                           "DATA_DIR": str(tmp_path / "data"), "SEGMENT_DIR": str(tmp_path / "seg"), **env})


@pytest.fixture
def spies(monkeypatch):
    """bl.record runs for real and bl.notify is a no-op; both note the calling thread."""
    s = SimpleNamespace(record=[], notify=[])
    real_record = bl.record

    def record(conn, when, cam, d, clip):
        s.record.append((threading.current_thread().name, when.isoformat(timespec="seconds"), cam.name, d.common_name))
        real_record(conn, when, cam, d, clip)

    def notify(cfg, title, body):
        s.notify.append((threading.current_thread().name, title, body))

    monkeypatch.setattr(bl, "record", record)
    monkeypatch.setattr(bl, "notify", notify)
    return s


@pytest.fixture
def tracked(monkeypatch):
    """Every thread run_stream (or a supervisor) registers with the run guard."""
    out: list[threading.Thread] = []
    real = stream._track

    def spy(t):
        out.append(t)
        real(t)

    monkeypatch.setattr(stream, "_track", spy)
    return out


@pytest.fixture
def registries(monkeypatch):
    out: list[stream.ProcRegistry] = []

    class SpyRegistry(stream.ProcRegistry):
        def __init__(self):
            super().__init__()
            out.append(self)

    monkeypatch.setattr(stream, "ProcRegistry", SpyRegistry)
    return out


@pytest.fixture
def install_hook(monkeypatch):
    """pytest's threadexception plugin swaps threading.excepthook after fixture
    setup, so the test body calls this to install a sentinel to compare against."""
    def install():
        def sentinel(args):
            pass
        monkeypatch.setattr(threading, "excepthook", sentinel)
        return sentinel
    return install


def run(cfg, spawner, **kw) -> int:
    kw.setdefault("stop", threading.Event())
    kw.setdefault("load_model", lambda: None)
    kw.setdefault("analyze_fn", lambda w, c, t: [])
    kw.setdefault("tuning", FAST)
    kw.setdefault("rng", random.Random(0))
    return stream.run_stream(cfg, spawn=spawner, **kw)


class BgRun:
    """run_stream in a background thread; signal() plays the part of SIGTERM.
    A run a failing test left behind is signalled and joined at teardown."""

    live: list[BgRun] = []

    def __init__(self, cfg, spawner, **kw):
        BgRun.live.append(self)
        self.flag = SimpleNamespace(value=False)
        self.rc = self.error = self.returned = self.sig_at = None
        kw.setdefault("signalled", lambda: self.flag.value)
        self.thread = threading.Thread(target=self._main, args=(cfg, spawner, kw), name="test-run-stream", daemon=True)

    def _main(self, cfg, spawner, kw):
        try:
            self.rc = run(cfg, spawner, **kw)
        except BaseException as exc:                  # noqa: BLE001 -- re-raised in join()
            self.error = exc
        finally:
            self.returned = time.monotonic()

    def start(self):
        self.thread.start()
        return self

    def signal(self):
        self.sig_at = time.monotonic()
        self.flag.value = True

    def join(self, timeout=5.0) -> int:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "run_stream did not return"
        if self.error is not None:
            raise self.error
        return self.rc


def _summaries(caplog) -> list[dict[str, str]]:
    return [dict(t.split("=", 1) for t in m.split()[2:]) for m in _msgs(caplog, logging.INFO, "timing summary ")]


def _spawned(sp: Spawner) -> int:
    with sp._lock:
        return sum(len(v) for v in sp.spawns.values())


def _rows(cfg) -> list[tuple]:
    conn = sqlite3.connect(cfg.data_dir / "birdlisten.sqlite")
    try:
        return conn.execute("SELECT heard_at, camera, common_name, clip_offset_s, clip_path FROM detections"
                            " ORDER BY camera, heard_at").fetchall()
    finally:
        conn.close()


def _reaped(f: FakeFfmpeg) -> bool:
    return not f.alive() and ("reaped" in [c for c, _ in f.calls] or f.poll() is not None)


# ----------------------------------------------------------------- 10: worker storage
def test_worker_stores_segment_start_keeps_clip_and_applies_cooldown(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path, KEEP_CLIPS="1")
    conn = bl.open_db(cfg.data_dir)                   # an aware row from 15 min earlier
    conn.execute("INSERT INTO notified VALUES (?, ?)", ("Bushtit", "2026-10-05T23:00:00+00:00"))
    conn.commit(); conn.close()
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})
    seen = []

    def analyze(w, c, when):
        seen.append((Path(w), Path(w).exists(), when, threading.current_thread().name))
        return [bl.Detection("Bushtit", "P. minimus", 0.6, 9.0, 12.0), BUSHTIT]

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, once=True, analyze_fn=analyze) == 0
    assert len(seen) == 1
    wav, existed, when, thread = seen[0]
    assert wav == cfg.segment_dir / "queued" / "00-front_r0001_20261005T231500Z.wav"
    assert existed and not wav.exists()
    assert when == BASE and thread == "birdlisten-worker"
    kept = cfg.data_dir / "clips" / "2026-10-05" / "231500_front.wav"
    assert _rows(cfg) == [("2026-10-05T23:15:00+00:00", "front", "Bushtit", 3.0, str(kept))]
    assert kept.exists()
    assert spies.notify == []                         # cooldown from the existing aware row
    assert not list(cfg.segment_dir.rglob("*.wav"))
    ok = _msgs(caplog, logging.INFO, "timing camera=front ")
    assert len(ok) == 1 and re.fullmatch(
        r"timing camera=front clip_s=0\.2 segment_lag_s=-?\d+\.\d\d queue_wait_s=\d+\.\d\d analyze_s=\d+\.\d\d "
        r"detections=1 cpu_s=-?\d+\.\d\d rss_mb=\d+", ok[0])


def test_worker_nothing_above_uses_real_segment_length(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05, seg_s=2.0)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, once=True) == 0
    assert "front: 2s, nothing above 0.50" in _msgs(caplog, logging.INFO)
    assert _rows(cfg) == []


def test_worker_failure_is_counted_and_scrubbed(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})

    def analyze(w, c, t):
        raise RuntimeError(f"database is locked {SECRET_URL}")

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, once=True, analyze_fn=analyze) == 1     # nothing analyzed
    assert _msgs(caplog, logging.ERROR) == ["front: database is locked rtsp://admin:***@10.0.0.2/x"]
    line, = [m for m in _msgs(caplog, logging.INFO, "timing camera=front ") if "error=" in m]
    assert line.endswith("error=database_is_locked_rtsp://admin:***@10.0.0.2/x") and "detections" not in line
    assert _summaries(caplog)[-1]["failed"] == "1"
    _no_secret(caplog)
    assert not list(cfg.segment_dir.rglob("*.wav"))


# ----------------------------------------------------------------- 7: restart while queued
def test_clip_queued_across_respawn_is_analyzed(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=2, gap_s=0.05, exit_after_s=0.05, exit_code=1,
                                    stderr_lines=["Connection reset by peer"])]})
    release = threading.Event()
    seen = []

    def analyze(w, c, t):
        seen.append((Path(w).name, Path(w).exists()))
        release.wait(5)
        return []

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, sp, analyze_fn=analyze).start()
        wait_for(lambda: len(seen) == 1 and sp.count("00-front") >= 2)   # worker blocked; A respawned
        queued = cfg.segment_dir / "queued" / "00-front_r0001_20261005T231530Z.wav"
        assert queued.exists()                        # the respawn's cleanup left queued/ alone
        release.set()
        wait_for(lambda: len(seen) == 2)
        r.signal()
        assert r.join() == 0
    assert seen == [("00-front_r0001_20261005T231500Z.wav", True), ("00-front_r0001_20261005T231530Z.wav", True)]
    sums = _summaries(caplog)
    assert sum(int(s["failed"]) for s in sums) == 0 and sum(int(s["analyzed"]) for s in sums) == 2


# ----------------------------------------------------------------- 11: concurrency and cooldown
def test_three_cameras_one_worker_one_notify(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path, ("a", "b", "c"))
    plan = [dict(n_segments=3, gap_s=0.03, exit_after_s=0.03)]
    sp = Spawner({"00-a": plan, "01-b": plan, "02-c": plan})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, sp, analyze_fn=lambda w, c, t: [BUSHTIT]).start()
        wait_for(lambda: len(spies.record) >= 9, timeout=5)
        r.signal()
        assert r.join() == 0
    assert len(spies.record) == 9
    assert {th for th, *_ in spies.record} == {"birdlisten-worker"}
    assert [(th, title) for th, title, _ in spies.notify] == [("birdlisten-worker", "Bushtit")]
    want = [(BASE + dt.timedelta(seconds=30 * k)).isoformat() for k in range(3)]
    rows = _rows(cfg)
    assert [r[:2] for r in rows] == [(t, cam) for cam in "abc" for t in want]


# ----------------------------------------------------------------- 12: summary integration + health grep
def test_summary_lines_and_health_grep(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=0, exit_code=1, stderr_lines=["Connection timed out"]),
                               dict(n_segments=None, gap_s=0.05)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, sp, analyze_fn=lambda w, c, t: [BUSHTIT], summary_s=0.2).start()
        wait_for(lambda: "front: recovered after 1 attempts" in _msgs(caplog) and len(_summaries(caplog)) >= 2,
                 timeout=5)
        r.signal()
        assert r.join() == 0
    sums = _summaries(caplog)
    for s in sums:
        assert list(s) == [k for k in SUMMARY_KEYS if k in s]
        assert set(SUMMARY_KEYS) - OPTIONAL_KEYS <= set(s)
        assert "errors" not in s and s["cameras"] == "1"
    assert [s.get("final") for s in sums] == [None] * (len(sums) - 1) + ["1"]
    assert sum(int(s["restarts"]) for s in sums) == 1
    assert sum(int(s["analyzed"]) for s in sums) >= 1
    assert any("analyze_p95_s" in s for s in sums)

    info = _msgs(caplog, logging.INFO)
    normal = [m for m in info if " error=" not in m]  # failure timing lines are meant to match
    for want in (f"stream mode: 1 cameras, queue 2, summary every 5 min, segments {cfg.segment_dir}",
                 "front: recovered after 1 attempts", "stopping capture (signal)", "stream pipeline stopped"):
        assert want in normal
    assert [m for m in normal if m.startswith("timing camera=front clip_s=") and "detections=1" in m]
    assert [m for m in normal if m.startswith("timing summary ")]
    assert not [m for m in normal if HEALTH_RE.search(m)]


# ----------------------------------------------------------------- 13: shutdown
def test_signal_shutdown_within_budget(tmp_path, caplog, spies, registries):
    cfg = stream_cfg(tmp_path, ("a", "b"))
    sp = Spawner({"00-a": [dict(n_segments=None, gap_s=0.1, ignore_terminate=True)],
                  "01-b": [dict(n_segments=None, gap_s=0.1)]})

    class SpyStop(threading.Event):
        set_at = None

        def set(self):
            if self.set_at is None:
                self.set_at = time.monotonic()
            super().set()

    stop = SpyStop()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, sp, stop=stop, analyze_fn=lambda w, c, t: time.sleep(0.15) or []).start()
        wait_for(lambda: len(list((cfg.segment_dir / "queued").glob("*.wav"))) >= 2)   # clips waiting
        r.signal()
        assert r.join() == 0
    assert r.returned - r.sig_at <= FAST.shutdown_budget_s + 0.5
    assert stop.set_at - r.sig_at <= 2 * FAST.tick_s + 0.05
    a = sp.fakes["00-a"][0]
    term = next(t for c, t in a.calls if c == "terminate")
    kill = next(t for c, t in a.calls if c == "kill")
    assert kill - term >= FAST.term_grace_s - 0.01
    assert sp.all_fakes() and all(_reaped(f) for f in sp.all_fakes())
    assert not list(cfg.segment_dir.rglob("*.wav"))
    assert registries[0].snapshot() == []
    assert _summaries(caplog)[-1].get("final") == "1"


def test_final_sweep_kills_all_then_waits_one_deadline(tmp_path):
    camdir = tmp_path / "00-x"
    camdir.mkdir()
    cmd = ["ffmpeg", "-segment_time", "30", str(camdir / "r0001_%Y%m%dT%H%M%SZ.wav")]
    fakes = [FakeFfmpeg(cmd, {}, n_segments=0, stall_after=0, reap_delay_s=0.4) for _ in range(4)]
    reg = stream.ProcRegistry()
    for i, f in enumerate(fakes):
        reg.add(f"c{i}", f)
    t0 = time.monotonic()
    stream._sweep(reg, time.monotonic)
    assert time.monotonic() - t0 <= 0.6
    kills = [t for f in fakes for c, t in f.calls if c == "kill"]
    waits = [t for f in fakes for c, t in f.calls if c == "wait"]
    assert len(kills) == 4 and len(waits) == 4 and max(kills) < min(waits)
    assert not [f for f in fakes if f.alive()]
    assert reg.add("late", object()) is False
    stream._sweep(reg, time.monotonic)               # idempotent
    for f in fakes:
        f._writer.join(timeout=1)


def test_worker_takes_no_clip_after_stop(tmp_path):
    cfg = stream_cfg(tmp_path)
    stats = stream.Stats(time.monotonic)
    q = stream.ClipQueue(4, stats, time.monotonic)
    stop = threading.Event()
    calls = []
    w = stream.Worker(cfg, queue=q, stats=stats, stop=stop, cameras={c.name: c for c in cfg.cameras},
                      analyze_fn=lambda *a: calls.append(a) or [], clock=time.monotonic)
    stream._track(w)
    w.start()
    assert w.ready.wait(2) and w.open_error is None
    time.sleep(0.05)                                  # blocked in get() on the empty queue
    c1, c2 = make_clip(tmp_path / "q" / "1.wav"), make_clip(tmp_path / "q" / "2.wav")
    with q.cond:                                      # the worker cannot wake in between
        q.put(c1); q.put(c2)
        stop.set(); q.close()
    w.join(timeout=2)
    assert not w.is_alive() and calls == []
    assert not c1.path.exists()                       # the clip it took is deleted, not analyzed
    assert c2.path.exists() and q.drain() == 1
    assert stats.totals()["failed"] == 0 and stats.totals()["analyzed"] == 0


# ----------------------------------------------------------------- 15: once mode
def test_once_mode_one_camera_failing(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path, ("ok", "bad"))
    sp = Spawner({"00-ok": [dict(n_segments=None, gap_s=0.05)],
                  "01-bad": [dict(n_segments=0, exit_code=1, stderr_lines=["Connection refused"])]})
    seen = []

    def analyze(w, c, t):
        seen.append(Path(w).exists())
        time.sleep(0.2)
        return []

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, once=True, analyze_fn=analyze) == 0
    assert seen == [True]
    assert sp.count("00-ok") == 1 and sp.count("01-bad") == 1     # no retry
    s = _summaries(caplog)[-1]
    assert (s["final"], s["analyzed"], s["restarts"]) == ("1", "1", "1")
    assert all(_reaped(f) for f in sp.all_fakes())


def test_once_mode_every_camera_failing_returns_1(tmp_path, caplog, spies):
    cfg = stream_cfg(tmp_path, ("a", "b"))
    bad = [dict(n_segments=0, exit_code=1, stderr_lines=["Connection refused"])]
    sp = Spawner({"00-a": bad, "01-b": bad})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, once=True) == 1
    assert sp.count("00-a") == 1 and sp.count("01-b") == 1


def test_once_mode_fatal_startup_does_not_wait(tmp_path, caplog):
    cfg = stream_cfg(tmp_path)
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        rc = run(cfg, Spawner({}), once=True, tuning=dataclasses.replace(FAST, fatal_wait_s=5),
                 load_model=lambda: 1 / 0)
    assert rc == 1 and time.monotonic() - t0 < 1.0
    assert _msgs(caplog, logging.ERROR) == ["birdnet model failed: division by zero"]


# ----------------------------------------------------------------- 16: fatal startup and the rc 1 path
def test_model_failure_is_fatal_before_any_thread(tmp_path, caplog, tracked):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({})

    def boom():
        raise RuntimeError(f"no model at {SECRET_URL}")

    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, load_model=boom) == 1
    assert time.monotonic() - t0 >= FAST.fatal_wait_s
    assert tracked == [] and _spawned(sp) == 0
    assert _msgs(caplog, logging.ERROR) == ["birdnet model failed: no model at rtsp://admin:***@10.0.0.2/x"]


def test_model_loaded_once_in_main_thread_before_first_spawn(tmp_path, spies):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})
    loads = []

    def load_model():
        loads.append((time.monotonic(), threading.current_thread() is threading.main_thread(), _spawned(sp)))

    assert run(cfg, sp, once=True, load_model=load_model) == 0
    assert len(loads) == 1
    at, in_main, spawned_before = loads[0]
    assert in_main and spawned_before == 0 and at < sp.spawns["00-front"][0][0]


def test_database_open_failure_is_fatal(tmp_path, caplog, monkeypatch):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({})

    def open_db(d):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(bl, "open_db", open_db)
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp) == 1
    assert _msgs(caplog, logging.ERROR) == ["database open failed: unable to open database file"]
    assert _spawned(sp) == 0


def test_signal_cuts_the_fatal_wait_short(tmp_path, caplog):
    cfg = stream_cfg(tmp_path)
    flag = SimpleNamespace(value=False)
    timer = threading.Timer(0.1, lambda: setattr(flag, "value", True))
    timer.start()
    t0 = time.monotonic()
    rc = run(cfg, Spawner({}), signalled=lambda: flag.value, tuning=dataclasses.replace(FAST, fatal_wait_s=5),
             load_model=lambda: 1 / 0)
    timer.join()
    assert rc == 1 and time.monotonic() - t0 < 1.0


def test_worker_death_is_rc1_after_fatal_wait(tmp_path, caplog, spies, tracked, install_hook):
    hook = install_hook()
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})

    def analyze(w, c, t):
        raise Die()

    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, analyze_fn=analyze) == 1
    took = time.monotonic() - t0
    worker = next(t for t in tracked if t.name == "birdlisten-worker")
    assert not worker.is_alive() and not [t.name for t in tracked if t.is_alive()]
    assert took >= FAST.fatal_wait_s
    errors = _msgs(caplog, logging.ERROR)
    assert "analysis worker died" in errors
    assert any(m.startswith("thread birdlisten-worker crashed:\nTraceback") for m in errors)
    assert all(_reaped(f) for f in sp.all_fakes())
    assert threading.excepthook is hook


def test_supervisor_death_joins_every_other_supervisor(tmp_path, caplog, spies, tracked, install_hook):
    hook = install_hook()
    cfg = stream_cfg(tmp_path, ("a", "b"))
    sp = Spawner({"00-a": [dict(n_segments=0, exit_code=1, exit_after_s=0.3), Die()],
                  "01-b": [dict(n_segments=None, gap_s=0.03)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp) == 1
    assert not [t.name for t in tracked if t.is_alive()]
    assert sorted(t.name for t in tracked if t.name.startswith("cam-") and not t.name.endswith("-stderr")) == \
        ["cam-a", "cam-b"]
    assert "capture thread a died" in _msgs(caplog, logging.ERROR)
    assert sp.fakes["01-b"] and all(_reaped(f) for f in sp.all_fakes())
    assert threading.excepthook is hook


def test_signal_during_unbounded_join_returns_within_budget(tmp_path, caplog, spies, tracked, install_hook):
    install_hook()
    cfg = stream_cfg(tmp_path, ("a", "b"))
    sp = Spawner({"00-a": [dict(n_segments=None, gap_s=0.03)],
                  "01-b": [dict(n_segments=0, exit_code=1, exit_after_s=0.3), Die()]})
    release = threading.Event()
    busy = threading.Event()

    def analyze(w, c, t):
        busy.set()
        release.wait(10)
        return []

    tuning = dataclasses.replace(FAST, fatal_wait_s=5)
    try:
        with caplog.at_level(logging.INFO, logger="birdlisten"):
            r = BgRun(cfg, sp, analyze_fn=analyze, tuning=tuning).start()
            wait_for(lambda: busy.is_set() and "capture thread b died" in _msgs(caplog, logging.ERROR))
            time.sleep(0.1)                           # main is in the unbounded worker join
            assert r.thread.is_alive()
            r.signal()
            assert r.join() == 1
        assert r.returned - r.sig_at <= tuning.shutdown_budget_s + 0.5
        worker = next(t for t in tracked if t.name == "birdlisten-worker")
        assert worker.is_alive()
        assert threading.excepthook is stream._scrubbed_excepthook      # a started thread still lives
    finally:
        release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    loads = []
    assert run(cfg, Spawner({}), once=True, load_model=lambda: loads.append(1) or 1 / 0) == 1
    assert loads == [1]                               # the guard let a new pipeline start


def test_database_open_that_hangs_times_out(tmp_path, caplog, monkeypatch, tracked):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({})
    gate = threading.Event()
    real_open = bl.open_db
    monkeypatch.setattr(bl, "open_db", lambda d: gate.wait(5) and real_open(d))
    gets = []
    real_get = stream.ClipQueue.get
    monkeypatch.setattr(stream.ClipQueue, "get", lambda self, timeout: gets.append(1) or real_get(self, timeout))
    analyzed = []
    tuning = dataclasses.replace(FAST, worker_ready_s=0.2)
    t0 = time.monotonic()
    try:
        with caplog.at_level(logging.INFO, logger="birdlisten"):
            rc = run(cfg, sp, tuning=tuning, analyze_fn=lambda *a: analyzed.append(a) or [])
        assert rc == 1
        assert time.monotonic() - t0 <= 2 * tuning.worker_ready_s + tuning.fatal_wait_s + 0.5
        assert "database open failed: timed out" in _msgs(caplog, logging.ERROR)
        assert _spawned(sp) == 0
        worker, = tracked
        assert worker.is_alive()                      # stuck in open_db, still tracked by the guard
    finally:
        gate.set()
    worker.join(timeout=2)
    assert not worker.is_alive() and gets == [] and analyzed == []


# ----------------------------------------------------------------- 16b: main loop crash
def _crashing_summary(monkeypatch, always: bool, on_crash=None):
    real = stream.Stats.snapshot_and_reset
    calls = [0]

    def snapshot_and_reset(self, *a, **k):
        calls[0] += 1
        if always or calls[0] == 1:
            if on_crash is not None and calls[0] == 1:
                on_crash()
            raise RuntimeError(f"boom {SECRET_URL}")
        return real(self, *a, **k)

    monkeypatch.setattr(stream.Stats, "snapshot_and_reset", snapshot_and_reset)
    return calls


def test_main_loop_crash_cleans_up_everything(tmp_path, caplog, monkeypatch, spies, tracked, registries,
                                              install_hook):
    hook = install_hook()
    cfg = stream_cfg(tmp_path, ("a", "b"))
    plan = [dict(n_segments=None, gap_s=0.03)]
    sp = Spawner({"00-a": plan, "01-b": plan})
    _crashing_summary(monkeypatch, always=False)
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        rc = run(cfg, sp, summary_s=0.3, analyze_fn=lambda w, c, t: time.sleep(0.05) or [])
    assert rc == 1 and time.monotonic() - t0 >= 0.3 + FAST.fatal_wait_s
    assert sorted(t.name for t in tracked) == ["birdlisten-worker", "cam-a", "cam-a-stderr", "cam-b", "cam-b-stderr"]
    assert not [t.name for t in tracked if t.is_alive()]
    assert sp.all_fakes() and all(_reaped(f) for f in sp.all_fakes())
    assert registries[0].snapshot() == []
    assert not list((cfg.segment_dir / "queued").iterdir())
    crashed = [m for m in _msgs(caplog, logging.ERROR) if m.startswith("stream main loop crashed:")]
    assert len(crashed) == 1 and "RuntimeError: boom rtsp://admin:***@10.0.0.2/x" in crashed[0]
    _no_secret(caplog)
    assert _summaries(caplog)[-1].get("final") == "1"         # the crash path's summary worked
    assert threading.excepthook is hook
    loads = []
    assert run(cfg, Spawner({}), once=True, load_model=lambda: loads.append(1) or 1 / 0) == 1
    assert loads == [1]


def test_main_loop_crash_with_summary_always_failing(tmp_path, caplog, monkeypatch, spies, tracked, registries):
    cfg = stream_cfg(tmp_path, ("a", "b"))
    plan = [dict(n_segments=None, gap_s=0.03)]
    sp = Spawner({"00-a": plan, "01-b": plan})
    calls = _crashing_summary(monkeypatch, always=True)
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, summary_s=0.2) == 1
    assert calls[0] == 2
    assert not [t.name for t in tracked if t.is_alive()]
    assert all(_reaped(f) for f in sp.all_fakes()) and registries[0].snapshot() == []
    errors = _msgs(caplog, logging.ERROR)
    assert any(m.startswith("stream main loop crashed:") for m in errors)
    assert any(m.startswith("stream shutdown step summary failed:\nTraceback") for m in errors)
    _no_secret(caplog)


def test_main_loop_crash_then_signal_skips_fatal_wait(tmp_path, caplog, monkeypatch, spies, tracked):
    cfg = stream_cfg(tmp_path, ("a", "b"))
    plan = [dict(n_segments=None, gap_s=0.03)]
    sp = Spawner({"00-a": plan, "01-b": plan})
    flag = SimpleNamespace(value=False, at=None)

    def later():
        flag.at = time.monotonic()
        flag.value = True

    timers = []
    _crashing_summary(monkeypatch, always=False,
                      on_crash=lambda: timers.append(threading.Timer(0.05, later)) or timers[-1].start())
    tuning = dataclasses.replace(FAST, fatal_wait_s=5)
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, sp, summary_s=0.2, tuning=tuning, signalled=lambda: flag.value) == 1
    done = time.monotonic()
    timers[0].join()
    assert flag.at is not None and done - flag.at <= tuning.shutdown_budget_s + 0.5
    assert not [t.name for t in tracked if t.is_alive()]


# ----------------------------------------------------------------- 16c: run guard
def test_run_guard_refuses_a_second_pipeline(tmp_path, caplog, spies, install_hook):
    hook = install_hook()
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        first = BgRun(cfg, sp).start()
        try:
            wait_for(lambda: sp.count("00-front") >= 1)
            ghost = make_wav(cfg.segment_dir / "queued" / "07-ghost_r0001_20260101T000000Z.wav", 0.1)
            assert threading.excepthook is stream._scrubbed_excepthook
            loads, spawns = [], []
            t0 = time.monotonic()
            rc = run(cfg, lambda cmd, env: spawns.append(cmd), once=True, load_model=lambda: loads.append(1))
            assert rc == 1 and time.monotonic() - t0 < 0.5
            assert loads == [] and spawns == []
            assert threading.excepthook is stream._scrubbed_excepthook   # the first run's hook, untouched
            assert ghost.exists()                     # no startup cleanup ran
            refused = [m for m in _msgs(caplog, logging.ERROR) if m.startswith("stream pipeline already running (")]
            assert len(refused) == 1 and refused[0].endswith(" threads alive), not starting another")
        finally:
            first.signal()
            assert first.join() == 0
    assert threading.excepthook is hook
    loads = []

    def failing():
        loads.append(1)
        raise RuntimeError("no model")

    assert run(cfg, sp, once=True, load_model=failing) == 1      # a third run claims the guard
    assert run(cfg, sp, once=True, load_model=failing) == 1      # and a failed load released it
    assert loads == [1, 1]


# ----------------------------------------------------------------- 18: main_stream
@pytest.fixture
def no_loop_stop(monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace())
    monkeypatch.delitem(sys.modules, "loop", raising=False)


def test_main_stream_installs_and_restores_signal_handlers(tmp_path, caplog, no_loop_stop):
    cfg = stream_cfg(tmp_path)
    sp = Spawner({})
    prev_calls = []

    def prev_term(s, f):
        prev_calls.append(s)

    def prev_int(s, f):
        prev_calls.append(s)

    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    signal.signal(signal.SIGTERM, prev_term)
    signal.signal(signal.SIGINT, prev_int)
    try:
        seen = {}

        def load_model():
            seen["term"], seen["int"] = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
            seen["term"](signal.SIGTERM, None)        # a SIGTERM during the model load

        with caplog.at_level(logging.INFO, logger="birdlisten"):
            rc = stream.main_stream(cfg, spawn=sp, load_model=load_model, tuning=FAST)
        assert rc == 0 and _spawned(sp) == 0          # startup step 4 saw the flag
        assert seen["term"] is seen["int"] and seen["term"] not in (prev_term, prev_int)
        assert prev_calls == [signal.SIGTERM]         # chained to the previous handler
        assert signal.getsignal(signal.SIGTERM) is prev_term
        assert signal.getsignal(signal.SIGINT) is prev_int
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_main_stream_early_signal_starts_nothing(tmp_path, monkeypatch, no_loop_stop):
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(_stop=True))   # loop.py saw SIGTERM first
    cfg = stream_cfg(tmp_path)
    sp = Spawner({})
    loads = []
    assert stream.main_stream(cfg, spawn=sp, load_model=lambda: loads.append(1), tuning=FAST) == 0
    assert loads == [] and _spawned(sp) == 0


def test_excepthook_restored_after_clean_run(tmp_path, spies, install_hook):
    hook = install_hook()
    cfg = stream_cfg(tmp_path)
    sp = Spawner({"00-front": [dict(n_segments=None, gap_s=0.05)]})
    assert run(cfg, sp, once=True) == 0
    assert threading.excepthook is hook


# ----------------------------------------------------------------- 19: selftest
class SegFake(FakeFfmpeg):
    """FakeFfmpeg writing one segment per entry of `durations` (seconds), then exiting rc 0."""

    def __init__(self, cmd, env, durations, **kw):
        self.durations = list(durations)
        super().__init__(cmd, env, n_segments=len(self.durations), **kw)

    def _segment(self, i):
        self.seg_s = self.durations[i]
        return super()._segment(i)


def _logger_state():
    lg = logging.getLogger("birdlisten")
    return lg.propagate, list(lg.handlers), lg.level


def test_selftest_passes_with_a_fake_ffmpeg_and_restores_the_logger():
    before = _logger_state()
    fakes = []

    def spawn(cmd, env):
        assert cmd[6:11] == stream.SELFTEST_INPUT and cmd[cmd.index("-segment_time") + 1] == "3"
        fakes.append(SegFake(cmd, env, [3.0, 3.0, 1.0]))
        return fakes[-1]

    records = []
    stream.selftest(spawn=spawn, records=records)
    assert len(fakes) == 1 and _reaped(fakes[0])
    assert _logger_state() == before
    loud = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert len(loud) == 1 and loud[0].startswith("selftest: ffmpeg exited after ") and "exit 0; retry in" in loud[0]


def test_selftest_one_segment_fails_with_clip_count():
    before = _logger_state()
    with pytest.raises(stream.SelftestError, match=r"^expected 3 clips, got 1$"):
        stream.selftest(spawn=lambda cmd, env: SegFake(cmd, env, [3.0]), deadline_s=1.0)
    assert _logger_state() == before


def test_selftest_main_prints_failure_and_records(monkeypatch, capsys):
    def boom(records=None, **kw):
        records.append(logging.makeLogRecord({"levelno": logging.ERROR, "levelname": "ERROR",
                                              "threadName": "cam-selftest", "msg": f"x {SECRET_URL}"}))
        raise stream.SelftestError("expected 3 clips, got 0")

    monkeypatch.setattr(stream, "selftest", boom)
    assert stream._selftest_main() == 1
    out = capsys.readouterr().out
    assert out.startswith("selftest FAILED: expected 3 clips, got 0\n")
    assert "  ERROR cam-selftest: x rtsp://admin:***@10.0.0.2/x" in out and "s3cret" not in out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs a real ffmpeg (runs in CI as stream.py --selftest)")
def test_selftest_real_ffmpeg():
    stream.selftest()


# ----------------------------------------------------------------- plan decision 7: what main() must keep doing
@pytest.fixture
def main_env(monkeypatch, tmp_path, no_loop_stop):
    for k in ("CAPTURE_MODE", "QUEUE_SIZE", "SUMMARY_MINUTES", "CLIP_SECONDS", "LOOP_ONCE", "KEEP_CLIPS",
              "NTFY_TOPIC", "MIN_CONFIDENCE"):
        monkeypatch.delenv(k, raising=False)
    base = {"CAMERAS": "front=rtsp://admin:s3cret@10.0.0.2/h264Preview_01_sub", "LATITUDE": "0",
            "LONGITUDE": "0", "DATA_DIR": str(tmp_path / "data"), "SEGMENT_DIR": str(tmp_path / "seg")}

    def apply(*argv, **env):
        for k, v in {**base, **env}.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(sys, "argv", ["birdlisten.py", *argv])
        return bl.load_config()
    return apply


def test_main_report_7(main_env, capsys):
    cfg = main_env("--report", "7")
    conn = bl.open_db(cfg.data_dir)
    bl.record(conn, dt.datetime.now(UTC), cfg.cameras[0], BUSHTIT, None)
    conn.close()
    assert bl.main() == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["species", "heard", "best", "last"]
    assert "Bushtit" in out


def test_main_dry_run_writes_nothing(main_env, monkeypatch, spies):
    cfg = main_env("--dry-run")
    monkeypatch.setattr(bl, "capture", lambda cam, seconds, out: out.write_bytes(b"RIFF"))
    monkeypatch.setattr(bl, "analyze", lambda wav, c, when: [BUSHTIT])
    assert bl.main() == 0
    assert spies.record == [] and spies.notify == []
    assert _rows(cfg) == []


def test_main_check_with_mocks(main_env, monkeypatch, capsys):
    main_env("--check")
    monkeypatch.setattr(bl.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(bl, "analyzer", lambda: None)
    monkeypatch.setattr(bl, "capture", lambda cam, seconds, out: out.write_bytes(b"RIFF"))
    assert bl.main() == 0
    out = capsys.readouterr().out
    assert "camera front: ok (rtsp://admin:***@10.0.0.2/h264Preview_01_sub)" in out
    assert "capture: stream, queue 2, summary every 5 min, segments " in out


def test_main_loop_once_stream_end_to_end(main_env, monkeypatch, caplog, spies):
    cfg = main_env(LOOP_ONCE="1")
    sp = Spawner({"00-front": [dict(n_segments=1, seg_s=3.0)]})
    monkeypatch.setattr(stream, "popen_spawn", sp)
    monkeypatch.setattr(bl, "analyzer", lambda: None)
    monkeypatch.setattr(bl, "analyze", lambda wav, c, when: [BUSHTIT])
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert bl.main() == 0
    assert time.monotonic() - t0 < 5
    assert sp.count("00-front") == 1 and all(_reaped(f) for f in sp.all_fakes())
    rows = _rows(cfg)
    assert [(r[1], r[2]) for r in rows] == [("front", "Bushtit")]
    assert rows[0][0] == "2026-10-05T23:15:00+00:00" and rows[0][3] == 3.0   # segment start, detection offset
    assert [n for n, *_ in spies.notify] == ["birdlisten-worker"]
    assert _summaries(caplog)[-1]["final"] == "1"
    assert "stream pipeline stopped" in _msgs(caplog, logging.INFO)
    assert not list((tmp := Path(cfg.segment_dir)).rglob("*.wav")), sorted(tmp.rglob("*"))


def test_loop_main_starts_the_server_before_the_app(monkeypatch):
    import loop
    calls = []
    monkeypatch.setattr(loop, "app_start_server", lambda: calls.append("server"))
    monkeypatch.setattr(loop, "app_main", lambda: calls.append("app") or 0)
    monkeypatch.setattr(loop, "_stop", False)
    monkeypatch.setenv("LOOP_ONCE", "1")
    monkeypatch.setenv("RUN_ARGS", "")
    monkeypatch.setattr(sys, "argv", ["loop.py"])
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert loop.main() == 0
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    assert calls == ["server", "app"]
