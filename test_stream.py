"""Tests for stream.py (CAPTURE_MODE=stream). Network-free, no real ffmpeg.
Run: uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q test_stream.py"""

from __future__ import annotations

import datetime as dt
import logging
import re
import signal
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
