"""Tests for AUDIO_SOURCE=inbox (birdlisten reads the shared listener's inbox, issue #40).
Network-free, no real ffmpeg, no BirdNET: the analyzer is a fake.
Run: uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q test_inbox.py"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

import birdlisten as bl
import listener as ls
import stream
from test_stream import (BASE, BUSHTIT, FAST, SECRET_URL, BgRun, Spawner, _run_guard,  # noqa: F401 (_run_guard is autouse)
                         make_wav, run, wait_for)

UTC = dt.timezone.utc
SEG_S = 0.2


def inbox_cfg(tmp_path, **env) -> bl.Config:
    (tmp_path / "inbox").mkdir(exist_ok=True)
    return bl.load_config({"AUDIO_SOURCE": "inbox", "INBOX_DIR": str(tmp_path / "inbox"), "LATITUDE": "0",
                           "LONGITUDE": "0", "DATA_DIR": str(tmp_path / "data"),
                           "SEGMENT_DIR": str(tmp_path / "seg"), **env})


def put(cfg: bl.Config, camdir: str, start: dt.datetime, seconds: float = SEG_S) -> Path:
    return make_wav(Path(cfg.inbox_dir) / f"{camdir}_r0001_{start:%Y%m%dT%H%M%S}Z.wav", seconds)


def rows(cfg) -> list[tuple]:
    conn = sqlite3.connect(Path(cfg.data_dir) / "birdlisten.sqlite")
    try:
        return conn.execute("SELECT heard_at, camera, common_name FROM detections ORDER BY heard_at, camera").fetchall()
    finally:
        conn.close()


class NoSpawn:
    """Inbox mode must never start an ffmpeg."""
    calls = 0

    def __call__(self, cmd, env):
        NoSpawn.calls += 1
        raise AssertionError("inbox mode spawned ffmpeg")


# ----------------------------------------------------------------- config
def test_config_inbox_mode_defaults():
    cfg = bl.load_config({"AUDIO_SOURCE": "inbox", "LATITUDE": "0", "LONGITUDE": "0"})
    assert (cfg.audio_source, cfg.inbox_dir, cfg.cameras, cfg.queue_size) == \
        ("inbox", Path("/audio/inbox/birdlisten"), (), 8)


def test_config_rtsp_is_default_and_still_needs_cameras():
    base = {"LATITUDE": "0", "LONGITUDE": "0"}
    assert bl.load_config({**base, "CAMERAS": "a=rtsp://h/x"}).audio_source == "rtsp"
    with pytest.raises(bl.ConfigError, match="CAMERAS"):
        bl.load_config(base)


def test_config_inbox_keeps_optional_cameras():
    cfg = bl.load_config({"AUDIO_SOURCE": "INBOX", "CAMERAS": "garage=rtsp://h/x", "LATITUDE": "0", "LONGITUDE": "0"})
    assert [c.name for c in cfg.cameras] == ["garage"] and cfg.audio_source == "inbox"


@pytest.mark.parametrize("extra, msg", [
    ({"AUDIO_SOURCE": "nvr"}, "rtsp or inbox"),
    ({"AUDIO_SOURCE": "inbox", "CAPTURE_MODE": "roundrobin"}, "CAPTURE_MODE=stream"),
    ({"AUDIO_SOURCE": "inbox", "INBOX_DIR": "audio/inbox"}, "INBOX_DIR must be an absolute path"),
])
def test_config_inbox_errors(extra, msg):
    with pytest.raises(bl.ConfigError, match=msg):
        bl.load_config({"LATITUDE": "0", "LONGITUDE": "0", **extra})


def test_dry_run_refused_in_inbox_mode(monkeypatch, capsys, tmp_path):
    for k, v in {"AUDIO_SOURCE": "inbox", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CAMERAS", raising=False)
    monkeypatch.setattr(sys, "argv", ["birdlisten.py", "--dry-run"])
    assert bl.main() == 2
    assert "needs AUDIO_SOURCE=rtsp" in capsys.readouterr().err


def test_check_inbox_dir(tmp_path, capsys):
    cfg = inbox_cfg(tmp_path)
    put(cfg, "00-garage", BASE)
    assert bl._check_inbox_dir(cfg) == 0
    assert f"capture: inbox {cfg.inbox_dir}, queue 8, 1 files waiting" in capsys.readouterr().out
    missing = inbox_cfg(tmp_path, INBOX_DIR=str(tmp_path / "nope"))
    assert bl._check_inbox_dir(missing) == 1
    assert "NOT FOUND" in capsys.readouterr().err


# ----------------------------------------------------------------- pipeline
def test_once_analyzes_oldest_first_names_cameras_and_deletes(tmp_path, caplog):
    cfg = inbox_cfg(tmp_path)
    t1, t2, t3 = BASE, BASE + dt.timedelta(seconds=30), BASE + dt.timedelta(seconds=60)
    files = [put(cfg, "01-garage", t3), put(cfg, "00-doorbell", t1), put(cfg, "01-garage", t2)]
    (Path(cfg.inbox_dir) / "notes.txt").write_text("not a segment")
    seen = []

    def analyze(wav, c, when):
        seen.append((Path(wav).name, when))
        return [BUSHTIT]

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, NoSpawn(), analyze_fn=analyze, once=True) == 0
    assert [w for _, w in seen] == [t1, t2, t3]                       # oldest first across cameras
    assert rows(cfg) == [(t1.isoformat(timespec="seconds"), "doorbell", "Bushtit"),
                         (t2.isoformat(timespec="seconds"), "garage", "Bushtit"),
                         (t3.isoformat(timespec="seconds"), "garage", "Bushtit")]
    assert not any(f.exists() for f in files)
    assert sorted(os.listdir(cfg.inbox_dir)) == ["notes.txt"]          # foreign files untouched
    assert NoSpawn.calls == 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith(f"stream mode, inbox {cfg.inbox_dir}") for m in msgs)
    assert any(m.startswith("timing camera=doorbell clip_s=0.2") for m in msgs)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_deletes_only_this_consumers_link(tmp_path):
    cfg = inbox_cfg(tmp_path)
    other = tmp_path / "transcriber"; other.mkdir()
    f = put(cfg, "01-garage", BASE)
    os.link(f, other / f.name)
    data = f.read_bytes()
    assert run(cfg, NoSpawn(), once=True) == 0
    assert not f.exists()
    assert (other / f.name).read_bytes() == data and os.stat(other / f.name).st_nlink == 1


def test_full_queue_leaves_files_waiting_instead_of_dropping(tmp_path, caplog):
    cfg = inbox_cfg(tmp_path, QUEUE_SIZE="2")
    files = [put(cfg, "01-garage", BASE + dt.timedelta(seconds=30 * i)) for i in range(6)]
    gate, started = threading.Event(), []

    def slow(wav, c, when):
        started.append(when)
        gate.wait(5)
        return []

    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, NoSpawn(), analyze_fn=slow).start()
        wait_for(lambda: len(started) == 1)
        time.sleep(0.2)                                # several reader polls while the worker is stuck
        # one in the worker, two queued, three still waiting in the inbox untouched
        assert sum(f.exists() for f in files) == 6
        gate.set()
        wait_for(lambda: not any(f.exists() for f in files))
        r.signal()
        assert r.join() == 0
    assert len(started) == 6 and started == sorted(started)
    summ = [m for m in (x.getMessage() for x in caplog.records) if m.startswith("timing summary ")]
    assert summ and "dropped=0" in summ[-1] and "clips=6" in summ[-1] and "cameras=0" in summ[-1]
    assert not [m for m in (x.getMessage() for x in caplog.records) if "dropped oldest" in m]


def test_bad_and_short_segments_deleted_and_counted(tmp_path, caplog):
    cfg = inbox_cfg(tmp_path)
    bad = Path(cfg.inbox_dir) / f"00-garage_r0001_{BASE:%Y%m%dT%H%M%S}Z.wav"
    bad.write_bytes(b"RIFF junk")
    short = put(cfg, "00-garage", BASE + dt.timedelta(seconds=30), seconds=0.05)   # FAST.min_segment_s = 0.1
    good = put(cfg, "00-garage", BASE + dt.timedelta(seconds=60))
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        assert run(cfg, NoSpawn(), once=True) == 0
    assert not bad.exists() and not short.exists() and not good.exists()
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("inbox: bad segment 00-garage_r0001_") for m in msgs)
    assert any(m.startswith("inbox: short segment") and "0.1s" in m for m in msgs)
    final = next(m for m in msgs if m.startswith("timing summary ") and "final=1" in m)
    assert "failed=1" in final and "short=1" in final and "analyzed=1" in final


def test_missing_inbox_is_fatal(tmp_path, caplog):
    cfg = inbox_cfg(tmp_path, INBOX_DIR=str(tmp_path / "not-mounted"))
    assert run(cfg, NoSpawn(), once=True) == 1
    assert any("not found (is the listener's audio volume mounted?)" in r.getMessage() for r in caplog.records)


def test_reader_survives_inbox_disappearing_and_returning(tmp_path, caplog):
    cfg = inbox_cfg(tmp_path)
    inbox = Path(cfg.inbox_dir)
    analyzed = []
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        r = BgRun(cfg, NoSpawn(), analyze_fn=lambda w, c, t: analyzed.append(t) or []).start()
        wait_for(lambda: any("stream mode, inbox" in x.getMessage() for x in caplog.records))
        inbox.rename(tmp_path / "away")
        wait_for(lambda: any("unreadable" in x.getMessage() for x in caplog.records))
        (tmp_path / "away").rename(inbox)
        put(cfg, "00-garage", BASE)
        wait_for(lambda: analyzed == [BASE])
        r.signal()
        assert r.join() == 0
    msgs = [x.getMessage() for x in caplog.records]
    assert sum("unreadable" in m for m in msgs) == 1                 # logged once, not every poll
    assert any("readable again" in m for m in msgs)


# ----------------------------------------------------------------- with the real listener
def test_listener_feeds_birdlisten_and_leaves_transcriber_copy(tmp_path, caplog):
    audio = tmp_path / "audio"
    lcfg = ls.ListenerConfig(cameras=(bl.Camera("doorbell", SECRET_URL), bl.Camera("garage", SECRET_URL)),
                             consumers=("birdlisten", "transcriber"), audio_dir=audio)
    ls.prepare(lcfg)
    bcfg = inbox_cfg(tmp_path, INBOX_DIR=str(lcfg.inbox("birdlisten")))
    heard = []

    def analyze(wav, c, when):
        heard.append(Path(wav).name)
        return [BUSHTIT]

    t0 = time.monotonic()
    now = lambda: BASE + dt.timedelta(seconds=30 * (time.monotonic() - t0))     # noqa: E731
    sp = Spawner({"00-doorbell": [dict(n_segments=3, gap_s=0.02, exit_after_s=5.0)],
                  "01-garage": [dict(n_segments=3, gap_s=0.02, exit_after_s=5.0)]})
    stop = threading.Event()
    with caplog.at_level(logging.INFO, logger="birdlisten"):
        # birdlisten first: run_stream refuses to start while other stream threads are alive
        b = BgRun(bcfg, NoSpawn(), analyze_fn=analyze).start()
        wait_for(lambda: any("stream mode, inbox" in x.getMessage() for x in caplog.records))
        out = []
        lt = threading.Thread(target=lambda: out.append(ls.run_listener(
            lcfg, stop=stop, spawn=sp, tuning=FAST, now_utc=now, status_every_s=0.05)), daemon=True)
        lt.start()
        try:
            wait_for(lambda: len(heard) >= 4, timeout=5)              # 2 complete per camera while ffmpeg runs
            wait_for(lambda: not os.listdir(lcfg.inbox("birdlisten")), timeout=2)
            left = sorted(os.listdir(lcfg.inbox("transcriber")))
            assert sorted(heard) == left                              # the transcriber still has every one
            assert all(os.stat(lcfg.inbox("transcriber") / n).st_nlink == 1 for n in left)
        finally:
            stop.set()
            lt.join(timeout=5)
            for f in sp.all_fakes():
                if f.alive():
                    f.kill()
                f._writer.join(timeout=1)
            b.signal()
            assert b.join() == 0
    assert out == [0]
    cams = {c for _, c, _ in rows(bcfg)}
    assert cams == {"doorbell", "garage"}
    assert not [x.getMessage() for x in caplog.records if "s3cret" in x.getMessage()]
