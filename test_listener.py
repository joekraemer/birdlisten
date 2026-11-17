"""Tests for listener.py (issue #40). Network-free, no real ffmpeg: the
integration tests drive the real CameraSupervisors with test_stream's fake.
Run: uv run --no-project --python 3.11 --with pillow==12.3.0 --with pytest==8.3.4 pytest -q test_listener.py"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from pathlib import Path

import pytest

import birdlisten as bl
import listener as ls
import stream
from test_stream import BASE, FAST, SECRET_URL, Spawner, _run_guard, make_wav, wait_for  # noqa: F401 (_run_guard is autouse)

UTC = dt.timezone.utc
ENV = {"CAMERAS": "doorbell=rtsp://a:b@h/Preview_01_main,garage=rtsp://a:b@h/Preview_04_sub",
       "CONSUMERS": "birdlisten,transcriber"}


# ----------------------------------------------------------------- config
def test_load_config_defaults():
    cfg = ls.load_config(ENV)
    assert [c.name for c in cfg.cameras] == ["doorbell", "garage"]
    assert cfg.consumers == ("birdlisten", "transcriber")
    assert cfg.audio_dir == Path("/audio") and cfg.segment_dir == Path("/audio/segments")
    assert cfg.inbox("transcriber") == Path("/audio/inbox/transcriber")
    assert (cfg.clip_seconds, cfg.inbox_max_age_s, cfg.inbox_max_bytes, cfg.camera_stale_s) == \
        (30, 3600, 512 * 1024 * 1024, 300)


def test_load_config_overrides():
    cfg = ls.load_config({**ENV, "AUDIO_DIR": "/x", "CLIP_SECONDS": "10", "INBOX_MAX_AGE_S": "600",
                          "INBOX_MAX_MB": "64", "CAMERA_STALE_S": "120"})
    assert (cfg.audio_dir, cfg.clip_seconds, cfg.inbox_max_age_s, cfg.inbox_max_bytes, cfg.camera_stale_s) == \
        (Path("/x"), 10, 600, 64 * 1024 * 1024, 120)


@pytest.mark.parametrize("extra, msg", [
    ({"CAMERAS": ""}, "CAMERAS is required"),
    ({"CONSUMERS": ""}, "CONSUMERS is required"),
    ({"CONSUMERS": " , "}, "CONSUMERS is empty"),
    ({"CONSUMERS": "Bird"}, "must match"),
    ({"CONSUMERS": "a/b"}, "must match"),
    ({"CONSUMERS": "a,a"}, "unique"),
    ({"AUDIO_DIR": "audio"}, "absolute"),
    ({"AUDIO_DIR": "/a%d"}, "'%'"),
    ({"CLIP_SECONDS": "2"}, "3..600"),
    ({"INBOX_MAX_MB": "lots"}, "whole number"),
    ({"CAMERA_STALE_S": "5"}, "30..86400"),
])
def test_load_config_errors(extra, msg):
    with pytest.raises(bl.ConfigError, match=msg):
        ls.load_config({**ENV, **extra})


# ----------------------------------------------------------------- names
def test_inbox_name_round_trip():
    name = ls.inbox_name("01-west-rooftop", "r0001_20261117T000238Z.wav")
    assert name == "01-west-rooftop_r0001_20261117T000238Z.wav"
    assert ls.parse_inbox_name(name) == ("01-west-rooftop", dt.datetime(2026, 11, 17, 0, 2, 38, tzinfo=UTC))
    assert ls.camera_of("01-west-rooftop") == "west-rooftop"
    # stream.py's queued names are inbox names
    assert stream.QUEUED_RE.match(name)


def test_parse_inbox_name_underscores_and_junk():
    assert ls.parse_inbox_name("00-front_door__r0002_20261005T231500Z.wav")[0] == "00-front_door_"
    for junk in ("status.json", ".tmp-x", "00-a_r0001_20261005T231500Z.wav.part", "a_r0001_20261005T231500Z.wav"):
        assert ls.parse_inbox_name(junk) is None


# ----------------------------------------------------------------- fan-out
def _clip(path: Path, camera="garage", ready=BASE) -> stream.Clip:
    return stream.Clip(camera=camera, path=path, start_utc=ready, duration_s=0.2, ready_utc=ready, ready_mono=0.0)


def _fanout(tmp_path, consumers=("a", "b")):
    queued = tmp_path / "segments" / "queued"
    queued.mkdir(parents=True)
    inboxes = {}
    for c in consumers:
        inboxes[c] = tmp_path / "inbox" / c
        inboxes[c].mkdir(parents=True)
    return queued, inboxes, ls.Fanout(inboxes, {"garage": "01-garage"})


def test_fanout_links_into_every_inbox_and_frees_on_last_delete(tmp_path):
    queued, inboxes, fan = _fanout(tmp_path)
    src = make_wav(queued / "01-garage_r0001_20261005T231500Z.wav", 0.2)
    data = src.read_bytes()
    fan.put(_clip(src))
    assert not src.exists()
    a, b = inboxes["a"] / src.name, inboxes["b"] / src.name
    assert os.stat(a).st_nlink == 2 and os.stat(a).st_ino == os.stat(b).st_ino
    a.unlink()                                       # consumer a is done
    assert os.stat(b).st_nlink == 1 and b.read_bytes() == data
    b.unlink()                                       # last one: the audio is gone
    assert not any(p.is_file() for p in tmp_path.rglob("*"))
    st = fan.snapshot()["garage"]
    assert (st.dir, st.segments, st.last_segment_utc) == ("01-garage", 1, BASE)


def test_fanout_keeps_first_on_name_clash(tmp_path, caplog):
    queued, inboxes, fan = _fanout(tmp_path)
    name = "01-garage_r0001_20261005T231500Z.wav"
    (inboxes["a"] / name).write_bytes(b"first")
    fan.put(_clip(make_wav(queued / name, 0.2)))
    assert (inboxes["a"] / name).read_bytes() == b"first"
    assert os.stat(inboxes["b"] / name).st_nlink == 1
    assert any("already in inbox a" in r.getMessage() for r in caplog.records)


def test_fanout_missing_inbox_does_not_block_others(tmp_path, caplog):
    queued, inboxes, fan = _fanout(tmp_path)
    inboxes["a"].rmdir()
    src = make_wav(queued / "01-garage_r0001_20261005T231500Z.wav", 0.2)
    with caplog.at_level(logging.WARNING):
        fan.put(_clip(src))
    assert (inboxes["b"] / src.name).is_file() and not src.exists()
    assert fan.link_failures == 1
    assert any("could not link" in r.getMessage() and "inbox a" in r.getMessage() for r in caplog.records)


def test_fanout_reaching_no_inbox_still_deletes_queued(tmp_path, caplog):
    queued, inboxes, fan = _fanout(tmp_path, consumers=("a",))
    inboxes["a"].rmdir()
    src = make_wav(queued / "01-garage_r0001_20261005T231500Z.wav", 0.2)
    fan.put(_clip(src))
    assert not src.exists()
    assert any("reached no inbox" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- caps
def _inbox_file(inbox: Path, start: dt.datetime, size: int, cam="01-garage") -> Path:
    p = inbox / ls.inbox_name(cam, f"r0001_{start:%Y%m%dT%H%M%S}Z.wav")
    p.write_bytes(b"x" * size)
    return p


def test_prune_by_age_then_size_oldest_first(tmp_path):
    inbox = tmp_path / "in"; inbox.mkdir()
    now = BASE + dt.timedelta(hours=2)
    old = _inbox_file(inbox, BASE, 10)                                  # 2 h old: over max age
    f1 = _inbox_file(inbox, now - dt.timedelta(seconds=90), 40)
    f2 = _inbox_file(inbox, now - dt.timedelta(seconds=60), 40)
    f3 = _inbox_file(inbox, now - dt.timedelta(seconds=30), 40, cam="00-doorbell")
    other = inbox / "notes.txt"; other.write_bytes(b"y" * 1000)          # not ours: never counted or touched
    got = ls.prune_inbox(inbox, max_age_s=3600, max_bytes=100, now_utc=now)
    assert got == (1, 1, 2, 80, now - dt.timedelta(seconds=60))
    assert not old.exists() and not f1.exists() and f2.exists() and f3.exists() and other.exists()


def test_prune_empty_and_within_caps(tmp_path):
    inbox = tmp_path / "in"; inbox.mkdir()
    assert ls.prune_inbox(inbox, max_age_s=60, max_bytes=1, now_utc=BASE) == (0, 0, 0, 0, None)
    _inbox_file(inbox, BASE, 5)
    assert ls.prune_inbox(inbox, max_age_s=60, max_bytes=5, now_utc=BASE) == (0, 0, 1, 5, BASE)


# ----------------------------------------------------------------- status + health
def _status(now, *, written=None, started=None, cams=None):
    return {"version": 1, "written_utc": ls._iso(written or now), "started_utc": ls._iso(started or now),
            "cameras": cams or {}, "inboxes": {}}


def test_evaluate_fresh_stale_never_and_status_age():
    now = BASE + dt.timedelta(hours=1)
    cams = {"garage": {"last_segment_utc": ls._iso(now - dt.timedelta(seconds=40)), "fails": 0},
            "doorbell": {"last_segment_utc": ls._iso(now - dt.timedelta(seconds=900)), "fails": 7}}
    assert ls.evaluate(_status(now, cams={"garage": cams["garage"]}), now, camera_stale_s=300) == []
    assert ls.evaluate(_status(now, cams=cams), now, camera_stale_s=300) == \
        ["doorbell: no segment for 900s (fails=7)"]
    never = {"doorbell": {"last_segment_utc": None, "fails": 3}}
    assert ls.evaluate(_status(now, started=now - dt.timedelta(seconds=100), cams=never), now, camera_stale_s=300) == []
    assert ls.evaluate(_status(now, started=now - dt.timedelta(seconds=400), cams=never), now, camera_stale_s=300) == \
        ["doorbell: no segment since start for 400s (fails=3)"]
    assert ls.evaluate(_status(now, written=now - dt.timedelta(seconds=61)), now, camera_stale_s=300) == \
        [f"status not updated since {ls._iso(now - dt.timedelta(seconds=61))}"]


def test_write_status_atomic_and_readable(tmp_path):
    ls.write_status(tmp_path, {"a": 1})
    ls.write_status(tmp_path, {"a": 2})
    assert json.loads((tmp_path / "status.json").read_text()) == {"a": 2}
    assert sorted(os.listdir(tmp_path)) == ["status.json"]


def test_healthcheck(tmp_path):
    env = {"AUDIO_DIR": str(tmp_path), "CAMERA_STALE_S": "300"}
    now = BASE + dt.timedelta(hours=1)
    ok, msg = ls.healthcheck(env, now_utc=now, started_at=now.timestamp() - 10)
    assert ok and "start-up grace" in msg
    ok, msg = ls.healthcheck(env, now_utc=now, started_at=now.timestamp() - 1000)
    assert not ok and "missing" in msg
    ls.write_status(tmp_path, _status(now, cams={"garage": {"last_segment_utc": ls._iso(now), "fails": 0}}))
    assert ls.healthcheck(env, now_utc=now) == (True, "1 cameras fresh")
    ls.write_status(tmp_path, _status(now, cams={"garage": {"last_segment_utc": ls._iso(BASE), "fails": 2}}))
    ok, msg = ls.healthcheck(env, now_utc=now)
    assert not ok and msg == "garage: no segment for 3600s (fails=2)"
    (tmp_path / "status.json").write_text("{not json")
    assert not ls.healthcheck(env, now_utc=now)[0]
    assert ls.healthcheck({**env, "CAMERA_STALE_S": "x"}, now_utc=now) == \
        (False, "CAMERA_STALE_S must be a whole number, got 'x'")


def test_main_healthcheck_exit_codes(tmp_path, capsys):
    now = dt.datetime.now(UTC)
    env = {"AUDIO_DIR": str(tmp_path)}
    ls.write_status(tmp_path, _status(now, cams={"garage": {"last_segment_utc": ls._iso(now), "fails": 0}}))
    assert ls.main(["--healthcheck"], env) == 0
    assert capsys.readouterr().out.startswith("healthy: ")
    ls.write_status(tmp_path, _status(now, written=now - dt.timedelta(minutes=5)))
    assert ls.main(["--healthcheck"], env) == 1
    assert capsys.readouterr().out.startswith("unhealthy: status not updated")


def test_main_config_error_and_bad_args(capsys, caplog):
    assert ls.main([], {"CONSUMERS": "a"}) == 2
    assert ls.main(["--what"], {}) == 2


def test_prepare_layout_and_stale_cleanup(tmp_path, caplog):
    cfg = ls.ListenerConfig(cameras=(bl.Camera("garage", "rtsp://h/x"),), consumers=("a", "b"), audio_dir=tmp_path)
    stale = tmp_path / "segments" / "00-garage"; stale.mkdir(parents=True)
    make_wav(stale / "r0003_20261005T231500Z.wav", 0.2)
    waiting = tmp_path / "inbox" / "a"; waiting.mkdir(parents=True)
    keep = make_wav(waiting / "00-garage_r0002_20261005T231500Z.wav", 0.2)
    (tmp_path / "inbox" / "old-app").mkdir()
    with caplog.at_level(logging.INFO):
        inboxes = ls.prepare(cfg)
    assert inboxes == {"a": tmp_path / "inbox" / "a", "b": tmp_path / "inbox" / "b"}
    assert (tmp_path / "segments" / "queued").is_dir()
    assert not any(stale.iterdir()) and keep.exists()          # crashed run's segment cleared, inbox kept
    assert any("not in CONSUMERS" in r.getMessage() and "old-app" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- integration
class ListenerRun:
    """run_listener in a background thread against fake ffmpegs, with a UTC
    clock that starts at BASE so the fake's segment names are 'recent'."""

    def __init__(self, tmp_path, plans, *, cameras=("doorbell", "garage"), consumers=("birdlisten", "transcriber"),
                 **cfg_kw):
        self.cfg = ls.ListenerConfig(
            cameras=tuple(bl.Camera(n, SECRET_URL) for n in cameras), consumers=consumers,
            audio_dir=tmp_path / "audio", clip_seconds=30, **cfg_kw)
        self.spawner = Spawner(plans)
        self.stop = threading.Event()
        t0 = time.monotonic()
        self.now = lambda: BASE + dt.timedelta(seconds=30 * (time.monotonic() - t0))
        self.rc: list[int] = []
        self.thread = threading.Thread(target=self._run, name="listener-test", daemon=True)

    def _run(self):
        self.rc.append(ls.run_listener(self.cfg, stop=self.stop, spawn=self.spawner, tuning=FAST,
                                       now_utc=self.now, status_every_s=0.05, summary_s=0.2, inbox_warn_s=60))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        for f in self.spawner.all_fakes():
            f.stderr_release.set()
            if f.alive():
                f.kill()
            f._writer.join(timeout=1)
        assert not self.thread.is_alive()

    def inbox(self, c) -> list[str]:
        d = self.cfg.inbox(c)
        return sorted(n for n in os.listdir(d) if ls.parse_inbox_name(n)) if d.is_dir() else []

    def status(self) -> dict:
        return json.loads((self.cfg.audio_dir / "status.json").read_text())


FIVE = dict(n_segments=5, gap_s=0.02, exit_after_s=5.0)
REFUSED_404 = dict(n_segments=0, exit_code=1, stderr_lines=[f"{SECRET_URL}: Server returned 404 Not Found"])


def test_listener_fans_out_and_flags_dead_camera(tmp_path, caplog):
    plans = {"00-doorbell": [REFUSED_404] * 200, "01-garage": [FIVE]}
    with caplog.at_level(logging.INFO), ListenerRun(tmp_path, plans) as run:
        # the 5th segment completes when ffmpeg exits (exit_after_s), so 4 arrive while it runs
        wait_for(lambda: len(run.inbox("birdlisten")) >= 4, timeout=5)
        wait_for(lambda: run.status()["cameras"]["garage"]["segments"] >= 4, timeout=2)
        assert run.inbox("birdlisten") == run.inbox("transcriber")
        for n in run.inbox("birdlisten"):
            assert n.startswith("01-garage_r0001_")
            assert os.stat(run.cfg.inbox("birdlisten") / n).st_nlink == 2
        st = run.status()
        assert st["cameras"]["doorbell"]["segments"] == 0 and st["cameras"]["doorbell"]["fails"] >= 1
        assert st["inboxes"]["birdlisten"]["files"] == len(run.inbox("birdlisten"))
        # the doorbell outage makes the healthcheck fail once it is older than CAMERA_STALE_S
        later = ls._from_iso(st["written_utc"]) + dt.timedelta(seconds=400)
        problems = ls.evaluate(st, later, camera_stale_s=300)
        assert any(p.startswith("doorbell: no segment since start") for p in problems)
        # consumers finish: every link deleted, nothing left anywhere on the volume
        for c in run.cfg.consumers:
            for n in run.inbox(c):
                (run.cfg.inbox(c) / n).unlink()
        assert not os.listdir(run.cfg.segment_dir / "queued")
    assert run.rc == [0]
    assert not [f for f in run.spawner.all_fakes() if f.alive()]
    final = run.status()
    assert all(not c["up"] for c in final["cameras"].values())
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("listener summary cameras_up=") for m in msgs)
    assert any("404 Not Found" in m for m in msgs)
    assert not [m for m in msgs if "s3cret" in m]


def test_listener_caps_a_stopped_consumer_without_blocking_the_other(tmp_path, caplog):
    seg_bytes = 44 + int(0.2 * 48000) * 2                  # one fake segment
    plans = {"00-garage": [dict(n_segments=None, gap_s=0.02)]}
    with caplog.at_level(logging.INFO), ListenerRun(tmp_path, plans, cameras=("garage",),
                                                    inbox_max_bytes=3 * seg_bytes) as run:
        taken = []

        def drain():                                       # the live consumer keeps up
            for n in run.inbox("transcriber"):
                (run.cfg.inbox("transcriber") / n).unlink()
                taken.append(n)
            return len(taken) >= 10

        wait_for(drain, timeout=5)
        wait_for(lambda: run.status()["inboxes"]["birdlisten"]["pruned_size"] > 0, timeout=2)
        assert len(run.inbox("birdlisten")) <= 3 + 2          # cap plus what arrived since the last prune
    assert run.rc == [0]
    warns = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(warns) == 1 and warns[0].startswith("inbox birdlisten: dropped")   # rate limited
    assert not [r for r in caplog.records if "inbox transcriber: dropped" in r.getMessage()]


def test_listener_unusable_audio_dir_returns_1(tmp_path, caplog):
    blocker = tmp_path / "audio"; blocker.write_text("a file, not a dir")
    cfg = ls.ListenerConfig(cameras=(bl.Camera("g", SECRET_URL),), consumers=("a",), audio_dir=blocker)
    assert ls.run_listener(cfg, stop=threading.Event(), tuning=FAST) == 1
    assert any("not usable" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="needs a real ffmpeg (CI runs it in the image)")
def test_selftest_real_ffmpeg():
    assert ls.selftest() == []
