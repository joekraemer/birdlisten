"""Unit tests for the parts of birdlisten that don't need a camera or the
BirdNET model. Run: uv run --group dev pytest -q"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

import birdlisten as bl

UTC = dt.timezone.utc


# ----------------------------------------------------------------- config
def test_parse_cameras_two():
    cams = bl.parse_cameras("front=rtsp://u:p@10.0.0.1:554/a, back = rtsp://u:p@10.0.0.2:554/b")
    assert [c.name for c in cams] == ["front", "back"]
    assert cams[1].rtsp == "rtsp://u:p@10.0.0.2:554/b"


@pytest.mark.parametrize("raw", ["", "front", "front=http://x", "a=rtsp://x,a=rtsp://y"])
def test_parse_cameras_rejects(raw):
    with pytest.raises(bl.ConfigError):
        bl.parse_cameras(raw)


def test_redacted_hides_password():
    cam = bl.Camera("c", "rtsp://admin:s3cret@192.168.1.5:554/h264Preview_01_sub")
    assert "s3cret" not in cam.redacted()
    assert cam.redacted().startswith("rtsp://admin:***@192.168.1.5")


@pytest.mark.parametrize("text", [
    "Error opening input: rtsp://admin:s3cret@192.168.1.5:554/x",
    "rtsp://admin:s3cret@192.168.1.5:554/x?tcp: Connection refused",
    "two urls rtsp://a:pw1@h1/x and rtsps://b:pw2@h2/y here",
    "percent rtsp://admin:s3%40cret@192.168.1.5/x",
])
def test_scrub_redacts_any_url_form(text):
    out = bl.scrub(text)
    for secret in ("s3cret", "pw1", "pw2", "s3%40cret"):
        assert secret not in out
    assert "***@" in out


def test_capture_error_is_scrubbed(tmp_path: Path, monkeypatch):
    cam = bl.Camera("c", "rtsp://admin:s3cret@10.0.0.1:554/x")

    class Proc:
        returncode = 1
        stderr = "rtsp://admin:s3cret@10.0.0.1:554/x?tcp: Connection refused\n"

    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(RuntimeError) as exc:
        bl.capture(cam, 3, tmp_path / "o.wav")
    assert "s3cret" not in str(exc.value) and "Connection refused" in str(exc.value)


def test_load_config_defaults_and_required():
    env = {"CAMERAS": "c=rtsp://x", "LATITUDE": "47.6", "LONGITUDE": "-122.3"}
    cfg = bl.load_config(env)
    assert cfg.clip_seconds == 30 and cfg.min_conf == 0.5 and cfg.ntfy_topic is None
    assert cfg.notify_cooldown == dt.timedelta(minutes=60)
    with pytest.raises(bl.ConfigError, match="LATITUDE"):
        bl.load_config({"CAMERAS": "c=rtsp://x"})
    with pytest.raises(bl.ConfigError, match="numeric"):
        bl.load_config({**env, "CLIP_SECONDS": "thirty"})


# ----------------------------------------------------------------- dedupe
def test_best_per_species_keeps_most_confident():
    d = bl.Detection
    dets = [
        d("Song Sparrow", "Melospiza melodia", 0.61, 0, 3),
        d("Song Sparrow", "Melospiza melodia", 0.88, 6, 9),
        d("American Robin", "Turdus migratorius", 0.55, 3, 6),
    ]
    best = bl.best_per_species(dets)
    assert set(best) == {"Song Sparrow", "American Robin"}
    assert best["Song Sparrow"].confidence == 0.88 and best["Song Sparrow"].start == 6


# ----------------------------------------------------------------- store
def test_record_and_report(tmp_path: Path, capsys):
    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    now = dt.datetime(2026, 9, 16, 14, 0, tzinfo=UTC)
    cam = cfg.cameras[0]
    bl.record(conn, now, cam, bl.Detection("Song Sparrow", "M. melodia", 0.9, 0, 3), None)
    bl.record(conn, now, cam, bl.Detection("Song Sparrow", "M. melodia", 0.7, 12, 15), None)
    bl.record(conn, now, cam, bl.Detection("Bushtit", "P. minimus", 0.6, 3, 6), None)
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 3
    conn.close()

    # report() opens its own connection and uses "now"; a 10-year window is safe.
    assert bl.report(cfg, 3650) == 0
    out = capsys.readouterr().out
    assert "Song Sparrow" in out and "Bushtit" in out
    assert out.index("Song Sparrow") < out.index("Bushtit")  # ordered by count desc


def test_should_notify_cooldown(tmp_path: Path):
    conn = bl.open_db(tmp_path)
    t0 = dt.datetime(2026, 9, 16, 14, 0, tzinfo=UTC)
    cd = dt.timedelta(minutes=60)
    assert bl.should_notify(conn, "Bushtit", t0, cd) is True
    assert bl.should_notify(conn, "Bushtit", t0 + dt.timedelta(minutes=30), cd) is False
    assert bl.should_notify(conn, "Bushtit", t0 + dt.timedelta(minutes=61), cd) is True
    assert bl.should_notify(conn, "Varied Thrush", t0, cd) is True  # different species unaffected


# ----------------------------------------------------------------- one pass, mocked
def test_listen_once_with_mocks(tmp_path: Path, monkeypatch):
    cfg = bl.load_config({
        "CAMERAS": "front=rtsp://x,back=rtsp://y", "LATITUDE": "0", "LONGITUDE": "0",
        "DATA_DIR": str(tmp_path), "KEEP_CLIPS": "1",
    })
    conn = bl.open_db(cfg.data_dir)

    def fake_capture(cam, seconds, out):
        if cam.name == "back":
            raise RuntimeError("ffmpeg failed: connection refused")
        out.write_bytes(b"RIFF" + b"\0" * 100_000)

    def fake_analyze(wav, cfg_, when):
        return [bl.Detection("Bushtit", "P. minimus", 0.7, 0, 3), bl.Detection("Bushtit", "P. minimus", 0.9, 3, 6)]

    sent = []
    monkeypatch.setattr(bl, "capture", fake_capture)
    monkeypatch.setattr(bl, "analyze", fake_analyze)
    monkeypatch.setattr(bl, "notify", lambda cfg_, title, body: sent.append(title))

    # One camera failed, one worked: pass counts as ok (1), the failure is logged.
    assert bl.listen_once(cfg, conn) == 1
    rows = conn.execute("SELECT camera, common_name, confidence, clip_path FROM detections").fetchall()
    assert len(rows) == 1 and rows[0][0] == "front" and rows[0][2] == 0.9
    assert rows[0][3] and Path(rows[0][3]).exists()  # KEEP_CLIPS copied the WAV
    assert sent == ["Bushtit"]

    # Second pass within the cooldown: recorded again, NOT notified again.
    assert bl.listen_once(cfg, conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2
    assert sent == ["Bushtit"]

    # Dry run writes nothing.
    assert bl.listen_once(cfg, conn, dry_run=True) == 1
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2


def test_listen_once_all_cameras_fail(tmp_path: Path, monkeypatch):
    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    monkeypatch.setattr(bl, "capture", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert bl.listen_once(cfg, conn) == 0
