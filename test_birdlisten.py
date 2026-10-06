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


_ENV = {"CAMERAS": "a=rtsp://x,b=rtsp://y,c=rtsp://z", "LATITUDE": "0", "LONGITUDE": "0"}


def test_load_config_stream_defaults():
    cfg = bl.load_config(_ENV)
    assert cfg.capture_mode == "stream"
    assert cfg.queue_size == 6  # 2 x 3 cameras
    assert cfg.summary_minutes == 5
    assert cfg.segment_dir == Path("/tmp/birdlisten-segments")


def test_load_config_capture_mode_normalised():
    assert bl.load_config({**_ENV, "CAPTURE_MODE": " RoundRobin "}).capture_mode == "roundrobin"
    assert bl.load_config({**_ENV, "CAPTURE_MODE": ""}).capture_mode == "stream"


@pytest.mark.parametrize("key, value, var", [
    ("CAPTURE_MODE", "parallel", "CAPTURE_MODE"),
    ("QUEUE_SIZE", "0", "QUEUE_SIZE"),
    ("QUEUE_SIZE", "1001", "QUEUE_SIZE"),
    ("SUMMARY_MINUTES", "0", "SUMMARY_MINUTES"),
    ("SUMMARY_MINUTES", "1441", "SUMMARY_MINUTES"),
    ("SEGMENT_DIR", "relative/seg", "SEGMENT_DIR"),
    ("SEGMENT_DIR", "/tmp/seg%d", "SEGMENT_DIR"),
    ("CLIP_SECONDS", "2", "CLIP_SECONDS"),
])
def test_load_config_rejects_stream_settings(key, value, var):
    with pytest.raises(bl.ConfigError, match=var):
        bl.load_config({**_ENV, key: value})


def test_load_config_segment_dir_vs_data_dir(tmp_path: Path):
    data = tmp_path / "data"
    env = {**_ENV, "DATA_DIR": str(data)}
    with pytest.raises(bl.ConfigError, match="SEGMENT_DIR"):
        bl.load_config({**env, "SEGMENT_DIR": str(data)})
    with pytest.raises(bl.ConfigError, match="SEGMENT_DIR"):
        bl.load_config({**env, "SEGMENT_DIR": str(data / "seg")})
    # A sibling sharing the string prefix is not inside DATA_DIR.
    sibling = tmp_path / "data-seg"
    assert bl.load_config({**env, "SEGMENT_DIR": str(sibling)}).segment_dir == sibling


def test_repr_hides_password():
    env = {**_ENV, "CAMERAS": "garage=rtsp://admin:s3cret@192.168.1.5:554/h264Preview_01_sub"}
    cfg = bl.load_config(env)
    for text in (repr(cfg.cameras[0]), repr(cfg)):
        assert "s3cret" not in text and "rtsp://admin:***@192.168.1.5" in text
    assert repr(cfg.cameras[0]) == "Camera(name='garage', rtsp='rtsp://admin:***@192.168.1.5:554/h264Preview_01_sub')"


def test_main_stream_urls():
    cfg = bl.load_config({**_ENV, "CAMERAS": "a=rtsp://h/h264Preview_01_main,b=rtsp://h/h264Preview_01_sub,"
                                             "c=rtsp://h/Preview_01_MAIN"})
    assert bl.main_stream_urls(cfg) == ["a", "c"]


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


def _timing_fields(caplog, prefix: str) -> list[dict[str, str]]:
    out = []
    for r in caplog.records:
        msg = r.getMessage()
        if msg.startswith(prefix):
            out.append(dict(tok.split("=", 1) for tok in msg.split()[1:] if "=" in tok))
    return out


def test_listen_once_logs_timing(tmp_path: Path, monkeypatch, caplog):
    cfg = bl.load_config({
        "CAMERAS": "front=rtsp://x,back=rtsp://admin:s3cret@10.0.0.2/y", "LATITUDE": "0", "LONGITUDE": "0",
        "DATA_DIR": str(tmp_path),
    })
    conn = bl.open_db(cfg.data_dir)

    def fake_capture(cam, seconds, out):
        if cam.name == "back":
            raise RuntimeError("ffmpeg failed: rtsp://admin:s3cret@10.0.0.2/y Connection refused")
        out.write_bytes(b"RIFF" + b"\0" * 100_000)

    def fake_analyze(wav, cfg_, when):
        return [bl.Detection("Bushtit", "P. minimus", 0.7, 0, 3), bl.Detection("Bushtit", "P. minimus", 0.9, 3, 6),
                bl.Detection("Song Sparrow", "M. melodia", 0.8, 6, 9)]

    monkeypatch.setattr(bl, "capture", fake_capture)
    monkeypatch.setattr(bl, "analyze", fake_analyze)
    monkeypatch.setattr(bl, "notify", lambda *a: None)
    caplog.set_level("INFO", logger="birdlisten")
    assert bl.listen_once(cfg, conn, dry_run=True) == 1

    cams = {f["camera"]: f for f in _timing_fields(caplog, "timing camera=")}
    assert set(cams) == {"front", "back"}
    front = cams["front"]
    assert set(front) == {"camera", "clip_s", "capture_s", "analyze_s", "detections", "cpu_s", "rss_mb"}
    assert front["clip_s"] == "30.0" and front["detections"] == "2"
    for k in ("capture_s", "analyze_s", "cpu_s", "rss_mb"):
        assert float(front[k]) >= 0

    back = cams["back"]
    assert back["error"].startswith("ffmpeg_failed:") and "s3cret" not in back["error"]
    assert "capture_s" in back and "analyze_s" not in back and "detections" not in back

    (p,) = _timing_fields(caplog, "timing pass ")
    assert p["cameras"] == "2" and p["ok"] == "1"
    assert float(p["wall_s"]) >= float(p["analyze_total_s"]) >= 0


def test_listen_once_timing_on_analysis_failure(tmp_path: Path, monkeypatch, caplog):
    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    monkeypatch.setattr(bl, "capture", lambda cam, s, out: None)
    monkeypatch.setattr(bl, "analyze", lambda *a: (_ for _ in ()).throw(RuntimeError("model missing")))
    caplog.set_level("INFO", logger="birdlisten")
    assert bl.listen_once(cfg, conn) == 0
    (f,) = _timing_fields(caplog, "timing camera=")
    assert f["error"] == "model_missing" and "analyze_s" in f and "cpu_s" in f and "detections" not in f


def test_model_load_logged_once_and_kept_out_of_analyze_s(tmp_path: Path, monkeypatch, caplog):
    import sys
    import time
    import types

    class SlowAnalyzer:
        def __init__(self):
            time.sleep(0.3)

    fake = types.ModuleType("birdnetlib.analyzer")
    fake.Analyzer = SlowAnalyzer
    monkeypatch.setitem(sys.modules, "birdnetlib", types.ModuleType("birdnetlib"))
    monkeypatch.setitem(sys.modules, "birdnetlib.analyzer", fake)
    monkeypatch.setattr(bl, "_ANALYZER", None)
    monkeypatch.setattr(bl, "_MODEL_LOAD_COST", [0.0, 0.0])
    monkeypatch.setattr(bl, "capture", lambda cam, s, out: None)
    monkeypatch.setattr(bl, "analyze", lambda *a: (bl.analyzer(), [])[1])

    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    caplog.set_level("INFO", logger="birdlisten")
    bl.listen_once(cfg, conn)
    bl.listen_once(cfg, conn)

    (load,) = _timing_fields(caplog, "timing model_load_s=")
    assert float(load["model_load_s"]) >= 0.3
    for f in _timing_fields(caplog, "timing camera="):
        assert float(f["analyze_s"]) < 0.2


def test_listen_once_all_cameras_fail(tmp_path: Path, monkeypatch):
    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    monkeypatch.setattr(bl, "capture", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert bl.listen_once(cfg, conn) == 0


# ----------------------------------------------------------------- shared helpers
def test_fmt_timing_skips_none_and_puts_error_last():
    line = bl._fmt_timing("camera=garage", [
        ("error", "ffmpeg failed: rtsp://admin:s3cret@10.0.0.2/y refused", "{}"),
        ("capture_s", 1.234, "{:.2f}"),
        ("analyze_s", None, "{:.2f}"),
        ("detections", 3, "{}"),
    ])
    assert line == "timing camera=garage capture_s=1.23 detections=3 error=ffmpeg_failed:_rtsp://admin:***@10.0.0.2/y_refused"
    assert bl._fmt_timing("summary", [("error", None, "{}"), ("ok", 2, "{}")]) == "timing summary ok=2"


def test_store_clip_returns_notify_time(tmp_path: Path, monkeypatch):
    import time

    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    conn = bl.open_db(cfg.data_dir)
    monkeypatch.setattr(bl, "notify", lambda *a: time.sleep(0.05))
    best = {"Bushtit": bl.Detection("Bushtit", "P. minimus", 0.9, 3, 6)}
    when = dt.datetime(2026, 9, 16, 14, 0, tzinfo=UTC)
    notify_s = bl.store_clip(cfg, conn, cfg.cameras[0], when, tmp_path / "x.wav", best, clip_s=12.4, dry_run=False)
    assert notify_s >= 0.05
    assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1


def test_store_clip_float_seconds_in_nothing_above(tmp_path: Path, caplog):
    cfg = bl.load_config({"CAMERAS": "c=rtsp://x", "LATITUDE": "0", "LONGITUDE": "0", "DATA_DIR": str(tmp_path)})
    caplog.set_level("INFO", logger="birdlisten")
    when = dt.datetime(2026, 9, 16, 14, 0, tzinfo=UTC)
    assert bl.store_clip(cfg, None, cfg.cameras[0], when, tmp_path / "x.wav", {}, clip_s=12.4, dry_run=False) == 0.0
    assert "c: 12s, nothing above 0.50" in caplog.messages
