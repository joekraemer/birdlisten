"""Tests for serve.py against a live CollageServer on 127.0.0.1:<random>.
No network beyond loopback: an autouse fixture makes every artwork fetch
raise NotFound. Run: uv run --group dev pytest -q   (arm64 macOS: see README "Tests")"""

from __future__ import annotations

import datetime as dt
import io
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

import birdlisten as bl
import frame
import serve

UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test starts with fetch_url refusing. Tests that need a success
    or a non-404 failure monkeypatch frame.fetch_url again themselves."""
    def refuse(url, timeout=None):
        raise frame.NotFound(url)
    monkeypatch.setattr(frame, "fetch_url", refuse)
    frame._meta_tried.clear()


@pytest.fixture
def server(tmp_path):
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "artwork"))
    srv = serve.CollageServer(cfg, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.headers, resp.read()


def seed(data_dir: Path):
    conn = bl.open_db(data_dir)
    now = dt.datetime.now(UTC)
    rec = lambda when, cam, com, sci: bl.record(  # noqa: E731
        conn, when, bl.Camera(cam, "rtsp://x"), bl.Detection(com, sci, 0.9, 0, 3), None)
    rec(now - dt.timedelta(hours=2), "back", "American Robin", "Turdus migratorius")
    rec(now - dt.timedelta(minutes=90), "front", "American Robin", "Turdus migratorius")
    rec(now - dt.timedelta(minutes=5), "back", "Varied Thrush", "Ixoreus naevius")
    conn.close()


# ----------------------------------------------------------------- config
def test_load_serve_config(tmp_path: Path):
    assert serve.load_serve_config({}) is None
    assert serve.load_serve_config({"SERVE_PORT": "  "}) is None
    cfg = serve.load_serve_config({"SERVE_PORT": "8085", "DATA_DIR": str(tmp_path)})
    assert cfg.port == 8085 and cfg.hours == 24
    assert cfg.db_path == tmp_path / "birdlisten.sqlite"
    assert cfg.art.dir == tmp_path / "artwork" and cfg.art.ref == frame.DEFAULT_ARTWORK_REF
    cfg = serve.load_serve_config({"SERVE_PORT": "8085", "ARTWORK_DIR": "/x/y", "ARTWORK_REF": "main", "COLLAGE_HOURS": "6"})
    assert cfg.art.dir == Path("/x/y") and cfg.art.ref == "main" and cfg.hours == 6
    for env in ({"SERVE_PORT": "abc"}, {"SERVE_PORT": "0"}, {"SERVE_PORT": "70000"},
                {"SERVE_PORT": "8085", "COLLAGE_HOURS": "0"}, {"SERVE_PORT": "8085", "COLLAGE_HOURS": "x"},
                {"SERVE_PORT": "8085", "ARTWORK_REF": "a b"}, {"SERVE_PORT": "8085", "ARTWORK_REF": " "}):
        with pytest.raises(bl.ConfigError):
            serve.load_serve_config(env)


def test_start_from_env_bad_config_logs_and_returns_none(caplog):
    with caplog.at_level("ERROR", logger="serve"):
        assert serve.start_from_env({"SERVE_PORT": "abc"}) is None
    assert "config error:" in caplog.text and "server disabled" in caplog.text
    assert serve.start_from_env({}) is None


# ----------------------------------------------------------------- routes
def test_index_html(server):
    _, base = server
    status, headers, body = get(base + "/")
    text = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store" and int(headers["Content-Length"]) == len(body)
    assert "/collage.png?hours=24" in text and "/attribution" in text
    assert "setInterval" in text and 'http-equiv="refresh"' in text
    assert "X-Frame-Options" not in headers
    status, _, body = get(base + "/?hours=6")
    assert "/collage.png?hours=6" in body.decode()


def test_collage_png_sizes_and_400s(server, tmp_path: Path):
    srv, base = server
    status, headers, body = get(base + "/collage.png")
    assert status == 200 and headers["Content-Type"] == "image/png"
    im = Image.open(io.BytesIO(body))
    assert im.format == "PNG" and im.size == (1600, 1200)
    _, _, body = get(base + "/collage.png?w=800&h=600")
    assert Image.open(io.BytesIO(body)).size == (800, 600)
    assert srv.cache.renders == 2
    get(base + "/collage.png?w=800&h=600")
    assert srv.cache.renders == 2            # cached, no change
    for q in ("hours=0", "hours=abc", "w=99999", "h=10", "hours=721"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base + "/collage.png?" + q)
        assert exc.value.code == 400
        assert exc.value.read().startswith(b"bad request: ")
    # With rows, the collage re-renders and shows them.
    seed(tmp_path)
    _, _, body = get(base + "/collage.png?w=800&h=600")
    assert Image.open(io.BytesIO(body)).size == (800, 600) and srv.cache.renders == 3


def test_api_recent_shape(server, tmp_path: Path):
    _, base = server
    seed(tmp_path)
    status, headers, body = get(base + "/api/recent")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    data = json.loads(body)
    assert set(data) == {"hours", "generated_at", "species"} and data["hours"] == 24
    gen = dt.datetime.fromisoformat(data["generated_at"])
    assert gen.tzinfo is not None and gen.microsecond == 0
    assert [s["scientific_name"] for s in data["species"]] == ["Ixoreus naevius", "Turdus migratorius"]
    robin = data["species"][1]
    assert set(robin) == {"scientific_name", "common_name", "last_heard", "count", "cameras", "first_ever", "has_plate"}
    assert robin["common_name"] == "American Robin" and robin["count"] == 2
    assert robin["cameras"] == ["front", "back"] and robin["first_ever"] is True
    assert robin["has_plate"] is False
    dt.datetime.fromisoformat(robin["last_heard"])
    _, _, body = get(base + "/api/recent?hours=1")
    assert [s["scientific_name"] for s in json.loads(body)["species"]] == ["Ixoreus naevius"]


def test_attribution_always_200(server):
    srv, base = server
    status, headers, body = get(base + "/attribution")
    text = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert "fugleramme" in text and "CC BY-SA 4.0" in text and "not fetched yet" in text
    art = srv.cfg.art
    art.dir.mkdir(parents=True, exist_ok=True)
    (art.dir / "ATTRIBUTION.md").write_text("# Plates\n\n* turdus <b>bold</b>\n")
    _, _, body = get(base + "/attribution")
    text = body.decode()
    assert "turdus &lt;b&gt;bold&lt;/b&gt;" in text and "not fetched yet" not in text


def test_404_and_favicon(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/nope")
    assert exc.value.code == 404
    status, headers, body = get(base + "/favicon.ico")
    assert status == 204 and body == b""


def test_server_before_db_exists(server):
    _, base = server
    _, _, body = get(base + "/api/recent")
    assert json.loads(body)["species"] == []
    _, _, body = get(base + "/collage.png?w=400&h=300")
    assert Image.open(io.BytesIO(body)).size == (400, 300)


# ----------------------------------------------------------------- startup
def test_start_server_busy_port(tmp_path: Path, caplog):
    # A listening wildcard socket, otherwise the test is OS-dependent (with
    # allow_reuse_address a wildcard bind succeeds on macOS past a 127.0.0.1 blocker).
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    cfg = serve.ServeConfig(port=port, hours=24, db_path=tmp_path / "db.sqlite", art=frame.Artwork(tmp_path / "a"))
    thread = None
    try:
        with caplog.at_level("ERROR", logger="serve"):
            thread = serve.start_server(cfg)
        assert thread is None
        assert f"cannot bind SERVE_PORT={port}" in caplog.text
    finally:
        blocker.close()
        if thread is not None:   # bind unexpectedly succeeded: do not leak the server
            thread._target.__self__.shutdown()


def test_start_server_ok_then_serves(tmp_path: Path, monkeypatch):
    # Bind a free port the same way start_server does (wildcard), then talk to it.
    probe = socket.socket()
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    probe.close()
    env = {"SERVE_PORT": str(port), "DATA_DIR": str(tmp_path), "COLLAGE_HOURS": "12"}
    thread = serve.start_from_env(env)
    assert thread is not None and thread.daemon and thread.name == "serve"
    try:
        _, _, body = get(f"http://127.0.0.1:{port}/api/recent")
        assert json.loads(body) == {"hours": 12, "generated_at": json.loads(body)["generated_at"], "species": []}
    finally:
        thread._target.__self__.shutdown()
