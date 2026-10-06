"""Tests for serve.py against a live CollageServer on 127.0.0.1:<random>.
No network beyond loopback: an autouse fixture makes every artwork fetch
raise NotFound. Run: uv run --group dev pytest -q   (arm64 macOS: see README "Tests")"""

from __future__ import annotations

import datetime as dt
import html
import io
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

import birdlisten as bl
import facts
import frame
import serve

UTC = dt.timezone.utc


def refuse_facts(url, *a, **kw):
    raise AssertionError(f"facts network call in a test: {url}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test starts with fetch_url refusing. Tests that need a success
    or a non-404 failure monkeypatch frame.fetch_url again themselves."""
    def refuse(url, timeout=None):
        raise frame.NotFound(url)
    monkeypatch.setattr(frame, "fetch_url", refuse)
    monkeypatch.setattr(facts, "http_get", refuse_facts)
    frame._meta_tried.clear()


@pytest.fixture
def server(tmp_path):
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "artwork"),
                            facts=facts.FactsConfig(tmp_path / "facts", fetch=True))
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
    assert cfg.port == 8085 and cfg.hours == 24 and cfg.min_confidence == 0.5
    assert cfg.db_path == tmp_path / "birdlisten.sqlite"
    assert cfg.art.dir == tmp_path / "artwork" and cfg.art.ref == frame.DEFAULT_ARTWORK_REF
    cfg = serve.load_serve_config({"SERVE_PORT": "8085", "ARTWORK_DIR": "/x/y", "ARTWORK_REF": "main", "COLLAGE_HOURS": "6"})
    assert cfg.art.dir == Path("/x/y") and cfg.art.ref == "main" and cfg.hours == 6
    assert serve.load_serve_config({"SERVE_PORT": "8085", "MIN_CONFIDENCE": "0.9"}).min_confidence == 0.9
    for env in ({"SERVE_PORT": "abc"}, {"SERVE_PORT": "0"}, {"SERVE_PORT": "70000"},
                {"SERVE_PORT": "8085", "COLLAGE_HOURS": "0"}, {"SERVE_PORT": "8085", "COLLAGE_HOURS": "x"},
                {"SERVE_PORT": "8085", "ARTWORK_REF": "a b"}, {"SERVE_PORT": "8085", "ARTWORK_REF": " "},
                {"SERVE_PORT": "8085", "MIN_CONFIDENCE": "x"}, {"SERVE_PORT": "8085", "MIN_CONFIDENCE": "1.5"},
                {"SERVE_PORT": "8085", "MIN_CONFIDENCE": "nan"}):
        with pytest.raises(bl.ConfigError):
            serve.load_serve_config(env)


def test_start_from_env_bad_config_logs_and_returns_none(caplog):
    with caplog.at_level("ERROR", logger="serve"):
        assert serve.start_from_env({"SERVE_PORT": "abc"}) is None
    assert "config error:" in caplog.text and "server disabled" in caplog.text
    assert serve.start_from_env({}) is None


# ----------------------------------------------------------------- routes
def page_layout(text: str) -> dict:
    m = re.search(r'<script type="application/json" id="layout">(.*?)</script>', text, re.S)
    assert m, "no embedded layout"
    return json.loads(m.group(1))


def test_index_html(server):
    _, base = server
    status, headers, body = get(base + "/")
    text = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store" and int(headers["Content-Length"]) == len(body)
    lay = page_layout(text)
    assert lay["hours"] == 24 and lay["png"] == f"/collage.png?v={lay['token']}"
    assert f'<img id="c" src="/collage.png?v={lay["token"]}" width="1600" height="1200"' in text
    assert "/attribution" in text and 'http-equiv="refresh"' in text
    assert '<script src="/static/page.js" defer></script>' in text
    assert 'data-refresh-ms="60000"' in text
    assert "X-Frame-Options" not in headers
    status, _, body = get(base + "/?hours=6")
    assert page_layout(body.decode())["hours"] == 6


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


def test_non_bird_labels_leave_page_api_and_popup(server, tmp_path: Path):
    """Issue #10: a Dog detection at or above MIN_CONFIDENCE is kept in SQLite
    but is not listed by /api/recent or drawn on the collage, and its pop-up
    is a 404."""
    _, base = server
    seed(tmp_path)
    conn = bl.open_db(tmp_path)
    bl.record(conn, dt.datetime.now(UTC), bl.Camera("back", "rtsp://x"),
              bl.Detection("Dog", "Dog", 0.95, 0, 3), None)
    conn.close()
    data = json.loads(get(base + "/api/recent")[2])
    assert [s["scientific_name"] for s in data["species"]] == ["Ixoreus naevius", "Turdus migratorius"]
    layout = json.loads(get(base + "/api/layout?w=800&h=600")[2])
    assert "Dog" not in json.dumps(layout)
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/api/species/Dog")
    assert exc.value.code == 404
    conn = bl.open_db(tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM detections WHERE scientific_name = 'Dog'").fetchone()[0] == 1
    conn.close()


def test_min_confidence_filters_page_and_api(tmp_path: Path):
    """Rows below MIN_CONFIDENCE stay in the DB but leave /api/recent and the
    collage at once, even though the capture loop wrote them earlier."""
    seed(tmp_path)   # all at 0.9
    conn = bl.open_db(tmp_path)
    bl.record(conn, dt.datetime.now(UTC), bl.Camera("back", "rtsp://x"),
              bl.Detection("Mallard", "Anas platyrhynchos", 0.6, 0, 3), None)
    conn.close()
    base = {"port": 0, "hours": 24, "db_path": tmp_path / "birdlisten.sqlite", "art": frame.Artwork(tmp_path / "artwork")}
    names = lambda cfg: [s["scientific_name"] for s in serve.recent_json(cfg, 24)["species"]]  # noqa: E731
    assert names(serve.ServeConfig(**base, min_confidence=0.5))[0] == "Anas platyrhynchos"
    strict = serve.ServeConfig(**base, min_confidence=0.9)
    assert names(strict) == ["Ixoreus naevius", "Turdus migratorius"]
    assert [s.scientific_name for s in serve.load_species(strict, 24, frame.utcnow())] == names(strict)


def test_index_footer_keeps_attribution_link(server):
    _, base = server
    _, _, body = get(base + "/")
    text = body.decode()
    assert '<a href="/attribution">' in text and "Fugleramme" in text and "CC BY-SA 4.0" in text


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


# ----------------------------------------------------------------- audubon
JAYS = "362 I. Yellow billed Magpie - 2. Stellers Jay - 3. Ultramarine Jay - 4. Clark's Crow.jpg"


def _entry(plate, file, title, credit="University of Pittsburgh"):
    return {"plate": plate, "title": title, "file": file,
            "page": "https://commons.wikimedia.org/wiki/File:" + file.replace(" ", "_"),
            "credit": credit, "credit_url": f"http://pitt.example/{plate}?a=1&b=2",
            "on_plate": 4, "via": ["wikidata"]}


def _audubon(tmp_path: Path) -> frame.Audubon:
    table = tmp_path / "audubon.json"
    table.write_text(json.dumps({"edition": "havell", "species": {
        "Cyanocitta stelleri": _entry(362, JAYS, "Jays <script>"),
        "Aphelocoma californica": _entry(362, JAYS, "Jays <script>"),
        "Ixoreus naevius": _entry(369, "369 Varied Thrush.jpg", "Varied Thrush", credit="Pitt & Co"),
    }}))
    return frame.Audubon.load(tmp_path / "artwork" / "audubon", table)


def _cache_vignette(aud: frame.Audubon, sci: str) -> None:
    p = aud.vignette_path(sci)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (60, 80), (90, 70, 50)).save(p, "WEBP")


@pytest.fixture
def aud_server(tmp_path):
    aud = _audubon(tmp_path)
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "artwork", audubon=aud),
                            facts=facts.FactsConfig(tmp_path / "facts", fetch=True))
    srv = serve.CollageServer(cfg, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_audubon_fallback_config(tmp_path: Path):
    base = {"SERVE_PORT": "8085", "DATA_DIR": str(tmp_path)}
    on = serve.load_serve_config(base)
    assert on.art.audubon is not None and on.art.audubon.dir == tmp_path / "artwork" / "audubon"
    assert len(on.art.audubon.table) >= 400
    assert serve.load_serve_config({**base, "AUDUBON_FALLBACK": "1"}).art.audubon is not None
    assert serve.load_serve_config({**base, "AUDUBON_FALLBACK": "0"}).art.audubon is None
    for bad in ("2", "yes", ""):
        with pytest.raises(bl.ConfigError, match="AUDUBON_FALLBACK must be 0 or 1"):
            serve.load_serve_config({**base, "AUDUBON_FALLBACK": bad})


def test_audubon_fallback_missing_table_runs_off(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(frame, "AUDUBON_MAP", tmp_path / "gone.json")
    with caplog.at_level("ERROR", logger="frame"):
        cfg = serve.load_serve_config({"SERVE_PORT": "8085", "DATA_DIR": str(tmp_path)})
    assert cfg.art.audubon is None and "Audubon plates off" in caplog.text


def test_attribution_lists_both_sources(aud_server):
    srv, base = aud_server
    _, _, body = get(base + "/attribution")
    text = body.decode()
    assert "Fugleramme" in text and "CC BY-SA 4.0" in text            # Fugleramme section intact
    assert serve.CREDIT_HTML in text
    assert "<h2>Audubon</h2>" in text and "Robert Havell Jr." in text and "Public domain" in text
    assert "University of Pittsburgh" in text and "(none fetched yet)" in text
    aud = srv.cfg.art.audubon
    for sci in ("Cyanocitta stelleri", "Aphelocoma californica", "Ixoreus naevius"):
        _cache_vignette(aud, sci)
    _, _, body = get(base + "/attribution")
    text = body.decode()
    jays = "https://commons.wikimedia.org/wiki/File:" + html.escape(JAYS.replace(" ", "_"), quote=True)
    assert text.count(f'<a href="{jays}">Plate 362, Jays &lt;script&gt;</a>') == 1   # shared plate once
    assert "Plate 369, Varied Thrush" in text and "Pitt &amp; Co" in text
    assert 'href="http://pitt.example/369?a=1&amp;b=2"' in text
    assert "<script>" not in text and "(none fetched yet)" not in text
    assert text.index("Plate 362") < text.index("Plate 369")


def test_attribution_credits_size_table(server):
    """#5: the size table's source and licence are on /attribution."""
    _, base = server
    text = get(base + "/attribution")[2].decode()
    assert "<h2>Body mass</h2>" in text
    assert '<a href="https://doi.org/10.1111/ele.13898">AVONET</a>' in text
    assert '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>' in text
    assert "Tobias, J. A. et al. (2022)" in text
    t = serve.attribution_html(frame.Artwork(Path("/nonexistent")), None, None)
    assert "Body mass" not in t                       # no table, no section


def test_attribution_without_audubon_has_no_section(server):
    _, base = server
    _, _, body = get(base + "/attribution")
    assert "Audubon" not in body.decode()


def test_footer_names_audubon_when_on(server, aud_server):
    _, off = server
    _, on = aud_server
    off_text, on_text = get(off + "/")[2].decode(), get(on + "/")[2].decode()
    notes = " · notes from Wikipedia, Wikidata, eBird and AVONET</a>"
    assert '<a href="/attribution">Plates from Fugleramme, CC BY-SA 4.0' + notes in off_text
    assert ('<a href="/attribution">Plates from Fugleramme (CC BY-SA 4.0) and Audubon\'s '
            '<i>Birds of America</i>' + notes) in on_text
    lay = {"png": "/collage.png?v=" + "0" * 16, "targets": []}
    assert serve.index_html(lay, 800, 600) == serve.index_html(lay, 800, 600, audubon=False)


def test_api_recent_has_plate_with_only_a_vignette(aud_server, tmp_path: Path):
    srv, base = aud_server
    seed(tmp_path)
    _cache_vignette(srv.cfg.art.audubon, "Ixoreus naevius")
    data = json.loads(get(base + "/api/recent")[2])
    flags = {s["scientific_name"]: s["has_plate"] for s in data["species"]}
    assert flags == {"Ixoreus naevius": True, "Turdus migratorius": False}
    assert set(data["species"][0]) == {"scientific_name", "common_name", "last_heard", "count",
                                       "cameras", "first_ever", "has_plate"}


# ----------------------------------------------------------------- click targets
def test_index_embeds_layout_matching_image(server, tmp_path: Path):
    srv, base = server
    assert page_layout(get(base + "/")[2].decode())["targets"] == []      # empty db: no targets
    seed(tmp_path)
    text = get(base + "/")[2].decode()
    lay = page_layout(text)
    assert [t["scientific_name"] for t in lay["targets"]] == ["Ixoreus naevius", "Turdus migratorius"]
    assert re.search(r'src="/collage\.png\?v=([0-9a-f]{16})"', text).group(1) == lay["token"]
    r = srv.cache.by_token(lay["token"])
    assert r is not None and r.layout(1600, 1200)["targets"] == lay["targets"]
    # The layout block is the same object /api/layout returns.
    assert json.loads(get(base + "/api/layout")[2]) == lay


def test_layout_block_escapes_markup(server, tmp_path: Path):
    _, base = server
    conn = bl.open_db(tmp_path)
    bl.record(conn, dt.datetime.now(UTC), bl.Camera("back", "rtsp://x"),
              bl.Detection("</script><b>&", "Turdus migratorius", 0.9, 0, 3), None)
    conn.close()
    text = get(base + "/")[2].decode()
    block = re.search(r'id="layout">(.*?)</script>', text, re.S).group(1)
    assert "<" not in block and ">" not in block and "&" not in block
    assert "\\u003c/script\\u003e\\u003cb\\u003e\\u0026" in block
    assert page_layout(text)["targets"][0]["common_name"] == "</script><b>&"


def test_index_size_and_refresh_params(server):
    _, base = server
    text = get(base + "/?w=533&h=400&refresh_ms=1000")[2].decode()
    lay = page_layout(text)
    assert (lay["w"], lay["h"]) == (533, 400) and 'width="533" height="400"' in text
    assert 'data-refresh-ms="1000"' in text
    for q in ("w=10", "w=abc", "refresh_ms=1", "refresh_ms=x", "h=99999"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base + "/?" + q)
        assert exc.value.code == 400, q


def test_collage_by_token(server, tmp_path: Path):
    srv, base = server
    seed(tmp_path)
    lay = json.loads(get(base + "/api/layout?w=800&h=600")[2])
    status, headers, body = get(base + lay["png"])
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert body == srv.cache.by_token(lay["token"]).png
    assert Image.open(io.BytesIO(body)).size == (800, 600)
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/collage.png?v=" + "0" * 16)
    assert exc.value.code == 404 and exc.value.headers["Cache-Control"] == "no-store"
    for bad in ("xyz", "0" * 15, "A" * 16, "0" * 17):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base + "/collage.png?v=" + bad)
        assert exc.value.code == 400, bad
    _, headers, _ = get(base + "/collage.png")
    assert headers["Cache-Control"] == "no-store"


def test_api_layout_new_token_after_db_change(server, tmp_path: Path):
    srv, base = server
    seed(tmp_path)
    a = json.loads(get(base + "/api/layout?w=800&h=600")[2])
    conn = bl.open_db(tmp_path)
    bl.record(conn, dt.datetime.now(UTC), bl.Camera("back", "rtsp://x"),
              bl.Detection("Mallard", "Anas platyrhynchos", 0.9, 0, 3), None)
    conn.close()
    b = json.loads(get(base + "/api/layout?w=800&h=600")[2])
    assert a["token"] != b["token"] and len(b["targets"]) == 3
    assert b["targets"][0]["scientific_name"] == "Anas platyrhynchos"
    r = srv.cache.by_token(b["token"])
    assert get(base + b["png"])[2] == r.png and r.layout(800, 600)["targets"] == b["targets"]
    assert get(base + a["png"])[0] == 200        # the old token is still served
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/api/layout?w=1")
    assert exc.value.code == 400


def test_csp_and_nosniff(server):
    _, base = server
    for path in ("/", "/attribution"):
        _, headers, body = get(base + path)
        assert headers["Content-Security-Policy"] == serve.CSP
        assert "script-src 'self'" in serve.CSP and "frame-ancestors" not in serve.CSP
        assert headers["X-Content-Type-Options"] == "nosniff"
        text = body.decode()
        assert "<style" not in text
        for m in re.finditer(r"<script([^>]*)>(.*?)</script>", text, re.S):
            attrs, code = m.groups()
            assert 'src="' in attrs or 'type="application/json"' in attrs, attrs
            if 'src="' in attrs:
                assert code == ""
    _, headers, _ = get(base + "/api/recent")
    assert headers["X-Content-Type-Options"] == "nosniff" and "Content-Security-Policy" not in headers


def test_static_files(server):
    _, base = server
    for path, ctype in (("/static/page.js", "text/javascript; charset=utf-8"),
                        ("/static/page.css", "text/css; charset=utf-8")):
        status, headers, body = get(base + path)
        assert status == 200 and headers["Content-Type"] == ctype and body
        assert headers["Cache-Control"] == "max-age=300"
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base + "/static/../serve.py")
    assert exc.value.code == 404


def test_page_css_keeps_overlay_aligned():
    css = (Path(serve.__file__).parent / "static" / "page.css").read_text()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", css)
    assert rules
    for sel, body in rules:
        sels = [x.strip() for x in sel.split(",")]
        assert "img" not in sels, sel
        if "#c" in sels or "#stage" in sels:
            for bad in ("max-height", "object-fit", "aspect-ratio", "transform"):
                assert bad not in body, (sel, bad)
            assert all(v.strip() == "0" for v in re.findall(r"padding\s*:([^;]*)", body)), (sel, body)


# ----------------------------------------------------------------- facts config
def test_facts_config_defaults_and_validation(tmp_path: Path, caplog):
    base = {"SERVE_PORT": "8085", "DATA_DIR": str(tmp_path)}
    cfg = serve.load_serve_config(base)
    assert cfg.facts == facts.FactsConfig(tmp_path / "facts", True, None, None, None)
    assert str(cfg.tz) == "America/Los_Angeles"
    assert serve.load_serve_config({**base, "FACTS_FETCH": "0", "FACTS_DIR": "/x/f"}).facts.dir == Path("/x/f")
    assert serve.load_serve_config({**base, "FACTS_FETCH": "0"}).facts.fetch is False
    for bad in ("2", "yes", ""):
        with pytest.raises(bl.ConfigError, match="FACTS_FETCH must be 0 or 1"):
            serve.load_serve_config({**base, "FACTS_FETCH": bad})
    with caplog.at_level("INFO"):
        cfg = serve.load_serve_config({**base, "EBIRD_API_KEY": "bad key!SENTINELKEY123"})
    assert cfg.facts.ebird_key is None
    assert "EBIRD_API_KEY is malformed" in caplog.text and "SENTINELKEY123" not in caplog.text
    caplog.clear()
    with caplog.at_level("INFO"):
        cfg = serve.load_serve_config({**base, "EBIRD_API_KEY": "SENTINELKEY123", "LATITUDE": "47.6",
                                       "LONGITUDE": "-222"})
    assert cfg.facts.ebird_key.reveal() == "SENTINELKEY123" and not cfg.facts.nearby_on
    assert "nearby eBird reports off" in caplog.text
    cfg = serve.load_serve_config({**base, "EBIRD_API_KEY": "SENTINELKEY123", "LATITUDE": "47.6",
                                   "LONGITUDE": "-122.3"})
    assert cfg.facts.nearby_on and (cfg.facts.lat, cfg.facts.lon) == (47.6, -122.3)
    assert "SENTINELKEY123" not in repr(cfg) and "SENTINELKEY123" not in str(cfg.facts)
    for lat in ("x", "91", "nan"):
        c = serve.load_serve_config({**base, "EBIRD_API_KEY": "k", "LATITUDE": lat, "LONGITUDE": "1"})
        assert not c.facts.nearby_on


@pytest.mark.parametrize("raw,zone,warns", [
    (None, "America/Los_Angeles", False), ("", "America/Los_Angeles", False),
    ("  ", "America/Los_Angeles", False), (":America/Los_Angeles", "America/Los_Angeles", False),
    ("America/New_York", "America/New_York", False), ("Not/AZone", "America/Los_Angeles", True),
    ("/etc/passwd", "America/Los_Angeles", True), ("../x", "America/Los_Angeles", True),
])
def test_tz_rule(raw, zone, warns, tmp_path: Path, caplog):
    env = {} if raw is None else {"TZ": raw}
    with caplog.at_level("WARNING", logger="serve"):
        assert str(serve.load_tz(env)) == zone
    assert ("is not an IANA zone" in caplog.text) == warns
    probe = socket.socket()
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    probe.close()
    thread = serve.start_from_env({**env, "SERVE_PORT": str(port), "DATA_DIR": str(tmp_path)})
    assert thread is not None and thread.is_alive()
    srv = thread._target.__self__
    try:
        assert str(srv.cfg.tz) == zone
    finally:
        srv.shutdown()
        srv.server_close()


# ----------------------------------------------------------------- /api/species
def get_err(url: str) -> int:
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(url)
    return exc.value.code


class FactsUpstream:
    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers=None, timeout=None, max_bytes=None):
        self.calls.append((url, dict(headers or {})))
        for prefix, v in self.routes.items():
            if url.startswith(prefix):
                if isinstance(v, BaseException):
                    raise v
                return v
        raise frame.NotFound(url)


def insert(db_dir: Path, rows):
    """rows: (heard_at text, camera, common, sci, confidence)."""
    conn = bl.open_db(db_dir)
    conn.executemany("INSERT INTO detections(heard_at,camera,common_name,scientific_name,confidence,clip_offset_s)"
                     " VALUES (?,?,?,?,?,0)", rows)
    conn.commit()
    conn.close()


@pytest.mark.parametrize("seg", ["", "..%2F..%2Fetc%2Fpasswd", "%252F", "%00", "x" * 101, "%C3%A9",
                                 "Turdus/migratorius", "%E0%A4%A", "Turdus%20migratorius%3B", "a%5Cb"])
def test_species_rejects_malformed(seg, server, tmp_path: Path, monkeypatch):
    srv, base = server
    seed(tmp_path)
    up = FactsUpstream()
    monkeypatch.setattr(facts, "http_get", up)
    assert get_err(base + "/api/species/" + seg) == 400
    assert up.calls == [] and not (tmp_path / "facts").exists()


def test_species_unknown_and_hours(server, tmp_path: Path, monkeypatch):
    srv, base = server
    seed(tmp_path)
    up = FactsUpstream()
    monkeypatch.setattr(facts, "http_get", up)
    assert get_err(base + "/api/species/Corvus%20corax") == 404       # well-formed, never heard
    for q in ("hours=0", "hours=721", "hours=abc"):
        assert get_err(base + "/api/species/Turdus%20migratorius?" + q) == 400
    assert up.calls == [] and not (tmp_path / "facts").exists()
    data = json.loads(get(base + "/api/species/Turdus%20migratorius")[2])
    assert data["hours"] == 24                                         # COLLAGE_HOURS default
    assert data["heard"]["count"] == 2


def test_species_known_check_respects_min_confidence(tmp_path: Path):
    insert(tmp_path, [("2026-10-03T18:00:00+00:00", "back", "Mallard", "Anas platyrhynchos", 0.6)])
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "a"), min_confidence=0.9)
    assert serve.species_known(cfg, "Anas platyrhynchos") is False
    assert serve.species_known(serve.ServeConfig(**{**cfg.__dict__, "min_confidence": 0.5}), "Anas platyrhynchos")


def test_species_audubon_known_and_fallback_off(aud_server, tmp_path: Path, monkeypatch):
    srv, base = aud_server
    monkeypatch.setattr(facts, "http_get", FactsUpstream())
    # No db yet: an Audubon-table species is known (deep links), with count 0.
    data = json.loads(get(base + "/api/species/Aphelocoma%20californica")[2])
    assert data["heard"] == {"count": 0, "by_hour": [0] * 24}
    assert data["common_name"] == "Jays <script>"                      # table title as last resort
    # AUDUBON_FALLBACK=0: art.audubon is None; unknown names are 404, db species 200.
    seed(tmp_path)
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "artwork"), facts=facts.FactsConfig(tmp_path / "f2", fetch=False))
    assert serve.species_known(cfg, "Aphelocoma californica") is False
    assert serve.species_known(cfg, "Turdus migratorius") is True


def test_species_stats(tmp_path: Path):
    from zoneinfo import ZoneInfo
    now = dt.datetime(2026, 10, 3, 19, 0, tzinfo=UTC)          # 12:00 PDT
    rows = [
        ("2026-10-03T18:30:00+00:00", "front", "Bushtit", "Psaltriparus minimus", 0.95),   # 11:30 AM
        ("2026-10-03T14:10:00+00:00", "back", "Bushtit", "Psaltriparus minimus", 0.80),    # 7:10 AM
        ("2026-10-03T14:40:00+00:00", "back", "Bushtit", "Psaltriparus minimus", 0.90),    # 7:40 AM
        ("2026-10-02T23:05:00+00:00", "side", "Bushtit", "Psaltriparus minimus", 0.70),    # 4:05 PM yesterday
        ("2026-10-03T15:00:00+00:00", "back", "Bushtit", "Psaltriparus minimus", 0.40),    # below MIN_CONFIDENCE
        ("2026-10-01T10:00:00+00:00", "back", "Bushtit", "Psaltriparus minimus", 0.99),    # outside 24 h
    ]
    insert(tmp_path, rows)
    cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                            art=frame.Artwork(tmp_path / "a"), tz=ZoneInfo("America/Los_Angeles"))
    heard, common = serve.species_stats(cfg, "Psaltriparus minimus", 24, now)
    assert common == "Bushtit"
    assert heard["count"] == 4 and heard["max_conf"] == 0.95
    assert heard["median_conf"] == 0.85                         # even count: mean of the middle two
    assert heard["first_heard"] == "2026-10-02T23:05:00+00:00" and heard["last_heard"] == "2026-10-03T18:30:00+00:00"
    assert heard["first_local"] == "Yesterday 4:05 PM" and heard["last_local"] == "11:30 AM"
    assert heard["cameras"] == ["front", "back", "side"]
    assert heard["by_hour"][7] == 2 and heard["by_hour"][11] == 1 and heard["by_hour"][16] == 1
    assert sum(heard["by_hour"]) == 4 and heard["busiest_hour"] == 7
    heard, _ = serve.species_stats(cfg, "Psaltriparus minimus", 72, now)
    assert heard["count"] == 5 and heard["first_local"] == "Oct 1, 3:00 AM"
    assert heard["median_conf"] == 0.9                          # odd count
    heard, common = serve.species_stats(cfg, "Psaltriparus minimus", 1, now + dt.timedelta(days=3))
    assert heard == {"count": 0, "by_hour": [0] * 24} and common == "Bushtit"


def test_species_stats_dst_and_naive_rows(tmp_path: Path, monkeypatch):
    from zoneinfo import ZoneInfo
    import time as _time
    monkeypatch.setenv("TZ", "Asia/Tokyo")                     # the process zone must not matter
    _time.tzset()
    try:
        insert(tmp_path, [
            ("2026-03-08T09:30:00+00:00", "a", "Robin", "Turdus migratorius", 0.9),   # 01 PST
            ("2026-03-08T10:30:00+00:00", "a", "Robin", "Turdus migratorius", 0.9),   # 03 PDT
            ("2026-11-01T08:30:00+00:00", "a", "Mallard", "Anas platyrhynchos", 0.9),   # 01 PDT
            ("2026-11-01T09:30:00+00:00", "a", "Mallard", "Anas platyrhynchos", 0.9),   # 01 PST
            ("2026-03-08T10:30:00", "a", "Robin", "Ixoreus naevius", 0.9),            # naive = UTC: 03 PDT
        ])
        cfg = serve.ServeConfig(port=0, hours=24, db_path=tmp_path / "birdlisten.sqlite",
                                art=frame.Artwork(tmp_path / "a"), tz=ZoneInfo("America/Los_Angeles"))
        heard, _ = serve.species_stats(cfg, "Turdus migratorius", 24, dt.datetime(2026, 3, 8, 20, 0, tzinfo=UTC))
        assert heard["count"] == 2 and heard["by_hour"][1] == 1 and heard["by_hour"][3] == 1   # spring forward
        heard, _ = serve.species_stats(cfg, "Anas platyrhynchos", 24, dt.datetime(2026, 11, 1, 20, 0, tzinfo=UTC))
        assert heard["count"] == 2 and heard["by_hour"][1] == 2                                # fall back
        heard, _ = serve.species_stats(cfg, "Ixoreus naevius", 24, dt.datetime(2026, 3, 8, 20, 0, tzinfo=UTC))
        assert heard["by_hour"][3] == 1 and heard["last_local"] == "3:30 AM"
        lt = serve.local_time("2026-03-08T10:30:00", ZoneInfo("America/Los_Angeles"))
        assert (lt.hour, lt.utcoffset()) == (3, dt.timedelta(hours=-7))
    finally:
        monkeypatch.delenv("TZ")
        _time.tzset()


def test_display_time():
    now = dt.datetime(2026, 10, 3, 12, 0)
    assert serve.display_time(dt.datetime(2026, 10, 3, 0, 5), now) == "12:05 AM"
    assert serve.display_time(dt.datetime(2026, 10, 3, 12, 41), now) == "12:41 PM"
    assert serve.display_time(dt.datetime(2026, 10, 2, 18, 41), now) == "Yesterday 6:41 PM"
    assert serve.display_time(dt.datetime(2026, 10, 1, 18, 41), now) == "Oct 1, 6:41 PM"


def test_species_response_shape_with_facts(server, tmp_path: Path, monkeypatch):
    srv, base = server
    seed(tmp_path)
    summary = json.dumps({"type": "standard", "title": "American_robin", "titles": {"normalized": "American robin"},
                          "extract": "A thrush.", "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/American_robin"}}}).encode()
    up = FactsUpstream({facts.WIKIDATA_API + "action=query": json.dumps({"query": {"search": []}}).encode(),
                        facts.SUMMARY_API + "Turdus_migratorius": summary})
    monkeypatch.setattr(facts, "http_get", up)
    status, headers, body = get(base + "/api/species/Turdus%20migratorius?hours=24")
    assert status == 200 and headers["Cache-Control"] == "no-store"
    data = json.loads(body)
    assert set(data) == {"scientific_name", "common_name", "hours", "binomial", "art", "plate_url",
                         "heard", "facts", "links", "pending", "tz"}
    assert data["binomial"] is True and data["art"] is None and data["plate_url"] is None
    assert data["facts"]["wikipedia"]["title"] == "American robin" and data["pending"] == []
    assert data["links"] == {"allaboutbirds": "https://www.allaboutbirds.org/guide/American_Robin"}
    assert data["tz"] == "America/Los_Angeles" and len(data["heard"]["by_hour"]) == 24
    assert data["heard"]["cameras"] == ["front", "back"]


def test_species_non_binomial_stats_only(server, tmp_path: Path, monkeypatch):
    _, base = server
    # A bird name that is not a binomial (Dog-style labels are now 404, see #10).
    insert(tmp_path, [(frame.utcnow().isoformat(timespec="seconds"), "back", "gull sp.", "Larus sp.", 0.9)])
    up = FactsUpstream()
    monkeypatch.setattr(facts, "http_get", up)
    data = json.loads(get(base + "/api/species/Larus%20sp.")[2])
    assert data["binomial"] is False and data["facts"] == {} and data["links"] == {}
    assert data["heard"]["count"] == 1 and up.calls == []


def test_species_external_failure_is_not_5xx(server, tmp_path: Path, monkeypatch):
    _, base = server
    seed(tmp_path)
    monkeypatch.setattr(facts, "http_get", FactsUpstream({"https://": urllib.error.URLError("down")}))
    data = json.loads(get(base + "/api/species/Turdus%20migratorius")[2])
    # Upstreams down: only the shipped size table (#5) still answers.
    assert set(data["facts"]) == {"size"} and data["heard"]["count"] == 2
    assert [src["name"] for src in data["facts"]["size"]["sources"]] == ["AVONET"]


# ----------------------------------------------------------------- /plate and /fonts
def test_plate_route(aud_server, tmp_path: Path, monkeypatch):
    srv, base = aud_server
    calls = []
    monkeypatch.setattr(frame, "fetch_url", lambda url, timeout=None: calls.append(url))
    art = srv.cfg.art
    p = art.plate_path("Turdus migratorius")
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (900, 600), (90, 60, 40, 255)).save(p, "WEBP")
    _cache_vignette(art.audubon, "Ixoreus naevius")
    status, headers, body = get(base + "/plate/turdus-migratorius.png")
    im = Image.open(io.BytesIO(body))
    assert status == 200 and headers["Content-Type"] == "image/png" and im.format == "PNG"
    assert im.size == (480, 320) and headers["Cache-Control"] == "max-age=3600"
    status, _, body = get(base + "/plate/ixoreus-naevius.png")
    assert status == 200 and Image.open(io.BytesIO(body)).format == "PNG"
    assert get_err(base + "/plate/corvus-corax.png") == 404
    for bad in ("/plate/..%2Fx.png", "/plate/Turdus.png", "/plate/a.b.png", "/plate/.png"):
        assert get_err(base + bad) == 400, bad
    p2 = art.plate_path("Corvus corax")
    p2.write_bytes(b"not an image")
    assert get_err(base + "/plate/corvus-corax.png") == 404 and p2.exists()     # never deleted
    assert calls == []


def test_fonts_routes(server):
    _, base = server
    for path, ctype in (("/fonts/LibreBaskerville.ttf", "font/ttf"),
                        ("/fonts/LibreBaskerville-Italic.ttf", "font/ttf"),
                        ("/fonts/OFL.txt", "text/plain; charset=utf-8")):
        status, headers, body = get(base + path)
        assert status == 200 and headers["Content-Type"] == ctype
        assert headers["Cache-Control"] == "max-age=86400" and body
        assert body == (frame.FONT_DIR / path.rsplit("/", 1)[1]).read_bytes()
    assert get_err(base + "/fonts/SOURCE.txt") == 404
    assert get_err(base + "/fonts/../serve.py") == 404


# ----------------------------------------------------------------- pop-up card markup
def test_card_skeleton_and_script_guards(server):
    _, base = server
    text = get(base + "/")[2].decode()
    assert '<section id="card" role="dialog" aria-modal="true" aria-labelledby="card-title" hidden>' in text
    assert '<h2 id="card-title"></h2>' in text and '<p class="sci"></p>' in text
    assert '<button type="button" class="close" aria-label="Close">' in text
    assert '<section class="heard"><h3>What we heard</h3>' in text
    assert '<section class="about"><h3>About the bird</h3>' in text
    assert text.index('<section class="about">') < text.index('<section class="heard">')   # #11
    assert '<div id="scrim" hidden></div>' in text
    js = (Path(serve.__file__).parent / "static" / "page.js").read_text()
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "onerror="):
        assert bad not in js, bad
    css = (Path(serve.__file__).parent / "static" / "page.css").read_text()
    assert 'url("/fonts/LibreBaskerville.ttf")' in css and 'url("/fonts/LibreBaskerville-Italic.ttf")' in css
    assert "font-display: swap" in css


def test_attribution_notes_sections(aud_server):
    srv, base = aud_server
    text = get(base + "/attribution")[2].decode()
    assert serve.CREDIT_HTML in text and "<h2>Audubon</h2>" in text
    for h in ("<h2>Wikipedia</h2>", "<h2>Wikidata</h2>", "<h2>eBird</h2>", "<h2>Libre Baskerville</h2>"):
        assert h in text
    assert text.index("<h2>Audubon</h2>") < text.index("<h2>Wikipedia</h2>")
    assert "CC BY-SA 4.0" in text and "CC0" in text and 'href="https://ebird.org"' in text
    assert 'href="/fonts/OFL.txt"' in text and "SIL Open Font License 1.1" in text
    assert "https://github.com/google/fonts/tree/9710da1eacb3be272583c3224dcb70f9da6eadbb/ofl/librebaskerville" in text
    wiki = text[text.index("<h2>Wikipedia</h2>"):text.index("<h2>Wikidata</h2>")]
    assert "(none cached yet)" in wiki
    d = srv.cfg.facts.dir
    facts.write_rec(d, "wikipedia", "psaltriparus-minimus", "ok", frame.utcnow(),
                    {"title": "Bush<b>tit&", "extract": "x", "trimmed": False,
                     "url": "https://en.wikipedia.org/wiki/A?b=1&c=2"})
    facts.write_rec(d, "wikipedia", "genus-species", "miss", frame.utcnow())
    text = get(base + "/attribution")[2].decode()
    wiki = text[text.index("<h2>Wikipedia</h2>"):text.index("<h2>Wikidata</h2>")]
    assert '<li><a href="https://en.wikipedia.org/wiki/A?b=1&amp;c=2">Bush&lt;b&gt;tit&amp;</a></li>' in wiki
    assert "(none cached yet)" not in wiki and "<b>" not in wiki


# ----------------------------------------------------------------- the eBird key never leaks (AC 46)
SENTINEL = "SENTINELKEY123"


def test_ebird_key_never_leaves_the_process(tmp_path: Path, monkeypatch, caplog):
    env = {"SERVE_PORT": "1", "DATA_DIR": str(tmp_path), "EBIRD_API_KEY": SENTINEL,
           "LATITUDE": "47.6", "LONGITUDE": "-122.3"}
    caplog.set_level("DEBUG")
    cfg = serve.load_serve_config(env)
    assert cfg.facts.ebird_key is not None and cfg.facts.nearby_on
    cfg = serve.ServeConfig(**{**cfg.__dict__, "port": 0})
    seed(tmp_path)
    p = cfg.art.plate_path("Turdus migratorius")
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (40, 40), (90, 60, 40, 255)).save(p, "WEBP")
    tax = json.dumps([{"sciName": f"Genus s{i}", "comName": f"B {i}", "speciesCode": f"c{i}x"} for i in range(5100)]
                     + [{"sciName": "Turdus migratorius", "comName": "American Robin", "speciesCode": "amerob"}]).encode()
    errors = iter([urllib.error.HTTPError(facts.EBIRD_NEARBY, 500, "boom", {}, None),
                   urllib.error.URLError("network down")])

    def nearby(url):
        return next(errors, urllib.error.URLError("down again"))

    calls = []

    def fake(url, headers=None, timeout=None, max_bytes=None):
        calls.append((url, dict(headers or {})))
        if url.startswith(facts.EBIRD_TAXONOMY):
            return tax
        if url.startswith(facts.EBIRD_NEARBY):
            raise nearby(url)
        raise urllib.error.HTTPError(url, 500, "upstream", {}, None)

    monkeypatch.setattr(facts, "http_get", fake)
    srv = serve.CollageServer(cfg, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    seen: list[str] = []
    try:
        def hit(path):
            try:
                status, headers, body = get(base + path)
            except urllib.error.HTTPError as exc:
                headers, body = exc.headers, exc.read()
            seen.append(str(headers))
            seen.append(body.decode("latin-1"))
        lay = json.loads(get(base + "/api/layout")[2])
        for path in ("/", "/collage.png", lay["png"], "/api/layout", "/api/recent",
                     "/api/species/Turdus%20migratorius", "/plate/turdus-migratorius.png",
                     "/fonts/OFL.txt", "/static/page.js", "/attribution", "/nope",
                     "/api/species/..%2Fx", "/api/species/Corvus%20corax"):
            hit(path)
        for _ in range(100):
            if not srv.facts._inflight:
                break
            time.sleep(0.05)
        # Second round after the background jobs finished (nearby failed twice by now).
        rec = facts.read_rec(cfg.facts.dir, "nearby", "turdus-migratorius")
        if rec is not None:
            facts.write_rec(cfg.facts.dir, "nearby", "turdus-migratorius", "error",
                            frame.utcnow() - dt.timedelta(hours=2), None, "x")
        hit("/api/species/Turdus%20migratorius")
        for _ in range(100):
            if not srv.facts._inflight:
                break
            time.sleep(0.05)
        hit("/api/species/Turdus%20migratorius")
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen and all(SENTINEL not in s for s in seen)
    for r in caplog.records:
        assert SENTINEL not in r.getMessage() and SENTINEL not in (r.exc_text or "")
        assert SENTINEL not in str(r.args)
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert SENTINEL.encode() not in f.read_bytes(), f
    assert SENTINEL not in repr(cfg) and SENTINEL not in repr(cfg.facts)
    ebird = [(u, h) for u, h in calls if "ebird.org" in u]
    assert any(u.startswith(facts.EBIRD_NEARBY) for u, _ in ebird) and len(ebird) >= 3
    for u, h in calls:
        assert SENTINEL not in u
        assert (h.get("X-eBirdApiToken") == SENTINEL) == ("ebird.org" in u)
    assert "facts: nearby for Turdus migratorius failed: HTTPError 500" in caplog.text
    assert "facts: nearby for Turdus migratorius failed: URLError" in caplog.text


def test_new_settings_documented():
    root = Path(serve.__file__).parent
    env = (root / ".env.example").read_text()
    readme = (root / "README.md").read_text()
    for var in ("FACTS_FETCH", "FACTS_DIR", "EBIRD_API_KEY"):
        assert var in env and f"`{var}`" in readme, var
    line = [ln for ln in env.splitlines() if "EBIRD_API_KEY=" in ln]
    assert line and all(ln.split("EBIRD_API_KEY=", 1)[1] == "" for ln in line)
