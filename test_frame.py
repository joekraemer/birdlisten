"""Unit tests for frame.py: query, plate cache, packer, renderer, render cache.
No network: an autouse fixture makes every fetch raise NotFound.
Run: uv run --group dev pytest -q   (arm64 macOS: see README "Tests")"""

from __future__ import annotations

import datetime as dt
import io
import itertools
import json
import time
import urllib.error
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

import birdlisten as bl
import frame

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 1, 1, tzinfo=UTC)
REAL_FETCH_URL = frame.fetch_url     # captured before no_network replaces it


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test starts with fetch_url refusing. Tests that need a success
    or a non-404 failure monkeypatch frame.fetch_url again themselves."""
    def refuse(url, timeout=None):
        raise frame.NotFound(url)
    monkeypatch.setattr(frame, "fetch_url", refuse)
    frame._meta_tried.clear()


def sp(sci="Turdus migratorius", com="American Robin", heard="2026-10-02T14:12:00+00:00",
       cams=("back",), first=False, count=1) -> frame.Species:
    return frame.Species(sci, com, heard, count, tuple(cams), first)


def many(n: int) -> list[frame.Species]:
    return [sp(f"Genus species{i}", f"Bird {i}", first=(i % 3 == 0)) for i in range(n)]


def seed(tmp_path: Path, rows):
    """rows: (when, camera, common, scientific[, confidence=0.9])."""
    conn = bl.open_db(tmp_path)
    for when, cam, com, sci, *conf in rows:
        bl.record(conn, when, bl.Camera(cam, "rtsp://x"), bl.Detection(com, sci, (conf or [0.9])[0], 0, 3), None)
    conn.close()
    return tmp_path / "birdlisten.sqlite"


# ----------------------------------------------------------------- stem
@pytest.mark.parametrize("name,expected", [
    ("Turdus migratorius", "turdus-migratorius"),
    ("  TURDUS   migratorius ", "turdus-migratorius"),
    ("Larus sp.", "larus-sp"),
    ("???", ""),
])
def test_stem_normalises(name, expected, tmp_path: Path, monkeypatch):
    import re
    out = frame.stem(name)
    assert out == expected
    assert out == "" or re.fullmatch(r"[a-z0-9-]+", out)
    if not expected:
        calls = []
        monkeypatch.setattr(frame, "fetch_url", lambda url, timeout=None: calls.append(url))
        art = frame.Artwork(tmp_path / "art")
        assert art.has_plate(name) is False
        assert art.ensure_plate(name, now=T0) is None
        assert calls == [] and not (tmp_path / "art").exists()


# ----------------------------------------------------------------- query
def test_recent_species_window_count_cameras_order(tmp_path: Path):
    now = dt.datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    h = dt.timedelta(hours=1)
    db = seed(tmp_path, [
        (now - 30 * h, "back", "American Robin", "Turdus migratorius"),    # outside 24 h window
        (now - 5 * h, "back", "American Robin", "Turdus migratorius"),
        (now - 3 * h, "front", "Robin (old name)", "Turdus migratorius"),
        (now - 2 * h, "back", "American Robin", "Turdus migratorius"),
        (now - 4 * h, "back", "Varied Thrush", "Ixoreus naevius"),
    ])
    conn = frame.open_ro(db)
    try:
        out = frame.recent_species(conn, now, 24)
    finally:
        conn.close()
    assert [s.scientific_name for s in out] == ["Turdus migratorius", "Ixoreus naevius"]
    robin = out[0]
    assert robin.count == 3                      # the 30 h old row is excluded
    assert robin.cameras == ("back", "front")    # most recent camera first
    assert robin.common_name == "American Robin"  # from the newest row
    assert robin.last_heard == (now - 2 * h).isoformat(timespec="seconds")
    assert out[1].count == 1 and out[1].cameras == ("back",)


def test_recent_species_first_ever(tmp_path: Path):
    now = dt.datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    h = dt.timedelta(hours=1)
    db = seed(tmp_path, [
        (now - 30 * h, "back", "American Robin", "Turdus migratorius"),
        (now - 2 * h, "back", "American Robin", "Turdus migratorius"),
        (now - 1 * h, "back", "Varied Thrush", "Ixoreus naevius"),
    ])
    conn = frame.open_ro(db)
    try:
        out = {s.scientific_name: s.first_ever for s in frame.recent_species(conn, now, 24)}
    finally:
        conn.close()
    assert out == {"Turdus migratorius": False, "Ixoreus naevius": True}


def test_recent_species_min_confidence(tmp_path: Path):
    now = dt.datetime(2026, 10, 2, 15, 0, tzinfo=UTC)
    h = dt.timedelta(hours=1)
    db = seed(tmp_path, [
        (now - 30 * h, "back", "American Robin", "Turdus migratorius", 0.6),   # old, low: not "seen before"
        (now - 3 * h, "back", "American Robin", "Turdus migratorius", 0.95),
        (now - 2 * h, "front", "American Robin", "Turdus migratorius", 0.6),   # low: no count, no camera
        (now - 1 * h, "back", "Mallard", "Anas platyrhynchos", 0.6),           # only low rows: hidden
        (now - 1 * h, "back", "Varied Thrush", "Ixoreus naevius", 0.9),        # exactly at the threshold
    ])
    conn = frame.open_ro(db)
    try:
        out = frame.recent_species(conn, now, 24, 0.9)
        everything = frame.recent_species(conn, now, 24)
    finally:
        conn.close()
    assert [s.scientific_name for s in out] == ["Ixoreus naevius", "Turdus migratorius"]
    robin = out[1]
    assert robin.count == 1 and robin.cameras == ("back",)
    assert robin.last_heard == (now - 3 * h).isoformat(timespec="seconds")
    assert robin.first_ever is True          # the 30 h old row is below the threshold
    assert len(everything) == 3 and everything[2].first_ever is False   # default 0.0 keeps all rows


def test_open_ro_cannot_write(tmp_path: Path):
    db = seed(tmp_path, [])
    conn = frame.open_ro(db)
    with pytest.raises(frame.sqlite3.OperationalError):
        conn.execute("INSERT INTO notified(common_name,last_sent) VALUES('x','y')")
    conn.close()


# ----------------------------------------------------------------- packer
@pytest.mark.parametrize("n", [1, 2, 3, 7, 12, 30, 60, 200])
@pytest.mark.parametrize("size", [(200, 200), (800, 600), (1600, 1200), (1200, 1600)])
def test_pack_no_overlap_in_canvas(n, size):
    w, h = size
    _, margin, gap, top = frame._metrics(w, h)
    boxes, dropped = frame.pack(n, w, h, top, margin, gap)
    assert len(boxes) >= 1 and len(boxes) + dropped == n
    for b in boxes:
        assert 0 <= b.x and b.x + b.w <= w and top <= b.y and b.y + b.h <= h
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
    if n <= 30 and w >= 800 and h >= 600:
        assert dropped == 0
    if n <= 60 and w >= 1600 and h >= 1200:
        assert dropped == 0


def test_pack_drop_policy():
    _, margin, gap, top = frame._metrics(1600, 1200)
    boxes, dropped = frame.pack(200, 1600, 1200, top, margin, gap)
    assert (len(boxes), dropped) == (98, 102)
    _, margin, gap, top = frame._metrics(200, 200)
    boxes, dropped = frame.pack(30, 200, 200, top, margin, gap)
    assert (len(boxes), dropped) == (2, 28)
    assert frame.pack(0, 800, 600, top, margin, gap) == ([], 0)
    assert frame.capacity(800, 600) == 32        # README quotes these
    assert frame.capacity(1600, 1200) == 98


@pytest.mark.parametrize("n,rows", [(22, [6, 6, 5, 5]), (5, [3, 2]), (7, [4, 3])])
def test_pack_balances_rows_and_centres_them(n, rows):
    w, h = 1600, 1200
    _, margin, gap, top = frame._metrics(w, h)
    boxes, _ = frame.pack(n, w, h, top, margin, gap)
    by_row: dict[int, list[frame.Box]] = {}
    for b in boxes:
        by_row.setdefault(b.y, []).append(b)
    assert [len(r) for _, r in sorted(by_row.items())] == rows
    steps = set()
    for r in by_row.values():
        left, right = r[0].x, w - (r[-1].x + r[-1].w)
        assert abs(left - right) <= 1                                  # centred
        steps |= {b.x - a.x for a, b in itertools.pairwise(r)}
    assert len(steps) <= 1                                             # one gutter everywhere
    assert len({b.w for b in boxes}) == 1 and len({b.h for b in boxes}) == 1


# ----------------------------------------------------------------- renderer
@pytest.mark.parametrize("n", [0, 1, 35])
def test_render_sizes_and_quiet(n, tmp_path: Path):
    art = frame.Artwork(tmp_path / "art")
    out = frame.render(many(n), art, 1600, 1200, 24, now=T0)
    im = Image.open(io.BytesIO(out.png))
    assert im.format == "PNG" and im.size == (1600, 1200)
    assert out.quiet is (n == 0)
    assert out.shown == n and out.dropped == 0 and out.deferred == 0


def test_render_ignores_json_only_fields(tmp_path: Path):
    """No NEW badge, cameras or time on the page: those fields stay in
    /api/recent only, so changing them must not change a pixel."""
    art = frame.Artwork(tmp_path / "art")
    a = frame.render([sp(first=False)], art, 800, 600, 24, now=T0).png
    b = frame.render([sp(first=True, cams=("front", "back"), heard="2026-10-02T03:00:00+00:00", count=9)],
                     art, 800, 600, 24, now=T0).png
    assert a == b


def test_render_falls_back_without_font_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(frame, "FONT_DIR", tmp_path / "no-fonts")
    frame._font.cache_clear()
    try:
        out = frame.render(many(3), frame.Artwork(tmp_path / "art"), 800, 600, 24, now=T0)
        assert Image.open(io.BytesIO(out.png)).size == (800, 600) and out.shown == 3
    finally:
        frame._font.cache_clear()


def test_font_files_are_shipped():
    for name in frame.FONT_FILES.values():
        assert (frame.FONT_DIR / name).is_file()
    assert (frame.FONT_DIR / "OFL.txt").is_file()
    assert frame._font(20).getname()[0] == "Libre Baskerville"
    assert frame._font(20, italic=True).getname() == ("Libre Baskerville", "Italic")


def test_render_placeholder_when_no_plate(tmp_path: Path):
    art = frame.Artwork(tmp_path / "art")
    out = frame.render([sp()], art, 800, 600, 24, now=T0)
    assert Image.open(io.BytesIO(out.png)).size == (800, 600)
    assert art.has_plate("Turdus migratorius") is False
    marker = art.marker_path("Turdus migratorius")
    assert marker.exists() and marker.read_text().splitlines()[0] == "404"


def test_render_drops_corrupt_cached_plate(tmp_path: Path):
    art = frame.Artwork(tmp_path / "art")
    p = art.plate_path("Turdus migratorius")
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not a webp")
    out = frame.render([sp()], art, 800, 600, 24, now=T0)
    assert out.shown == 1 and not p.exists()      # deleted so the next render re-fetches


# ----------------------------------------------------------------- plate cache
def test_missing_marker_retry(tmp_path: Path, monkeypatch):
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        raise frame.NotFound(url)
    monkeypatch.setattr(frame, "fetch_url", fake)
    art = frame.Artwork(tmp_path / "art")
    name = "Ixoreus naevius"
    assert art.ensure_plate(name, now=T0) is None
    assert art.ensure_plate(name, now=T0 + dt.timedelta(hours=1)) is None
    assert len(calls) == 1
    assert art.ensure_plate(name, now=T0 + dt.timedelta(hours=25)) is None
    assert len(calls) == 2
    assert calls[0] == frame.RAW_BASE.format(ref=frame.DEFAULT_ARTWORK_REF) + "birds/ixoreus-naevius.webp"
    assert art.marker_path(name).stat().st_mtime == (T0 + dt.timedelta(hours=25)).timestamp()


def test_error_marker_retries_sooner(tmp_path: Path, monkeypatch):
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        raise OSError("timed out")
    monkeypatch.setattr(frame, "fetch_url", fake)
    art = frame.Artwork(tmp_path / "art")
    name = "Ixoreus naevius"
    assert art.ensure_plate(name, now=T0) is None
    assert art.marker_path(name).read_text().startswith("OSError: timed out")
    art.ensure_plate(name, now=T0 + dt.timedelta(minutes=30))
    assert len(calls) == 1
    art.ensure_plate(name, now=T0 + dt.timedelta(hours=2))
    assert len(calls) == 2


def tiny_webp() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (8, 12), (10, 20, 30, 255)).save(buf, "WEBP")
    return buf.getvalue()


def test_successful_fetch_is_cached(tmp_path: Path, monkeypatch):
    calls = []
    data = tiny_webp()

    def fake(url, timeout=None):
        calls.append(url)
        return data
    monkeypatch.setattr(frame, "fetch_url", fake)
    art = frame.Artwork(tmp_path / "art")
    p = art.ensure_plate("Turdus migratorius", now=T0)
    assert p == art.plate_path("Turdus migratorius") and p.read_bytes() == data
    assert art.ensure_plate("Turdus migratorius", now=T0) == p
    assert len(calls) == 1 and art.has_plate("Turdus migratorius") is True
    assert not art.marker_path("Turdus migratorius").exists()
    # And the renderer uses it without fetching again.
    out = frame.render([sp()], art, 800, 600, 24, now=T0)
    assert out.shown == 1 and sum("/birds/" in u for u in calls) == 1   # the rest are meta files


def test_bad_bytes_are_an_error_marker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(frame, "fetch_url", lambda url, timeout=None: b"<html>rate limited</html>")
    art = frame.Artwork(tmp_path / "art")
    assert art.ensure_plate("Turdus migratorius", now=T0) is None
    first = art.marker_path("Turdus migratorius").read_text().splitlines()[0]
    assert not first.startswith("404") and not art.plate_path("Turdus migratorius").exists()


def test_render_fetch_budget(tmp_path: Path, monkeypatch):
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        raise frame.NotFound(url)
    monkeypatch.setattr(frame, "fetch_url", fake)
    art = frame.Artwork(tmp_path / "art")
    species = many(5)

    monkeypatch.setattr(frame, "FETCH_BUDGET", 0)
    out = frame.render(species, art, 800, 600, 24, now=T0)
    assert calls == []                      # deadline already passed; ensure_meta skipped too
    assert out.deferred == 5 and out.shown == 5
    assert not list((art.dir / "birds").glob("*.missing")) if (art.dir / "birds").exists() else True

    monkeypatch.setattr(frame, "FETCH_BUDGET", 15)
    out = frame.render(species, art, 800, 600, 24, now=T0)
    # 5 plate fetches plus 1 meta fetch (ensure_meta stops at its first failure).
    assert sum("/birds/" in u for u in calls) == 5
    assert out.deferred == 0
    assert len(list((art.dir / "birds").glob("*.missing"))) == 5


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_fetch_url_user_agent_and_size_cap(monkeypatch):
    reqs = []

    def urlopen(req, timeout=None):
        reqs.append(req)
        return _FakeResp(body)
    monkeypatch.setattr(frame.urllib.request, "urlopen", urlopen)
    body = b"x" * 100
    assert REAL_FETCH_URL("https://example.org/a") == body
    ua = reqs[0].get_header("User-agent")
    assert ua == frame.USER_AGENT and "https://github.com/joekraemer/birdlisten" in ua
    body = b"x" * frame.MAX_FETCH_BYTES
    assert len(REAL_FETCH_URL("https://example.org/b")) == frame.MAX_FETCH_BYTES
    body = b"x" * (frame.MAX_FETCH_BYTES + 1)
    with pytest.raises(ValueError, match="response too large"):
        REAL_FETCH_URL("https://example.org/c")


def test_ensure_meta_retry_keyed_by_dir(tmp_path: Path, monkeypatch):
    a = frame.Artwork(tmp_path / "a")
    a.ensure_meta()                          # fixture: fails, remembered for dir a
    assert a.attribution_text() is None and a.dir in frame._meta_tried
    monkeypatch.setattr(frame, "fetch_url", lambda url, timeout=None: f"body of {url}".encode())
    b = frame.Artwork(tmp_path / "b")
    b.ensure_meta()
    assert (b.dir / "ATTRIBUTION.md").exists() and (b.dir / "manifest.json").exists()
    assert b.attribution_text().startswith("body of ")
    assert b.dir not in frame._meta_tried
    a.ensure_meta()                          # still within ERROR_RETRY: not retried
    assert a.attribution_text() is None


# ----------------------------------------------------------------- labels
def _draw():
    return frame.ImageDraw.Draw(Image.new("RGB", (10, 10)))


def test_wrap_name_one_line_or_balanced_two():
    draw, font = _draw(), frame._font(20)
    assert frame.wrap_name(draw, "Bushtit", font, 200) == ["Bushtit"]
    full = draw.textlength("Chestnut-backed Chickadee", font=font)
    lines = frame.wrap_name(draw, "Chestnut-backed Chickadee", font, int(full) - 1)
    assert lines == ["Chestnut-backed", "Chickadee"]
    w = draw.textlength("Collared-", font=font)
    assert frame.wrap_name(draw, "Collared-Dove", font, int(w) + 1) == ["Collared-", "Dove"]
    assert frame.wrap_name(draw, "Supercalifragilistic", font, 30) is None


def test_name_layout_one_size_for_all():
    draw = _draw()
    names = ["Bushtit", "Chestnut-backed Chickadee", "Northern Pygmy-Owl"]
    font, lines = frame.name_layout(draw, names, 200, 20)
    assert font.size == 20                                   # wrapping, not shrinking
    assert lines[0] == ["Bushtit"] and len(lines[1]) == 2
    for ls in lines:
        assert all(draw.textlength(t, font=font) <= 200 for t in ls)
    # A name that cannot be set in two lines shrinks every name, then ellipsises.
    font, lines = frame.name_layout(draw, ["Bushtit", "Supercalifragilistic Bird"], 40, 20)
    assert font.size == frame.MIN_FONT
    assert all(draw.textlength(t, font=font) <= 40 for ls in lines for t in ls)
    assert lines[1][0].endswith("…")


def test_window_phrase():
    assert frame.window_phrase(1) == "the last hour"
    assert frame.window_phrase(24) == "the last 24 hours"
    assert frame.window_phrase(6) == "the last 6 hours"
    assert frame.window_phrase(72) == "the last 3 days"


# ----------------------------------------------------------------- render cache
def test_render_cache_rerenders_only_on_change(tmp_path: Path):
    cache = frame.RenderCache(frame.Artwork(tmp_path / "art"))
    a = [sp(), sp("Ixoreus naevius", "Varied Thrush")]
    png1 = cache.get(a, 800, 600, 24, now=T0)
    png2 = cache.get(list(a), 800, 600, 24, now=T0)
    assert cache.renders == 1 and png1 == png2
    cache.get([sp(heard="2026-10-02T14:13:00+00:00", first=True, cams=("front",)), a[1]], 800, 600, 24, now=T0)
    assert cache.renders == 1              # time, cameras and first_ever are not drawn
    cache.get(a + [sp("Poecile rufescens", "Chestnut-backed Chickadee")], 800, 600, 24, now=T0)
    assert cache.renders == 2
    b = [sp(com="Robin"), a[1]]
    cache.get(b, 800, 600, 24, now=T0)
    assert cache.renders == 3
    cache.get(b, 400, 300, 24, now=T0)     # different size: its own entry
    assert cache.renders == 4
    cache.get(b, 800, 600, 24, now=T0)     # still cached
    assert cache.renders == 4


# ----------------------------------------------------------------- audubon table and cache
FIXTURES = Path(__file__).resolve().parent / "tests" / "fixtures" / "audubon"


def entry(plate=1, file="1 Wild Turkey.jpg", title="Wild Turkey", **kw):
    page = "https://commons.wikimedia.org/wiki/File:" + file.replace(" ", "_")
    return {"plate": plate, "title": title, "file": file, "page": page,
            "credit": "University of Pittsburgh", "credit_url": f"http://pitt.example/{plate}",
            "on_plate": 1, "via": ["commons", "wikidata"], **kw}


JAYS = "362 I. Yellow billed Magpie - 2. Stellers Jay - 3. Ultramarine Jay - 4. Clark's Crow.jpg"


def write_table(tmp_path: Path, species=None) -> Path:
    species = species if species is not None else {
        "Meleagris gallopavo": entry(),
        "Cyanocitta stelleri": entry(362, JAYS, "Jays"),
        "Aphelocoma californica": entry(362, JAYS, "Jays"),
    }
    p = tmp_path / "audubon.json"
    p.write_text(json.dumps({"edition": "havell", "generated": "2026-10-03", "species": species}))
    return p


def audubon(tmp_path: Path, species=None) -> frame.Audubon:
    a = frame.Audubon.load(tmp_path / "artwork" / "audubon", write_table(tmp_path, species))
    assert a is not None
    return a


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_audubon_lookup_and_validation(tmp_path: Path, caplog):
    species = {
        "Meleagris gallopavo": entry(),
        "Bad plate": entry(plate=436),
        "Bool plate": entry(plate=True),
        "No file": entry(file=""),
        "Bad page": {**entry(), "page": "https://evil.example/x"},
        "Not a dict": [1, 2],
        "???": entry(),
    }
    with caplog.at_level("WARNING", logger="frame"):
        a = audubon(tmp_path, species)
    assert a.entry("meleagris  Gallopavo")["plate"] == 1
    assert len(a.table) == 1 and a.edition == "havell"
    for bad in ("Bad plate", "Bool plate", "No file", "Bad page", "Not a dict"):
        assert a.entry(bad) is None and repr(bad) in caplog.text
    assert a.vignette_path("Meleagris gallopavo") == a.dir / "v1" / "meleagris-gallopavo.webp"


def test_audubon_load_missing_or_corrupt(tmp_path: Path, caplog):
    with caplog.at_level("ERROR", logger="frame"):
        assert frame.Audubon.load(tmp_path, tmp_path / "nope.json") is None
        (tmp_path / "bad.json").write_text("{not json")
        assert frame.Audubon.load(tmp_path, tmp_path / "bad.json") is None
        (tmp_path / "nospecies.json").write_text('{"edition": "havell"}')
        assert frame.Audubon.load(tmp_path, tmp_path / "nospecies.json") is None
    assert caplog.text.count("Audubon plates off") == 3


def test_audubon_duplicate_stem_later_wins(tmp_path: Path, caplog):
    with caplog.at_level("WARNING", logger="frame"):
        a = audubon(tmp_path, {"Meleagris gallopavo": entry(1), "Meleagris  gallopavo": entry(6, "6 Hen.jpg")})
    assert a.entry("Meleagris gallopavo")["plate"] == 6 and "duplicate" in caplog.text


def test_audubon_no_entry_no_fetch_no_marker(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(frame, "fetch_url", lambda url, timeout=None: calls.append(url))
    a = audubon(tmp_path)
    assert a.ensure("Turdus migratorius", T0) is None
    assert a.ensure("???", T0) is None
    assert calls == [] and not a.dir.exists()


def test_audubon_success_cached(tmp_path: Path, monkeypatch):
    calls = []
    data = (FIXTURES / "8.jpg").read_bytes()

    def fake(url, timeout=None):
        calls.append(url)
        return data
    monkeypatch.setattr(frame, "fetch_url", fake)
    a = audubon(tmp_path)
    m = a.marker_path("Meleagris gallopavo")
    m.parent.mkdir(parents=True)
    frame._write_marker(m, "OSError: old", T0 - dt.timedelta(hours=2))
    p = a.ensure("Meleagris gallopavo", T0)
    assert p == a.vignette_path("Meleagris gallopavo") and a.has_art("Meleagris gallopavo")
    assert not m.exists()
    with Image.open(p) as im:
        assert im.format == "WEBP" and max(im.size) <= frame.VIGNETTE_MAX
    assert a.ensure("Meleagris gallopavo", T0) == p and len(calls) == 1
    assert calls[0] == "https://commons.wikimedia.org/wiki/Special:FilePath/1_Wild_Turkey.jpg?width=960"


@pytest.mark.parametrize("exc,first,retry_h", [
    (frame.NotFound("x"), "404", 24),
    (urllib.error.HTTPError("u", 500, "boom", {}, None), "HTTPError", 1),
    (OSError("timed out"), "OSError: timed out", 1),
])
def test_audubon_markers(tmp_path: Path, monkeypatch, exc, first, retry_h):
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        raise exc
    monkeypatch.setattr(frame, "fetch_url", fake)
    a = audubon(tmp_path)
    name = "Meleagris gallopavo"
    assert a.ensure(name, T0) is None
    assert a.marker_path(name).read_text().startswith(first)
    assert a.ensure(name, T0 + dt.timedelta(hours=retry_h) - dt.timedelta(minutes=1)) is None
    assert len(calls) == 1
    assert a.ensure(name, T0 + dt.timedelta(hours=retry_h, minutes=1)) is None
    assert len(calls) == 2 and not a.has_art(name)


def test_audubon_vignette_error_waits_a_day(tmp_path: Path, monkeypatch):
    calls = []

    def fake(url, timeout=None):
        calls.append(url)
        return jpeg(Image.new("RGB", (96, 138), (240, 229, 200)))

    def bad(img, sci=""):
        raise frame.VignetteError("no picture")
    monkeypatch.setattr(frame, "fetch_url", fake)
    monkeypatch.setattr(frame, "vignette", bad)
    a = audubon(tmp_path)
    name = "Meleagris gallopavo"
    assert a.ensure(name, T0) is None
    assert a.marker_path(name).read_text().startswith("vignette: no picture")
    assert a.ensure(name, T0 + dt.timedelta(hours=23)) is None and len(calls) == 1
    assert a.ensure(name, T0 + dt.timedelta(hours=25)) is None and len(calls) == 2


def test_audubon_refuses_non_thumbnail(tmp_path: Path, monkeypatch):
    calls = []
    big = jpeg(Image.new("RGB", (3000, 3000), (240, 229, 200)))

    def fake(url, timeout=None):
        calls.append(url)
        return big
    monkeypatch.setattr(frame, "fetch_url", fake)
    a = audubon(tmp_path)
    assert a.ensure("Cyanocitta stelleri", T0) is None
    first = a.marker_path("Cyanocitta stelleri").read_text()
    assert first.startswith("ValueError: not a thumbnail")
    url = calls[0]
    assert "Special:FilePath/" in url and url.endswith("?width=960")
    assert "362_I._Yellow_billed_Magpie_-_2._Stellers_Jay_-_3._Ultramarine_Jay_-_4._Clark%27s_Crow.jpg" in url


def test_audubon_cached_plates_distinct(tmp_path: Path):
    a = audubon(tmp_path)
    assert a.cached_plates() == []
    for sci in ("Cyanocitta stelleri", "Aphelocoma californica", "Meleagris gallopavo"):
        p = a.vignette_path(sci)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    plates = a.cached_plates()
    assert [p["plate"] for p in plates] == [1, 362]
    assert set(plates[0]) == {"plate", "title", "page", "credit", "credit_url"}


# ----------------------------------------------------------------- vignette
SHEET_PAPER = (240, 229, 200)
DARK = (70, 55, 40)


def _glyphs(d: ImageDraw.ImageDraw, x, y, n, w=8, h=12, gap=3, stroke=2, fill=DARK) -> list[tuple]:
    out = []
    for i in range(n):
        x0 = x + i * (w + gap)
        d.rectangle((x0, y, x0 + w - 1, y + h - 1), outline=fill, width=stroke)
        out.append((x0, y, x0 + w, y + h))
    return out


def synthetic_sheet(w=960, h=1380, figures=True, title_y=None):
    """A Havell-like sheet: white scanner surround, yellowish paper with a
    faint plate mark, plate number and heading at the top, credit lines and a
    three-line title at the bottom. Returns (image, regions)."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    s = round(0.03 * w)
    d.rectangle((s, s, w - s - 1, h - s - 1), fill=SHEET_PAPER)
    faint = tuple(v - 15 for v in SHEET_PAPER)
    d.rectangle((s + 30, s + 30, w - s - 31, h - s - 31), outline=faint, width=2)
    reg = {"text": []}
    reg["text"].append(_glyphs(d, w - 200, round(0.07 * h), 5, h=10))          # "No. 1"
    reg["text"].append(_glyphs(d, w // 2 - 40, round(0.07 * h), 7, h=10))      # "PLATE I"
    title_y = title_y if title_y is not None else round(0.83 * h)
    reg["credit"] = [_glyphs(d, 100, title_y - 30, 15, h=6), _glyphs(d, w - 270, title_y - 30, 15, h=6)]
    reg["title"] = [g for i in range(3) for g in _glyphs(d, w // 2 - 150 + 30 * i, title_y + 20 * i, 25 - 5 * i)]
    reg["title_top"] = title_y
    if figures:
        cx, cy = w // 2, round(0.43 * h)
        d.ellipse((cx - 200, cy - 260, cx + 200, cy + 260), fill=(90, 70, 50))
        d.ellipse((cx - 60, cy - 330, cx + 140, cy - 180), fill=(110, 80, 60))       # head: irregular
        d.ellipse((cx + 220, cy + 140, cx + 340, cy + 260), fill=(60, 90, 120))      # side figure
        reg["figures"] = [(cx - 200, cy - 330, cx + 200, cy + 260), (cx + 220, cy + 140, cx + 340, cy + 260)]
    return img, reg


def _contains(outer, inner) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _disjoint(a, b) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _to_out(g: frame.VignetteGeometry, out: Image.Image):
    """Map a source-pixel box to the output image (crop, aspect canvas, resize)."""
    cw, ch = g.crop[2] - g.crop[0], g.crop[3] - g.crop[1]
    a = cw / ch
    canvas = ((round(0.75 * ch), ch) if a < 0.75 else (cw, round(cw / 1.33)) if a > 1.33 else (cw, ch))
    ox, oy = (canvas[0] - cw) // 2, (canvas[1] - ch) // 2
    k = out.width / canvas[0]

    def f(box):
        return tuple(round(v) for v in ((box[0] - g.crop[0] + ox) * k, (box[1] - g.crop[1] + oy) * k,
                                        (box[2] - g.crop[0] + ox) * k, (box[3] - g.crop[1] + oy) * k))
    return f


def _maxdiff(img: Image.Image, color) -> Image.Image:
    r, g, b = ImageChops.difference(img, Image.new("RGB", img.size, color)).split()
    return ImageChops.lighter(ImageChops.lighter(r, g), b)


def _card_frac(out: Image.Image, tol=6) -> float:
    return sum(_maxdiff(out, frame.CARD).histogram()[:tol + 1]) / (out.width * out.height)


def _mean(img: Image.Image, box) -> tuple[float, ...]:
    from PIL import ImageStat
    return tuple(ImageStat.Stat(img.crop(box)).mean)


def test_vignette_synthetic_base():
    img, reg = synthetic_sheet()
    g = frame._vignette_geometry(img)
    out = frame.vignette(img)
    for f in reg["figures"]:
        assert _contains(g.crop, f)
    for strip in reg["text"] + reg["credit"] + [reg["title"]]:
        for glyph in strip:
            assert _disjoint(g.crop, glyph)
    assert g.paper == SHEET_PAPER and g.paper_source == "margin"
    assert 0.75 <= out.width / out.height <= 1.33 and max(out.size) <= frame.VIGNETTE_MAX
    # Everything away from the figures is card-coloured.
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for f in reg["figures"]:
        md.rectangle((f[0] - 14, f[1] - 14, f[2] + 14, f[3] + 14), fill=255)
    to = _to_out(g, out)
    om = Image.new("L", out.size, 0)
    for f in reg["figures"]:
        om.paste(255, to((f[0] - 14, f[1] - 14, f[2] + 14, f[3] + 14)))
    diff = ImageChops.multiply(_maxdiff(out, frame.CARD), ImageChops.invert(om))
    assert diff.getextrema()[1] <= 4


def test_vignette_painted_plate():
    w, h = 960, 667
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    s = round(0.03 * w)
    d.rectangle((s, s, w - s - 1, h - s - 1), fill=SHEET_PAPER)
    sky = (round(0.05 * w), round(0.05 * h), w - round(0.05 * w), round(0.78 * h))
    d.rectangle(sky, fill=(100, 140, 190))
    blob = (380, 180, 580, 340)
    d.ellipse(blob, fill=(225, 225, 220))
    _glyphs(d, w // 2 - 120, round(0.86 * h), 20)
    g = frame._vignette_geometry(img)
    out = frame.vignette(img)
    assert 0.299 * g.paper[0] + 0.587 * g.paper[1] + 0.114 * g.paper[2] >= 200
    inner = (blob[0] + 40, blob[1] + 30, blob[2] - 40, blob[3] - 30)
    m = _mean(out, _to_out(g, out)(inner))
    assert max(abs(m[i] - frame.CARD[i]) for i in range(3)) > 15
    assert _card_frac(out) < 0.5


def test_vignette_caption_touching():
    img, reg = synthetic_sheet(title_y=872)
    d = ImageDraw.Draw(img)
    cx, bottom = 480, reg["figures"][0][3]
    d.line((cx, bottom - 5, cx, reg["title_top"] + 30), fill=DARK, width=3)       # branch into the title
    touching = [gl for gl in reg["title"] if gl[0] <= cx + 2 and gl[2] >= cx - 2]
    g = frame._vignette_geometry(img)
    out = frame.vignette(img)
    assert g.picture[3] < reg["title_top"]
    to = _to_out(g, out)
    dark = 0
    for gl in reg["title"]:
        if _disjoint(g.crop, gl):
            continue
        m = _mean(out, to(gl))
        if max(abs(m[i] - frame.CARD[i]) for i in range(3)) > 4:
            dark += 1
    assert dark <= 1 and touching
    assert any(not _disjoint(g.crop, gl) for gl in reg["title"])   # the title does reach the crop


def test_vignette_enclosed_pale_region_is_not_flattened():
    img, reg = synthetic_sheet(figures=False)
    d = ImageDraw.Draw(img)
    body = (240, 300, 720, 900)
    d.ellipse(body, fill=(80, 60, 45))
    belly = (body[0] + 40, body[1] + 40, body[2] - 40, body[3] - 40)
    pale = tuple(v - 25 for v in SHEET_PAPER)
    d.ellipse(belly, fill=pale)
    g = frame._vignette_geometry(img)
    out = frame.vignette(img)
    expected = [pale[i] * frame.CARD[i] / g.paper[i] for i in range(3)]
    cx, cy = (belly[0] + belly[2]) // 2, (belly[1] + belly[3]) // 2
    m = _mean(out, _to_out(g, out)((cx - 120, cy - 150, cx + 120, cy + 150)))
    assert all(abs(m[i] - expected[i]) <= 6 for i in range(3)), (m, expected)


def test_vignette_margin_stain():
    img, reg = synthetic_sheet()
    d = ImageDraw.Draw(img)
    s = round(0.03 * 960)
    blotch = (s, 1000, s + 90, 1090)                         # crosses the 6 % band from the sheet edge
    d.rectangle(blotch, fill=tuple(v - 60 for v in SHEET_PAPER))
    faint = (700, 150, 900, 230)                              # outer margin, outside the padded picture
    d.rectangle(faint, fill=tuple(v - 25 for v in SHEET_PAPER))
    g = frame._vignette_geometry(img)
    out = frame.vignette(img)
    assert _disjoint(g.crop, faint)
    if not _disjoint(g.crop, blotch):
        assert _maxdiff(out.crop(_to_out(g, out)(blotch)), frame.CARD).getextrema()[1] <= 6
    for f in reg["figures"]:
        assert _contains(g.crop, f)


def test_vignette_blank_sheet_raises():
    img, _ = synthetic_sheet(figures=False)
    with pytest.raises(frame.VignetteError):
        frame.vignette(img)
    with pytest.raises(frame.VignetteError):
        frame.vignette(Image.new("RGB", (480, 690), SHEET_PAPER))


@pytest.mark.parametrize("plate,box", [
    (8, (316, 412, 644, 996)),
    (362, (152, 296, 844, 1064)),
    (376, (48, 52, 908, 616)),
])
def test_vignette_real_fixtures(plate, box):
    with Image.open(FIXTURES / f"{plate}.jpg") as im:
        t = time.monotonic()
        g = frame._vignette_geometry(im)
        out = frame.vignette(im)
        elapsed = time.monotonic() - t
    w, h = g.size
    assert w == 960
    tol = (0.05 * w, 0.05 * h, 0.05 * w, 0.05 * h)
    assert all(abs(g.picture[i] - box[i]) <= tol[i] for i in range(4)), g.picture
    assert elapsed < 2
    if plate == 376:
        assert _card_frac(out) < 0.5
    if plate == 8:
        assert all(abs(g.paper[i] - (244, 229, 198)[i]) <= 6 for i in range(3)), g.paper


# ----------------------------------------------------------------- card frames
# sha256 of _plate_card at plate widths 60, 173, 400, recorded before the
# _card_frame refactor: the placeholder must not change by a pixel.
PLATE_CARD_SHA = {
    60: "5e528c9c4cb2f236b39fb93698802e65af1bb8cc101f7316b085010657d9fa3b",
    173: "b99c6744b971b9e476e93d73a8520cba9bb985cb6f6233002789547e0b822bdb",
    400: "248d722477ea9c54ab362582da1470b0ef642354c24d64d40bb914ef300024d9",
}


@pytest.mark.parametrize("pw", sorted(PLATE_CARD_SHA))
def test_plate_card_pixels_unchanged(pw):
    import hashlib
    im = Image.new("RGB", (pw + 20, pw + 20), frame.PAPER)
    frame._plate_card(ImageDraw.Draw(im), 10, 10, pw, pw)
    assert hashlib.sha256(im.tobytes()).hexdigest() == PLATE_CARD_SHA[pw]


@pytest.mark.parametrize("pw", [60, 173, 400])
@pytest.mark.parametrize("aspect", [0.75, 1.0, 1.33])
def test_vignette_frame_geometry(pw, aspect):
    px, py = 10, 20
    c0, c1, c2, c3 = frame._placeholder_box(px, py, pw, pw)
    x0, y0, x1, y1 = frame._vignette_frame_box(px, py, pw, pw, aspect)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    assert w * h <= (c2 - c0 + 1) * (c3 - c1 + 1)
    assert h <= c3 - c1 + 1 and w <= pw and x0 >= px and x1 <= px + pw - 1
    assert y1 == c3                                       # bottom-aligned with the placeholder
    assert abs((x0 - px) - (px + pw - 1 - x1)) <= 1       # centred
    assert abs(w / h - aspect) < 0.05
    if aspect == 0.75:
        assert h >= (c3 - c1 + 1) - 1                      # portrait: the placeholder's height
    if aspect == 1.33:
        assert w > c2 - c0 + 1                             # landscape: wider than the placeholder


def test_vignette_card_draws_and_drops_corrupt(tmp_path: Path):
    p = tmp_path / "v.webp"
    Image.new("RGB", (300, 400), (20, 120, 40)).save(p, "WEBP")
    im = Image.new("RGB", (220, 220), frame.PAPER)
    assert frame._vignette_card(im, ImageDraw.Draw(im), p, 10, 10, 200, 200) is True
    x0, y0, x1, y1 = frame._vignette_frame_box(10, 10, 200, 200, 0.75)
    assert im.getpixel((x0, y1)) == frame.RULE                       # outer rule
    assert im.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))[1] > 100    # the picture, no lozenge
    p.write_bytes(b"junk")
    assert frame._vignette_card(im, ImageDraw.Draw(im), p, 10, 10, 200, 200) is False
    assert not p.exists()


# ----------------------------------------------------------------- audubon in the renderer
def _fixture_fetch(calls):
    """Fugleramme 404s, Commons serves the plate 8 fixture, meta files 404."""
    data = (FIXTURES / "8.jpg").read_bytes()

    def fake(url, timeout=None):
        calls.append(url)
        if "Special:FilePath" in url:
            return data
        raise frame.NotFound(url)
    return fake


def test_priority_fugleramme_then_audubon_then_card(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(frame, "fetch_url", _fixture_fetch(calls))
    aud = audubon(tmp_path)
    art = frame.Artwork(tmp_path / "artwork", audubon=aud)
    drawn = []
    real_card, real_vig = frame._plate_card, frame._vignette_card
    monkeypatch.setattr(frame, "_plate_card", lambda *a: (drawn.append("card"), real_card(*a))[1])
    monkeypatch.setattr(frame, "_vignette_card", lambda *a: (drawn.append("vignette"), real_vig(*a))[1])

    # 1. A Fugleramme plate on disk beats an Audubon entry: no Commons fetch.
    p = art.plate_path("Meleagris gallopavo")
    p.parent.mkdir(parents=True)
    p.write_bytes(tiny_webp())
    frame.render([sp("Meleagris gallopavo", "Wild Turkey")], art, 800, 600, 24, now=T0)
    assert art.art_kind("Meleagris gallopavo") == "fugleramme" and drawn == []
    assert not any("Special:FilePath" in u for u in calls)

    # 2. Fugleramme 404 + an Audubon entry: the vignette.
    out = frame.render([sp("Cyanocitta stelleri", "Steller's Jay")], art, 800, 600, 24, now=T0)
    assert art.art_kind("Cyanocitta stelleri") == "audubon" and drawn == ["vignette"]
    assert art.marker_path("Cyanocitta stelleri").read_text().startswith("404")
    assert out.deferred == 0 and art.has_art("Cyanocitta stelleri")

    # 3. Neither source: the placeholder.
    drawn.clear()
    out = frame.render([sp("Turdus migratorius", "American Robin")], art, 800, 600, 24, now=T0)
    assert art.art_kind("Turdus migratorius") is None and drawn == ["card"] and out.deferred == 0


def test_corrupt_vignette_falls_back_to_card(tmp_path: Path):
    aud = audubon(tmp_path)
    art = frame.Artwork(tmp_path / "artwork", audubon=aud)
    v = aud.vignette_path("Meleagris gallopavo")
    v.parent.mkdir(parents=True)
    v.write_bytes(b"junk")
    out = frame.render([sp("Meleagris gallopavo", "Wild Turkey")], art, 800, 600, 24, now=T0)
    assert out.shown == 1 and not v.exists()


def test_cache_rerenders_when_vignette_then_cutout_appear(tmp_path: Path, monkeypatch):
    aud = audubon(tmp_path)
    art = frame.Artwork(tmp_path / "artwork", audubon=aud)
    cache = frame.RenderCache(art)
    species = [sp("Meleagris gallopavo", "Wild Turkey")]
    cache.get(species, 800, 600, 24, now=T0)
    cache.get(species, 800, 600, 24, now=T0)
    assert cache.renders == 1
    calls = []
    monkeypatch.setattr(frame, "fetch_url", _fixture_fetch(calls))
    aud.ensure("Meleagris gallopavo", T0 + dt.timedelta(days=2))     # vignette arrives
    cache.get(species, 800, 600, 24, now=T0)
    assert cache.renders == 2 and art.art_kind("Meleagris gallopavo") == "audubon"
    p = art.plate_path("Meleagris gallopavo")                          # later, a cut-out
    p.write_bytes(tiny_webp())
    cache.get(species, 800, 600, 24, now=T0)
    assert cache.renders == 3 and art.art_kind("Meleagris gallopavo") == "fugleramme"


def test_budget_defers_audubon(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(frame, "fetch_url", _fixture_fetch(calls))
    aud = audubon(tmp_path)
    art = frame.Artwork(tmp_path / "artwork", audubon=aud)
    species = [sp("Meleagris gallopavo", "Wild Turkey"), sp("Cyanocitta stelleri", "Steller's Jay")]
    # Fugleramme already said 404 for both, but Audubon was never tried.
    for s in species:
        m = art.marker_path(s.scientific_name)
        m.parent.mkdir(parents=True, exist_ok=True)
        frame._write_marker(m, "404", T0)
    monkeypatch.setattr(frame, "FETCH_BUDGET", 0)
    out = frame.render(species, art, 800, 600, 24, now=T0)
    assert calls == [] and out.deferred == 2
    monkeypatch.setattr(frame, "FETCH_BUDGET", 15)
    out = frame.render(species, art, 800, 600, 24, now=T0)
    assert out.deferred == 0 and sum("Special:FilePath" in u for u in calls) == 2


def test_audubon_off_renders_like_fugleramme_only(tmp_path: Path):
    """Artwork without Audubon, and with an Audubon table that has none of
    the species, draw identical pages: the fallback adds nothing when off."""
    species = many(5) + [sp("Meleagris gallopavo", "Wild Turkey")]
    off = frame.render(species, frame.Artwork(tmp_path / "a"), 800, 600, 24, now=T0).png
    empty = audubon(tmp_path, {})
    on = frame.render(species, frame.Artwork(tmp_path / "b", audubon=empty), 800, 600, 24, now=T0).png
    assert off == on


# ----------------------------------------------------------------- committed table
TARGET_PLATES = {
    "Zonotrichia albicollis": 8, "Aphelocoma californica": 362, "Psaltriparus minimus": 353,
    "Myadestes townsendi": 419, "Cygnus buccinator": 376, "Meleagris gallopavo": 1,
    "Tachycineta bicolor": 98, "Ixoreus naevius": 369, "Cyanocitta stelleri": 362,
}


def test_committed_audubon_table(tmp_path: Path):
    a = frame.Audubon.load(tmp_path, frame.AUDUBON_MAP)
    assert a is not None and a.edition == "havell"
    raw = json.loads(frame.AUDUBON_MAP.read_text(encoding="utf-8"))["species"]
    assert len(a.table) == len(raw) >= 400                 # nothing skipped as invalid
    for sci, plate in TARGET_PLATES.items():
        assert a.entry(sci)["plate"] == plate, sci
    files: dict[int, set[str]] = {}
    for e in raw.values():
        files.setdefault(e["plate"], set()).add(e["file"])
    assert all(len(f) == 1 for f in files.values())
    assert not [k for k in raw if k.split()[0] in ("Sciurus", "Tamias", "Tamiasciurus", "Canis")]
    assert raw.get("Anous minutus", {}).get("plate") != 275
    assert {e["on_plate"] for e in raw.values() if e["plate"] == 353} == {3}
