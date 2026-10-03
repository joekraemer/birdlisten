"""Unit tests for frame.py: query, plate cache, packer, renderer, render cache.
No network: an autouse fixture makes every fetch raise NotFound.
Run: uv run --group dev pytest -q   (arm64 macOS: see README "Tests")"""

from __future__ import annotations

import datetime as dt
import io
import itertools
from pathlib import Path

import pytest
from PIL import Image

import birdlisten as bl
import frame

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 1, 1, tzinfo=UTC)


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
