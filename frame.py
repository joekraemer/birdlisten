"""Recent-species query, Fugleramme plate cache, and the Pillow collage.

Everything that produces pixels for the collage page lives here; serve.py
owns HTTP. Nothing in this module is reached unless SERVE_PORT is set, so the
capture loop never fetches artwork or renders anything.

Plates are from the Fugleramme project (CC BY-SA 4.0), fetched lazily one
species at a time from raw.githubusercontent.com at a pinned commit and cached
under ARTWORK_DIR. A species Fugleramme does not have gets a `<stem>.missing`
marker (retried daily) and a plain paper card in its place.
"""

from __future__ import annotations

import datetime as dt
import io
import itertools
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

log = logging.getLogger("frame")

FUGLERAMME_REPO = "https://github.com/arnegiacomo/fugleramme"
DEFAULT_ARTWORK_REF = "8e8b0034f069b4d3b021bc7195482c1fe7caf880"
RAW_BASE = "https://raw.githubusercontent.com/arnegiacomo/fugleramme/{ref}/assets/artwork/classic/"
USER_AGENT = "birdlisten/1.0 (https://github.com/joekraemer/birdlisten)"
MAX_FETCH_BYTES = 8_000_000  # per response; thumbnails are well under 1 MB
FETCH_TIMEOUT = 5            # seconds, per HTTP request
FETCH_BUDGET = 15            # seconds of plate fetching per render, total
MISSING_RETRY = dt.timedelta(hours=24)   # after a 404
ERROR_RETRY = dt.timedelta(hours=1)      # after a timeout / 5xx / bad image
PAPER = (244, 236, 216)      # page
INK = (52, 44, 34)           # names and title, a warm near-black
GREY = (124, 112, 94)        # species count, quiet text
RULE = (186, 170, 142)       # ornaments and the placeholder card's rules
CARD = (236, 226, 202)       # placeholder card, a shade darker than the page
MIN_CELL = 80                # px; below this we stop shrinking and drop species
MIN_FONT = 9                 # px; name_layout never goes smaller
CELL_RATIO = 1.32            # cell height / width: a square plate plus two name lines
MAX_CELL_FRAC = 0.4          # largest cell as a fraction of the short side (n = 1)
GUTTER_MAX = 0.45            # spare width widens gutters up to this x cell
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {False: "LibreBaskerville.ttf", True: "LibreBaskerville-Italic.ttf"}
META_FILES = ("ATTRIBUTION.md", "manifest.json")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)   # the one place "now" is made


# ----------------------------------------------------------------- query
@dataclass(frozen=True)
class Species:
    scientific_name: str
    common_name: str
    last_heard: str          # ISO-8601 UTC as stored, e.g. 2026-10-02T14:12:00+00:00
    count: int
    cameras: tuple[str, ...]
    first_ever: bool


def open_ro(db_path: Path) -> sqlite3.Connection:
    # mode=ro: the server can never write, so the loop's DB is safe by construction.
    # timeout=5: wait out the capture loop's short write transactions.
    # as_uri() percent-encodes '?', '#', '%' and spaces in the path.
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def recent_species(conn: sqlite3.Connection, now: dt.datetime, hours: int,
                   min_confidence: float = 0.0) -> list[Species]:
    """Species heard in the last `hours`, most recent first. `now` must be
    UTC-aware so `since` has the same fixed width and +00:00 suffix as every
    heard_at written by record(), which makes TEXT comparison chronological.
    Rows below `min_confidence` are ignored everywhere, first_ever included,
    so raising MIN_CONFIDENCE hides older low-confidence rows immediately."""
    since = (now - dt.timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT scientific_name, common_name, camera, heard_at FROM detections"
        " WHERE heard_at >= ? AND confidence >= ? ORDER BY heard_at DESC, id DESC",
        (since, min_confidence),
    ).fetchall()
    first_heard = dict(conn.execute(
        "SELECT scientific_name, MIN(heard_at) FROM detections"
        " WHERE confidence >= ? GROUP BY scientific_name",
        (min_confidence,),
    ).fetchall())

    order: list[str] = []
    common: dict[str, str] = {}
    last: dict[str, str] = {}
    count: dict[str, int] = {}
    cameras: dict[str, list[str]] = {}
    for sci, com, cam, heard in rows:
        if sci not in count:
            order.append(sci)
            common[sci] = com          # most recent row wins
            last[sci] = heard
            count[sci] = 0
            cameras[sci] = []
        count[sci] += 1
        if cam not in cameras[sci]:
            cameras[sci].append(cam)   # most recent camera first
    return [
        Species(sci, common[sci], last[sci], count[sci], tuple(cameras[sci]), first_heard[sci] >= since)
        for sci in order
    ]


# ----------------------------------------------------------------- artwork
def stem(scientific_name: str) -> str:
    """'Turdus  migratorius' -> 'turdus-migratorius'. Anything outside
    [a-z0-9-] is dropped so the stem is always a safe single path segment."""
    return re.sub(r"[^a-z0-9-]+", "", "-".join(scientific_name.lower().split()))


class NotFound(Exception):
    pass


def fetch_url(url: str, timeout: float = FETCH_TIMEOUT) -> bytes:
    """The only network call in this module; tests monkeypatch it. Sends a
    descriptive User-Agent (Wikimedia asks for one) and never reads more than
    MAX_FETCH_BYTES, so a misrouted request cannot pull a full-size scan."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(MAX_FETCH_BYTES + 1)
        if len(data) > MAX_FETCH_BYTES:
            raise ValueError("response too large")
        return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NotFound(url) from exc
        raise


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a sibling .tmp and os.replace so a concurrent reader never
    sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_marker(m: Path, reason: str, now: dt.datetime) -> None:
    """The marker's mtime is the attempt time, taken from `now` so the retry
    check in ensure_plate has a single clock that tests can drive."""
    _write_atomic(m, (reason + "\n").encode())
    ts = now.timestamp()
    os.utime(m, (ts, ts))


_meta_lock = threading.Lock()
_meta_tried: dict[Path, float] = {}      # artwork dir -> time.monotonic() of last failed ensure_meta


@dataclass(frozen=True)
class Artwork:
    dir: Path
    ref: str = DEFAULT_ARTWORK_REF

    def url(self, rel: str) -> str:
        return RAW_BASE.format(ref=self.ref) + rel

    def plate_path(self, scientific_name: str) -> Path:
        return self.dir / "birds" / f"{stem(scientific_name)}.webp"

    def marker_path(self, scientific_name: str) -> Path:
        return self.dir / "birds" / f"{stem(scientific_name)}.missing"

    def has_plate(self, scientific_name: str) -> bool:
        """Disk only, never fetches; what /api/recent reports."""
        return bool(stem(scientific_name)) and self.plate_path(scientific_name).exists()

    def ensure_plate(self, scientific_name: str, now: dt.datetime | None = None) -> Path | None:
        """Return the cached plate, fetching it on first need. None when
        Fugleramme has no plate (marker, retried after MISSING_RETRY) or the
        fetch failed (marker, retried after ERROR_RETRY)."""
        now = now or utcnow()
        s = stem(scientific_name)
        if not s:
            return None   # otherwise every such name would share birds/.webp
        p = self.plate_path(scientific_name)
        if p.exists():
            return p
        m = self.marker_path(scientific_name)
        if m.exists():
            try:
                first = (m.read_text(errors="replace").splitlines() or [""])[0]
            except OSError:
                first = ""
            interval = MISSING_RETRY if first.startswith("404") else ERROR_RETRY
            elapsed = now - dt.datetime.fromtimestamp(m.stat().st_mtime, dt.timezone.utc)
            if elapsed < interval:
                return None
        try:
            data = fetch_url(self.url(f"birds/{s}.webp"))
            Image.open(io.BytesIO(data)).load()   # bad bytes are an error, not a plate
        except NotFound:
            _write_marker(m, "404", now)
            log.info("no plate for %s", scientific_name)
            return None
        except Exception as exc:  # noqa: BLE001
            _write_marker(m, f"{type(exc).__name__}: {exc}"[:200], now)
            log.warning("plate fetch failed for %s: %s", scientific_name, exc)
            return None
        _write_atomic(p, data)
        if m.exists():
            m.unlink()
        return p

    def ensure_meta(self, deadline: float | None = None) -> None:
        """Fetch ATTRIBUTION.md and manifest.json once. A failed attempt is
        remembered per artwork dir for ERROR_RETRY; a deadline skip is not a
        failure and is not remembered. Fetches run under _meta_lock, so a
        concurrent /attribution may wait up to ~2 x FETCH_TIMEOUT; accepted."""
        if all((self.dir / n).exists() for n in META_FILES):
            return
        if deadline is not None and time.monotonic() >= deadline:
            return
        with _meta_lock:
            last = _meta_tried.get(self.dir)
            if last is not None and time.monotonic() - last < ERROR_RETRY.total_seconds():
                return
            for name in META_FILES:
                p = self.dir / name
                if p.exists():
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    return
                try:
                    _write_atomic(p, fetch_url(self.url(name)))
                except Exception as exc:  # noqa: BLE001
                    _meta_tried[self.dir] = time.monotonic()
                    log.warning("artwork %s fetch failed: %s", name, exc)
                    return
            _meta_tried.pop(self.dir, None)

    def attribution_text(self) -> str | None:
        p = self.dir / "ATTRIBUTION.md"
        if not p.exists():
            return None
        return p.read_text(errors="replace")


# ----------------------------------------------------------------- layout
@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


def _metrics(width: int, height: int) -> tuple[float, int, int, int]:
    """(unit, margin, gap, top): everything scales with the shorter side.
    `top` is the header band (title, subtitle, rule) above the grid."""
    unit = min(width, height) / 100
    return unit, round(3.5 * unit), round(2 * unit), round(10 * unit)


def pack(n: int, width: int, height: int, top: int, margin: int, gap: int) -> tuple[list[Box], int]:
    """Uniform 1:CELL_RATIO portrait cells in a grid. Picks the column count
    that gives the largest cell (capped at MAX_CELL_FRAC of the short side),
    shrinks to MIN_CELL, then drops the tail of the list (the oldest species,
    since callers pass most-recent-first).

    Spare width widens the gutters (up to GUTTER_MAX x cell) instead of
    piling up at the sides; spare height is split above and below the grid.
    Species are spread evenly over the rows (fuller rows first, row lengths
    differ by at most one) and every row is centred, so a short last row is
    one bird shorter rather than a ragged stub."""
    if n <= 0:
        return [], 0
    W = width - 2 * margin
    H = height - top - 2 * margin
    cap = round(MAX_CELL_FRAC * min(width, height))
    best_cell, best_cols = -1, 1
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = (W - (cols - 1) * gap) // cols
        cell_h = (H - (rows - 1) * gap) // rows
        cell = min(cell_w, int(cell_h / CELL_RATIO), cap)
        if cell > best_cell:
            best_cell, best_cols = cell, cols
    cell, cols = best_cell, best_cols
    if cell < MIN_CELL:
        cell = MIN_CELL
        cols = max(1, (W + gap) // (cell + gap))
        rows = max(1, (H + gap) // (round(CELL_RATIO * cell) + gap))
        shown = min(n, cols * rows)
    else:
        shown = n
    dropped = n - shown
    rows = math.ceil(shown / cols)
    base, extra = divmod(shown, rows)
    counts = [base + 1] * extra + [base] * (rows - extra)
    widest = counts[0]
    gx = gap
    if widest > 1:
        spare = (W - widest * cell) // (widest - 1)
        gx = max(gap, min(spare, round(GUTTER_MAX * cell)))
    bh = round(CELL_RATIO * cell)
    grid_h = rows * bh + (rows - 1) * gap
    y = top + margin + max(0, H - grid_h) * 2 // 5   # a little above centre reads as centred
    boxes: list[Box] = []
    for in_row in counts:
        x0 = margin + (W - in_row * cell - (in_row - 1) * gx) // 2
        for i in range(in_row):
            boxes.append(Box(x0 + i * (cell + gx), y, cell, bh))
        y += bh + gap
    return boxes, dropped


def capacity(width: int, height: int) -> int:
    """Largest n that fits without dropping. Documentation/test helper."""
    _, margin, gap, top = _metrics(width, height)
    n = 0
    while pack(n + 1, width, height, top, margin, gap)[1] == 0:
        n += 1
    return n


@lru_cache(maxsize=128)
def _font(size: int, italic: bool = False) -> ImageFont.FreeTypeFont:
    """Libre Baskerville from fonts/ (SIL OFL, see fonts/SOURCE.txt). Falls
    back to Pillow's bundled sans (Aileron, scalable from 10.1 on) when the
    file is missing, so a checkout without fonts/ still renders."""
    try:
        return ImageFont.truetype(str(FONT_DIR / FONT_FILES[italic]), size)
    except OSError:
        log.warning("font %s not found; using Pillow's default", FONT_FILES[italic])
        return ImageFont.load_default(size=size)


def _ellipsise(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    while draw.textlength(text, font=font) > max_w and len(text) > 1:
        text = text[:-2].rstrip() + "…" if text.endswith("…") else text[:-1].rstrip() + "…"
    return text


def wrap_name(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str] | None:
    """One line if it fits, else the most balanced two-line split at a space
    or after a hyphen ('Collared-' / 'Dove'). None if no split fits."""
    if draw.textlength(text, font=font) <= max_w:
        return [text]
    splits = [(text[:i], text[i + 1:]) for i, c in enumerate(text) if c == " "]
    splits += [(text[:i + 1], text[i + 1:]) for i, c in enumerate(text) if c == "-" and 0 < i < len(text) - 1]
    best: tuple[float, list[str]] | None = None
    for a, b in splits:
        a, b = a.strip(), b.strip()
        if not a or not b:
            continue
        widest = max(draw.textlength(a, font=font), draw.textlength(b, font=font))
        if widest <= max_w and (best is None or widest < best[0]):
            best = (widest, [a, b])
    return best[1] if best else None


def name_layout(draw: ImageDraw.ImageDraw, names: list[str], max_w: int,
                size: int) -> tuple[ImageFont.FreeTypeFont, list[list[str]]]:
    """One font size for every name on the page. Starts at `size` and steps
    down only if some name cannot be set in two lines; at MIN_FONT the
    second line is ellipsised. Returns the font and each name's lines."""
    size = max(size, MIN_FONT)
    while True:
        font = _font(size)
        lines = [wrap_name(draw, n, font, max_w) for n in names]
        if all(lines) or size <= MIN_FONT:
            break
        size -= 1
    out: list[list[str]] = []
    for n, ls in zip(names, lines):
        if ls is None:   # MIN_FONT and still too long: first word(s) then ellipsis
            words = n.split()
            first = _ellipsise(draw, words[0], font, max_w)
            rest = " ".join(words[1:])
            ls = [first, _ellipsise(draw, rest, font, max_w)] if rest else [first]
        out.append(ls)
    return font, out


def window_phrase(hours: int) -> str:
    if hours == 1:
        return "the last hour"
    if hours % 24 == 0 and hours > 24:
        return f"the last {hours // 24} days"
    return f"the last {hours} hours"


def _ornament(draw: ImageDraw.ImageDraw, cx: int, cy: int, half: int, fill, width: int = 1) -> None:
    """A short engraved-style rule with a lozenge in the middle: ──◆──."""
    d = max(2, round(math.sqrt(half) / 1.8))   # grows slower than the rule
    draw.line([(cx - half, cy), (cx - 2 * d, cy)], fill=fill, width=width)
    draw.line([(cx + 2 * d, cy), (cx + half, cy)], fill=fill, width=width)
    draw.polygon([(cx - d, cy), (cx, cy - d), (cx + d, cy), (cx, cy + d)], fill=fill)


def _plate_card(draw: ImageDraw.ImageDraw, px: int, py: int, pw: int, ph: int) -> None:
    """Stand-in for a species Fugleramme has no plate for: a tinted paper
    card with a double rule and a lozenge, bottom-aligned like the plates.
    The name is set beneath it like every other bird."""
    ci = round(pw * 0.14)
    x0, y0, x1, y1 = px + ci, py + round(ph * 0.16), px + pw - ci - 1, py + ph - 1
    lw = max(1, round(pw / 160))
    draw.rectangle((x0, y0, x1, y1), fill=CARD, outline=RULE, width=lw)
    ii = max(3, round(pw * 0.03))
    draw.rectangle((x0 + ii, y0 + ii, x1 - ii, y1 - ii), outline=RULE, width=lw)
    _ornament(draw, (x0 + x1) // 2, (y0 + y1) // 2, round((x1 - x0) * 0.24), RULE, lw)


# ----------------------------------------------------------------- render
@dataclass(frozen=True)
class Rendered:
    png: bytes
    shown: int
    dropped: int
    quiet: bool              # True when the "all quiet" path was taken
    deferred: int            # plates not attempted because FETCH_BUDGET ran out


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def render(species: list[Species], art: Artwork, width: int, height: int, hours: int,
           now: dt.datetime | None = None) -> Rendered:
    """A field-guide style page: centred title and species count, then one
    plate per species with its common name beneath, all in one type size."""
    now = now or utcnow()
    deadline = time.monotonic() + FETCH_BUDGET
    art.ensure_meta(deadline)   # first, so the two small meta files win over plates
    unit, margin, gap, top = _metrics(width, height)
    img = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(img)
    cx = width // 2
    rule_w = max(1, round(unit / 8))

    if not species:
        font = _font(max(MIN_FONT, round(3.6 * unit)), italic=True)
        text = _ellipsise(draw, f"Nothing heard in {window_phrase(hours)}", font, width - 2 * margin)
        cy = height // 2
        draw.text((cx, cy), text, font=font, fill=GREY, anchor="ms")
        _ornament(draw, cx, cy + round(3 * unit), round(6 * unit), RULE, rule_w)
        return Rendered(_png(img), 0, 0, True, 0)

    boxes, dropped = pack(len(species), width, height, top, margin, gap)
    title_font = _font(max(MIN_FONT, round(3.4 * unit)))
    sub_font = _font(max(MIN_FONT, round(2.1 * unit)), italic=True)
    title = _ellipsise(draw, f"Heard in {window_phrase(hours)}", title_font, width - 2 * margin)
    sub = f"{len(species)} species" + (f", {dropped} not shown" if dropped else "")
    # Baselines inside the header band [0, top + margin): title, count, rule.
    draw.text((cx, round(6 * unit)), title, font=title_font, fill=INK, anchor="ms")
    draw.text((cx, round(9 * unit)), sub, font=sub_font, fill=GREY, anchor="ms")
    _ornament(draw, cx, round(11 * unit), round(5 * unit), RULE, rule_w)

    # Fetch (within budget) or look up each shown species' plate.
    deferred = 0
    paths: list[Path | None] = []
    for sp in species[:len(boxes)]:
        name = sp.scientific_name
        if not stem(name):
            paths.append(None)
        elif time.monotonic() < deadline:
            paths.append(art.ensure_plate(name, now))
        else:
            p = art.plate_path(name)
            if p.exists():
                paths.append(p)
            else:
                paths.append(None)
                if not art.marker_path(name).exists():
                    deferred += 1
    if deferred:
        log.info("fetch budget exhausted, %d plates deferred", deferred)

    cell = boxes[0].w   # pack makes every cell the same size
    inset = round(cell * 0.04)
    # Names may spill into 3/8 of the gutter on each side, leaving 1/4 of it
    # between neighbours, so fewer of them need a second line.
    row_gap = min((b.x - a.x - cell for a, b in itertools.pairwise(boxes) if b.y == a.y), default=0)
    label_w = cell + min(row_gap * 3 // 4, margin)   # and never past the page margin
    font, names = name_layout(draw, [sp.common_name for sp in species[:len(boxes)]], label_w,
                              min(round(cell / 10), round(2.3 * unit)))   # never rivals the title
    ascent, descent = font.getmetrics()
    line_h = round((ascent + descent) * 1.1)
    for sp, box, path, lines in zip(species, boxes, paths, names):
        px, py, pw, ph = box.x + inset, box.y + inset, cell - 2 * inset, cell - 2 * inset
        drawn = False
        if path is not None:
            try:
                with Image.open(path) as im:
                    im = ImageOps.contain(im.convert("RGBA"), (pw, ph))
                img.paste(im, (px + (pw - im.width) // 2, py + ph - im.height), im)
                drawn = True
            except Exception as exc:  # noqa: BLE001 -- corrupt cached file
                log.warning("bad cached plate %s: %s; deleting", path, exc)
                try:
                    path.unlink()
                except OSError:
                    pass
        if not drawn:
            _plate_card(draw, px, py, pw, ph)
        # Names hang from the same line in every cell; a wrapped name adds a
        # second line below rather than shifting the first.
        y = box.y + cell - inset + round(cell * 0.05) + ascent
        for line in lines:
            draw.text((box.x + cell // 2, y), line, font=font, fill=INK, anchor="ms")
            y += line_h

    return Rendered(_png(img), len(boxes), dropped, False, deferred)


# ----------------------------------------------------------------- cache
def cache_key(species: list[Species], art: Artwork) -> tuple:
    """Everything that changes a pixel: species order, names, and whether a
    plate is on disk. Cameras, times and first_ever are JSON-only."""
    return tuple(
        (s.scientific_name, s.common_name, art.has_plate(s.scientific_name))
        for s in species
    )


@dataclass
class RenderCache:
    art: Artwork
    max_entries: int = 4
    renders: int = 0                                  # test observability
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _entries: dict[tuple[int, int, int], tuple[tuple, bytes]] = field(default_factory=dict)

    def get(self, species: list[Species], width: int, height: int, hours: int,
            now: dt.datetime | None = None) -> bytes:
        with self._lock:
            k = (width, height, hours)
            key = cache_key(species, self.art)
            hit = self._entries.get(k)
            if hit and hit[0] == key:
                return hit[1]
            out = render(species, self.art, width, height, hours, now=now)
            self.renders += 1
            log.info("rendered %sx%s: %s species, %s dropped, %s deferred, quiet=%s",
                     width, height, out.shown, out.dropped, out.deferred, out.quiet)
            if k not in self._entries and len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))   # oldest inserted
            self._entries[k] = (key, out.png)
            return out.png
