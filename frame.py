"""Recent-species query, Fugleramme plate cache, and the Pillow collage.

Everything that produces pixels for the collage page lives here; serve.py
owns HTTP. Nothing in this module is reached unless SERVE_PORT is set, so the
capture loop never fetches artwork or renders anything.

Plates are from the Fugleramme project (CC BY-SA 4.0), fetched lazily one
species at a time from raw.githubusercontent.com at a pinned commit and cached
under ARTWORK_DIR. A species Fugleramme does not have gets a `<stem>.missing`
marker (retried daily) and a dashed placeholder card.
"""

from __future__ import annotations

import datetime as dt
import io
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
FETCH_TIMEOUT = 5            # seconds, per HTTP request
FETCH_BUDGET = 15            # seconds of plate fetching per render, total
MISSING_RETRY = dt.timedelta(hours=24)   # after a 404
ERROR_RETRY = dt.timedelta(hours=1)      # after a timeout / 5xx / bad image
PAPER = (244, 236, 216)
INK = (40, 36, 30)
GREY = (120, 112, 100)
BADGE = (176, 48, 32)
MIN_CELL = 80                # px; below this we stop shrinking and drop species
MIN_FONT = 9                 # px; fit_text never goes smaller, labels never larger than cell/9
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


def recent_species(conn: sqlite3.Connection, now: dt.datetime, hours: int) -> list[Species]:
    """Species heard in the last `hours`, most recent first. `now` must be
    UTC-aware so `since` has the same fixed width and +00:00 suffix as every
    heard_at written by record(), which makes TEXT comparison chronological."""
    since = (now - dt.timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT scientific_name, common_name, camera, heard_at FROM detections"
        " WHERE heard_at >= ? ORDER BY heard_at DESC, id DESC",
        (since,),
    ).fetchall()
    first_heard = dict(conn.execute(
        "SELECT scientific_name, MIN(heard_at) FROM detections GROUP BY scientific_name"
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
    """The only network call in this module; tests monkeypatch it."""
    req = urllib.request.Request(url, headers={"User-Agent": "birdlisten"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
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
    """(unit, margin, gap, top): everything scales with the shorter side."""
    unit = min(width, height) / 100
    return unit, round(2.5 * unit), round(1.5 * unit), round(5 * unit)


def pack(n: int, width: int, height: int, top: int, margin: int, gap: int) -> tuple[list[Box], int]:
    """Uniform 1:1.3 portrait cells in a grid (a shelf packer with equal
    shelves). Shrinks to MIN_CELL, then drops the tail of the list (the
    oldest species, since callers pass most-recent-first). Rows are centred."""
    if n <= 0:
        return [], 0
    W = width - 2 * margin
    H = height - top - 2 * margin
    best_cell, best_cols = -1, 1
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = (W - (cols - 1) * gap) // cols
        cell_h = (H - (rows - 1) * gap) // rows
        cell = min(cell_w, int(cell_h / 1.3))
        if cell > best_cell:
            best_cell, best_cols = cell, cols
    cell, cols = best_cell, best_cols
    if cell < MIN_CELL:
        cell = MIN_CELL
        cols = max(1, (W + gap) // (cell + gap))
        rows = max(1, (H + gap) // (round(1.3 * cell) + gap))
        shown = min(n, cols * rows)
    else:
        shown = n
    dropped = n - shown
    bh = round(1.3 * cell)
    boxes: list[Box] = []
    y = top + margin
    for start in range(0, shown, cols):
        in_row = min(cols, shown - start)
        x0 = margin + (W - in_row * cell - (in_row - 1) * gap) // 2
        for i in range(in_row):
            boxes.append(Box(x0 + i * (cell + gap), y, cell, bh))
        y += bh + gap
    return boxes, dropped


def capacity(width: int, height: int) -> int:
    """Largest n that fits without dropping. Documentation/test helper."""
    _, margin, gap, top = _metrics(width, height)
    n = 0
    while pack(n + 1, width, height, top, margin, gap)[1] == 0:
        n += 1
    return n


def local_hhmm(iso: str, tz: dt.tzinfo | None = None) -> str:
    """'2026-10-02T14:12:00+00:00' -> '07:12' in tz (None = process local, TZ)."""
    return dt.datetime.fromisoformat(iso).astimezone(tz).strftime("%H:%M")


def label_lines(sp: Species, tz: dt.tzinfo | None = None) -> tuple[str, str]:
    return sp.common_name, f"{', '.join(sp.cameras)} · {local_hhmm(sp.last_heard, tz)}"


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled default (Aileron) is scalable from 10.1 on. It has
    # '·' and '…' but no accented Latin letters or the Hawaiian okina; those
    # render as boxes. Good enough for BirdNET's English common names.
    return ImageFont.load_default(size=size)


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_w: int, size: int,
             min_size: int = MIN_FONT) -> tuple[str, ImageFont.FreeTypeFont]:
    """Shrink the font 1 px at a time down to min_size, then ellipsise with
    '…' until the text fits in max_w. Text never leaves its cell."""
    size = max(size, min_size)
    font = _font(size)
    while draw.textlength(text, font=font) > max_w and size > min_size:
        size -= 1
        font = _font(size)
    while draw.textlength(text, font=font) > max_w and len(text) > 1:
        text = text[:-2].rstrip() + "…" if text.endswith("…") else text[:-1].rstrip() + "…"
    return text, font


def _centered(draw: ImageDraw.ImageDraw, text: str, font, cx: int, y: int, fill) -> None:
    draw.text((cx - draw.textlength(text, font=font) / 2, y), text, font=font, fill=fill)


def _dashed_rect(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, fill, dash=6, space=4) -> None:
    step = dash + space
    for x in range(x0, x1, step):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=fill)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=fill)
    for y in range(y0, y1, step):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=fill)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=fill)


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
           tz: dt.tzinfo | None = None, now: dt.datetime | None = None) -> Rendered:
    now = now or utcnow()
    deadline = time.monotonic() + FETCH_BUDGET
    art.ensure_meta(deadline)   # first, so the two small meta files win over plates
    unit, margin, gap, top = _metrics(width, height)
    img = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(img)

    if not species:
        font = _font(round(4 * unit))
        text = f"Nothing heard in the last {hours} h"
        tw = draw.textlength(text, font=font)
        draw.text(((width - tw) / 2, (height - font.size) / 2), text, font=font, fill=INK)
        return Rendered(_png(img), 0, 0, True, 0)

    boxes, dropped = pack(len(species), width, height, top, margin, gap)
    header = f"{len(species)} species · last {hours} h"
    if dropped:
        header += f" · +{dropped} more"
    draw.text((margin, round(unit)), header, font=_font(round(2.6 * unit)), fill=INK)

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

    for sp, box, path in zip(species, boxes, paths):
        cell = box.w
        inset = gap // 2
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
        size1 = max(MIN_FONT, round(cell / 9))
        size2 = max(MIN_FONT, round(cell / 11))
        if not drawn:
            _dashed_rect(draw, px, py, px + pw - 1, py + ph - 1, GREY)
            text, font = fit_text(draw, sp.scientific_name, pw - 4, size2)
            _centered(draw, text, font, px + pw // 2, py + (ph - font.size) // 2, GREY)
        line1, line2 = label_lines(sp, tz)
        text, font = fit_text(draw, line1, cell, size1)
        y1 = box.y + cell
        _centered(draw, text, font, box.x + cell // 2, y1, INK)
        text, font = fit_text(draw, line2, cell, size2)
        _centered(draw, text, font, box.x + cell // 2, y1 + size1 + 2, GREY)
        if sp.first_ever:
            font = _font(max(MIN_FONT, round(cell / 10)))
            tw = draw.textlength("NEW", font=font)
            pad = max(2, font.size // 3)
            x0, y0 = box.x, box.y
            draw.rounded_rectangle((x0, y0, x0 + tw + 2 * pad, y0 + font.size + 2 * pad),
                                   radius=pad + 1, fill=BADGE)
            draw.text((x0 + pad, y0 + pad), "NEW", font=font, fill=(255, 255, 255))

    return Rendered(_png(img), len(boxes), dropped, False, deferred)


# ----------------------------------------------------------------- cache
def cache_key(species: list[Species], art: Artwork, tz: dt.tzinfo | None = None) -> tuple:
    """Everything that changes a pixel: species order, names, cameras, the
    last-heard minute (local), the NEW badge, and whether a plate is on disk."""
    return tuple(
        (s.scientific_name, s.common_name, s.cameras, local_hhmm(s.last_heard, tz), s.first_ever,
         art.has_plate(s.scientific_name))
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
