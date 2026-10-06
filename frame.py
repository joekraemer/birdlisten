"""Recent-species query, plate caches (Fugleramme, Audubon), and the Pillow collage.

Everything that produces pixels for the collage page lives here; serve.py
owns HTTP. Nothing in this module is reached unless SERVE_PORT is set, so the
capture loop never fetches artwork or renders anything.

Cut-outs are from the Fugleramme project (CC BY-SA 4.0), fetched lazily one
species at a time from raw.githubusercontent.com at a pinned commit and cached
under ARTWORK_DIR. A species Fugleramme does not have gets a `<stem>.missing`
marker (retried daily). If audubon.json maps it to a Havell plate of Audubon's
*Birds of America* (public domain), a 960 px Wikimedia Commons thumbnail is
fetched, cropped to its picture and recoloured into a vignette, cached under
ARTWORK_DIR/audubon/v1/, and drawn in the placeholder's frame. With neither,
a plain paper card stands in. Priority: cut-out, vignette, card.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import itertools
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from urllib.parse import quote

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

import taxa

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
    # first_ever: no qualifying row before the window. One indexed probe per
    # species shown, instead of a MIN() over the whole table on every render.
    heard_before = "SELECT EXISTS(SELECT 1 FROM detections WHERE scientific_name = ? AND heard_at < ? AND confidence >= ?)"

    order: list[str] = []
    common: dict[str, str] = {}
    last: dict[str, str] = {}
    count: dict[str, int] = {}
    cameras: dict[str, list[str]] = {}
    for sci, com, cam, heard in rows:
        if not taxa.is_bird(sci):
            continue                   # kept in SQLite, never drawn (Dog, Engine, frogs)
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
        Species(sci, common[sci], last[sci], count[sci], tuple(cameras[sci]),
                not conn.execute(heard_before, (sci, since, min_confidence)).fetchone()[0])
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


def _marker_waiting(m: Path, now: dt.datetime) -> bool:
    """True while a miss marker says not to retry yet. A first line starting
    with '404' or 'vignette' (the source will not change) waits MISSING_RETRY;
    anything else (network, bad bytes) waits ERROR_RETRY."""
    if not m.exists():
        return False
    try:
        first = (m.read_text(errors="replace").splitlines() or [""])[0]
    except OSError:
        first = ""
    mtime = m.stat().st_mtime
    interval = MISSING_RETRY if first.startswith(("404", "vignette")) else ERROR_RETRY
    return now - dt.datetime.fromtimestamp(mtime, dt.timezone.utc) < interval


_meta_lock = threading.Lock()
_meta_tried: dict[Path, float] = {}      # artwork dir -> time.monotonic() of last failed ensure_meta


@dataclass(frozen=True)
class Artwork:
    dir: Path
    ref: str = DEFAULT_ARTWORK_REF
    audubon: Audubon | None = None      # fallback source; None = Fugleramme only

    def art_kind(self, scientific_name: str) -> str | None:
        """'fugleramme', 'audubon' or None, from disk only. Fugleramme wins."""
        if self.has_plate(scientific_name):
            return "fugleramme"
        if self.audubon is not None and self.audubon.has_art(scientific_name):
            return "audubon"
        return None

    def has_art(self, scientific_name: str) -> bool:
        """Any artwork on disk; what /api/recent reports as has_plate."""
        return self.art_kind(scientific_name) is not None

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
        if _marker_waiting(m, now):
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


# ----------------------------------------------------------------- audubon
AUDUBON_MAP = Path(__file__).resolve().parent / "audubon.json"
COMMONS_THUMB = "https://commons.wikimedia.org/wiki/Special:FilePath/{file}?width=960"
THUMB_MAX = 2000             # px; a larger response is not a thumbnail and is never decoded
VIGNETTE_VERSION = "v1"      # bump when vignette() changes; old caches are then ignored
VIGNETTE_MAX = 800           # px, longest side of the stored WebP
COMMONS_PREFIX = "https://commons.wikimedia.org/"


class VignetteError(Exception):
    """The scan has no usable picture; retried after MISSING_RETRY."""


def _valid_entry(e) -> bool:
    return (isinstance(e, dict)
            and isinstance(e.get("file"), str) and bool(e["file"])
            and type(e.get("plate")) is int and 1 <= e["plate"] <= 435
            and isinstance(e.get("page"), str) and e["page"].startswith(COMMONS_PREFIX))


@dataclass(frozen=True)
class Audubon:
    """Havell plates of Audubon's *Birds of America* from Wikimedia Commons,
    for species Fugleramme lacks. `table` is audubon.json's species map keyed
    by stem(); see tools/build_audubon_map.py."""
    dir: Path                                           # ARTWORK_DIR/audubon
    table: Mapping[str, dict] = field(repr=False, compare=False)
    edition: str = "havell"

    @classmethod
    def load(cls, dir: Path, path: Path | None = None) -> Audubon | None:
        """Read `path` (default AUDUBON_MAP). None (and an ERROR) when the
        table is missing or unreadable, so the server runs Fugleramme-only.
        Invalid entries are skipped."""
        path = AUDUBON_MAP if path is None else path
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("audubon table %s unreadable: %s; Audubon plates off", path, exc)
            return None
        species = doc.get("species") if isinstance(doc, dict) else None
        if not isinstance(species, dict):
            log.error("audubon table %s has no species map; Audubon plates off", path)
            return None
        table: dict[str, dict] = {}
        for key, e in species.items():
            s = stem(key) if isinstance(key, str) else ""
            if not s or not _valid_entry(e):
                log.warning("audubon table: skipping invalid entry %r", key)
                continue
            if s in table:
                log.warning("audubon table: duplicate key %r, the later one wins", key)
            table[s] = e
        edition = doc.get("edition") if isinstance(doc.get("edition"), str) else "havell"
        return cls(Path(dir), MappingProxyType(table), edition)

    def entry(self, sci: str) -> dict | None:
        s = stem(sci)
        return self.table.get(s) if s else None

    def vignette_path(self, sci: str) -> Path:
        return self.dir / VIGNETTE_VERSION / f"{stem(sci)}.webp"

    def marker_path(self, sci: str) -> Path:
        return self.dir / VIGNETTE_VERSION / f"{stem(sci)}.missing"

    def has_art(self, sci: str) -> bool:
        """Disk only, never fetches."""
        return bool(stem(sci)) and self.vignette_path(sci).exists()

    def thumb_url(self, e: dict) -> str:
        return COMMONS_THUMB.format(file=quote(e["file"].replace(" ", "_"), safe=""))

    def ensure(self, sci: str, now: dt.datetime | None = None) -> Path | None:
        """Return the cached vignette, fetching and processing the plate's
        thumbnail on first need. None when the species has no plate (no
        request, no marker) or the fetch or processing failed (marker)."""
        now = now or utcnow()
        e = self.entry(sci)
        if e is None:
            return None
        p = self.vignette_path(sci)
        if p.exists():
            return p
        m = self.marker_path(sci)
        if _marker_waiting(m, now):
            return None
        try:
            data = fetch_url(self.thumb_url(e))
            with Image.open(io.BytesIO(data)) as im:
                if max(im.size) > THUMB_MAX:
                    raise ValueError(f"not a thumbnail: {im.size[0]}x{im.size[1]}")
                im.load()
                out = vignette(im, sci)
            buf = io.BytesIO()
            out.convert("RGB").save(buf, "WEBP", quality=85)
        except NotFound:
            _write_marker(m, "404", now)
            log.info("no audubon scan for %s", sci)
            return None
        except VignetteError as exc:
            _write_marker(m, f"vignette: {exc}"[:200], now)
            log.warning("audubon plate %s for %s unusable: %s", e["plate"], sci, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            _write_marker(m, f"{type(exc).__name__}: {exc}"[:200], now)
            log.warning("audubon fetch failed for %s: %s", sci, exc)
            return None
        _write_atomic(p, buf.getvalue())
        if m.exists():
            m.unlink()
        return p

    def cached_plates(self) -> list[dict]:
        """Distinct plates with a vignette on disk, by plate number, for /attribution."""
        seen: dict[int, dict] = {}
        for s, e in self.table.items():
            if e["plate"] not in seen and (self.dir / VIGNETTE_VERSION / f"{s}.webp").exists():
                seen[e["plate"]] = {k: e.get(k) for k in ("plate", "title", "page", "credit", "credit_url")}
        return [seen[n] for n in sorted(seen)]


# ----------------------------------------------------------------- vignette
# Constants were set on 16 Havell plates (see the design doc); all geometry
# runs at a 960 px working width, the Commons thumbnail width.
WORK_W = 960
DEFAULT_PAPER = (238, 228, 200)   # median margin paper of 14 measured plates
INK_T = 40           # "ink" = darker than paper by >= 40 in some channel
BAND = 0.06          # sheet edges / scanner surround, zeroed for the component search
OPEN = 5             # opening size at half scale: removes strokes thinner than ~10 px
CELL = 4             # grid cell, px
KEEP = 0.02          # min mass of a secondary component, fraction of main
DEBRIS = 0.10        # band-touching components below this fraction of main are dropped
DENSE = 0.25         # a bled side grows while the next row/column is >= 25 % core
PAD = 0.04           # pad on non-bleed sides, fraction of the picture's longer side
GROW = 1             # picture-mask growth beyond the picture box, in cells
TAIL = 4             # rows inside a non-bleed top/bottom held to the core's edge, in cells
                     # (about an engraved title's height at the working width)
THICK = 5            # px; ink at least this wide still counts as picture past the core's edge
FILL = 128           # a cell this full of ink (of 255) continues a tip past the core's edge
LETTER = 80          # engraved lettering is darker than paper by >= 80 somewhere; wash is not
SOFT_LO, SOFT_HI = 18, 40         # background flattening ramp (darker-than-paper amount)
ASPECT = (0.75, 1.33)


class _Comp(NamedTuple):
    mass: int
    x0: int
    y0: int
    x1: int          # half-open, grid cells
    y1: int


@dataclass(frozen=True)
class VignetteGeometry:
    size: tuple[int, int]                     # working image, 960 wide
    picture: tuple[int, int, int, int]        # picture box, px at the working scale
    crop: tuple[int, int, int, int]           # padded crop box, px
    paper: tuple[int, int, int]
    paper_source: str                         # "margin" | "default"
    bleed: str                                # sides that bled, subset of "TBLR"
    _img: Image.Image = field(repr=False, compare=False)
    _ink: Image.Image = field(repr=False, compare=False)
    _pm: Image.Image = field(repr=False, compare=False)   # picture mask, cell grid


def _solid(size, color) -> Image.Image:
    return Image.new("RGB", size, color)


def _darker_than(img: Image.Image, color) -> Image.Image:
    """Per pixel, the most any channel is darker than `color` (one-sided)."""
    r, g, b = ImageChops.subtract(_solid(img.size, color), img).split()
    return ImageChops.lighter(ImageChops.lighter(r, g), b)


def _thresh(img: Image.Image, test) -> Image.Image:
    return img.point(lambda v: 255 if test(v) else 0)


def _components(dil: bytes, grid: bytes, gw: int, gh: int) -> list[_Comp]:
    """4-connected components of `dil`; mass counts `grid` cells. Largest first."""
    seen = bytearray(gw * gh)
    comps = []
    for i in range(gw * gh):
        if not dil[i] or seen[i]:
            continue
        seen[i] = 1
        stack = [i]
        mass = 0
        x0 = x1 = i % gw
        y0 = y1 = i // gw
        while stack:
            j = stack.pop()
            y, x = divmod(j, gw)
            if grid[j]:
                mass += 1
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for k, ok in ((j - 1, x > 0), (j + 1, x < gw - 1), (j - gw, y > 0), (j + gw, y < gh - 1)):
                if ok and dil[k] and not seen[k]:
                    seen[k] = 1
                    stack.append(k)
        comps.append(_Comp(mass, x0, y0, x1 + 1, y1 + 1))
    comps.sort(key=lambda c: -c.mass)
    return comps


def _median_masked(img: Image.Image, mask: Image.Image) -> tuple[tuple[int, int, int] | None, int]:
    out = []
    n = 0
    for band in img.split():
        h = band.histogram(mask)
        n = sum(h)
        if n == 0:
            return None, 0
        acc = 0
        for v in range(256):
            acc += h[v]
            if acc * 2 >= n:
                out.append(v)
                break
    return tuple(out), n


def _vignette_geometry(img: Image.Image, sci: str = "") -> VignetteGeometry:
    """Steps 1-9 of the design's vignette pipeline: find the picture, the
    crop, the paper colour, and the picture mask. Raises VignetteError."""
    img = img.convert("RGB")
    if img.width != WORK_W:
        img = img.resize((WORK_W, max(1, round(img.height * WORK_W / img.width))), Image.LANCZOS)
    W, H = img.size
    bx, by = round(W * BAND), round(H * BAND)
    # 1. one-sided ink mask against DEFAULT_PAPER
    ink = _thresh(_darker_than(img, DEFAULT_PAPER), lambda v: v >= INK_T)
    # 2. opening at half scale, then the 4 px grid
    half = _thresh(ink.reduce(2), lambda v: v >= 128)
    core = half.filter(ImageFilter.MinFilter(OPEN)).filter(ImageFilter.MaxFilter(OPEN))
    core = _thresh(core.reduce(2), lambda v: v > 0)
    gw, gh = core.size
    # 3. band zeroed for the component search
    ib = (bx // CELL, by // CELL, -(-(W - bx) // CELL), -(-(H - by) // CELL))
    inner = Image.new("L", core.size, 0)
    ImageDraw.Draw(inner).rectangle((ib[0], ib[1], ib[2] - 1, ib[3] - 1), fill=255)
    grid = ImageChops.multiply(core, inner)
    # 4. components
    comps = _components(grid.filter(ImageFilter.MaxFilter(3)).tobytes(), grid.tobytes(), gw, gh)
    comps = [c for c in comps if c.mass > 0]
    if not comps:
        raise VignetteError("no picture")
    # 5. choose the picture
    main = comps[0]
    keep = [main]
    for c in comps[1:]:
        if c.mass < KEEP * main.mass:
            continue
        w, h = c.x1 - c.x0, c.y1 - c.y0
        if (c.y1 <= main.y0 or c.y0 >= main.y1) and w / h >= 3:
            continue                     # a text line that survived the opening
        touches = c.x0 <= ib[0] or c.y0 <= ib[1] or c.x1 >= ib[2] or c.y1 >= ib[3]
        if touches and c.mass < DEBRIS * main.mass:
            continue                     # margin stain or sheet debris
        keep.append(c)
    x0, y0 = min(c.x0 for c in keep), min(c.y0 for c in keep)
    x1, y1 = max(c.x1 for c in keep), max(c.y1 for c in keep)
    # 6. full-bleed sides grow over the unbanded core while it stays dense
    cb = core.tobytes()

    def row(y, a, b):
        return sum(1 for x in range(a, b) if cb[y * gw + x])

    def col(x, a, b):
        return sum(1 for y in range(a, b) if cb[y * gw + x])
    bleed = ""
    if y0 <= ib[1] + 1:
        while y0 > 0 and row(y0 - 1, x0, x1) >= DENSE * (x1 - x0):
            y0 -= 1
        bleed += "T"
    if y1 >= ib[3] - 1:
        while y1 < gh and row(y1, x0, x1) >= DENSE * (x1 - x0):
            y1 += 1
        bleed += "B"
    if x0 <= ib[0] + 1:
        while x0 > 0 and col(x0 - 1, y0, y1) >= DENSE * (y1 - y0):
            x0 -= 1
        bleed += "L"
    if x1 >= ib[2] - 1:
        while x1 < gw and col(x1, y0, y1) >= DENSE * (y1 - y0):
            x1 += 1
        bleed += "R"
    px0, py0, px1, py1 = x0 * CELL, y0 * CELL, min(W, x1 * CELL), min(H, y1 * CELL)
    # 7. margin paper: inner rectangle minus the picture box, light and low-chroma
    m = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(m)
    md.rectangle((bx, by, W - bx - 1, H - by - 1), fill=255)
    md.rectangle((px0, py0, px1 - 1, py1 - 1), fill=0)
    r, g, b = img.split()
    chroma = ImageChops.subtract(ImageChops.lighter(ImageChops.lighter(r, g), b),
                                 ImageChops.darker(ImageChops.darker(r, g), b))
    m = ImageChops.multiply(m, _thresh(img.convert("L"), lambda v: v >= 200))
    m = ImageChops.multiply(m, _thresh(chroma, lambda v: v <= 60))
    paper, n = _median_masked(img, m)
    source = "margin"
    if paper is None or n < 0.005 * W * H:
        paper, source = DEFAULT_PAPER, "default"
        log.debug("audubon %s: default paper", sci)
    # 8. crop box: pad non-bleed sides
    pad = round(PAD * max(px1 - px0, py1 - py0))
    crop = (px0 if "L" in bleed else max(0, px0 - pad), py0 if "T" in bleed else max(0, py0 - pad),
            px1 if "R" in bleed else min(W, px1 + pad), py1 if "B" in bleed else min(H, py1 + pad))
    if (crop[2] - crop[0]) * (crop[3] - crop[1]) < 0.01 * W * H:
        raise VignetteError("picture too small")
    # 9. picture mask: raw ink cells 8-connected to kept core, inside the grown box
    raw = _thresh(ink.reduce(CELL), lambda v: v > 0)
    rw, rh = raw.size
    rb = raw.tobytes()
    pm = bytearray(rw * rh)
    stack = []

    def seed(ax0, ay0, ax1, ay1):
        for y in range(ay0, min(ay1, rh)):
            for x in range(ax0, min(ax1, rw)):
                j = y * rw + x
                if cb[j] and rb[j] and not pm[j]:
                    pm[j] = 1
                    stack.append(j)
    for c in keep:
        seed(c.x0, c.y0, c.x1, c.y1)
    if bleed:
        seed(x0, y0, x1, y1)
    # grown box, clipped to the crop's cells
    gx0, gy0 = max(crop[0] // CELL, x0 - GROW), max(crop[1] // CELL, y0 - GROW)
    gx1 = min(rw, -(-crop[2] // CELL), x1 + GROW)
    gy1 = min(rh, -(-crop[3] // CELL), y1 + GROW)
    while stack:
        j = stack.pop()
        y, x = divmod(j, rw)
        for dy in (-1, 0, 1):
            ny = y + dy
            if not gy0 <= ny < gy1:
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                k = ny * rw + nx
                if gx0 <= nx < gx1 and rb[k] and not pm[k]:
                    pm[k] = 1
                    stack.append(k)
    # 9b. the title under the picture joins it through raw ink. On a
    # non-bleed bottom, within TAIL rows of the picture box and below, a cell
    # lower than the core in its own and both neighbouring columns is a
    # suspect, unless it holds solid ink (a twig or leaf tip that the opening
    # shortened; engraved lettering is thinner than THICK). Suspects joined
    # to letter-dark ink are dropped; the rest are faint wash (a cloud's
    # lower edge) and stay. The top is left alone: plate numbers there stand
    # clear of the picture, and clipping it only trimmed feather tips.
    drop = bytearray(rw * rh)
    if "B" not in bleed:
        thick = ink.filter(ImageFilter.MinFilter(THICK)).filter(ImageFilter.MaxFilter(THICK)).reduce(CELL).tobytes()
        dark = _thresh(_darker_than(img, DEFAULT_PAPER), lambda v: v >= LETTER).reduce(CELL).tobytes()
        edge = [None] * rw
        for x in range(x0, min(x1, rw)):
            edge[x] = next((y for y in range(min(y1, rh) - 1, y0 - 1, -1) if cb[y * rw + x]), None)
        for x in range(rw):
            near = [e for e in edge[max(0, x - 1):x + 2] if e is not None]
            lim = max(near) if near else -1
            for y in range(max(0, y1 - TAIL, lim + 1), rh):
                j = y * rw + x
                if pm[j] and not thick[j]:
                    drop[j] = 1
        # a pointed tip (tail, bill, leaf) tapers below THICK before it ends:
        # a mostly-ink cell touching the core stays. Lettering is open line
        # work and rarely fills a cell this densely.
        fill = ink.reduce(CELL).tobytes()
        for j in range(rw * rh):
            if drop[j] and fill[j] >= FILL and any(
                    0 <= j + d < rw * rh and cb[j + d] for d in (-rw - 1, -rw, -rw + 1, -1, 1)):
                drop[j] = 0
        stack = [j for j in range(rw * rh) if drop[j] and dark[j]]
        for j in stack:
            drop[j] = 2
        while stack:
            j = stack.pop()
            y, x = divmod(j, rw)
            for ny in range(max(0, y - 1), min(rh, y + 2)):
                for nx in range(max(0, x - 1), min(rw, x + 2)):
                    k = ny * rw + nx
                    if drop[k] == 1:
                        drop[k] = 2
                        stack.append(k)
    # the 1-cell halo keeps the picture's soft edges; dropped cells lose it
    pmi = Image.frombytes("L", (rw, rh), bytes(255 if v and drop[j] != 2 else 0 for j, v in enumerate(pm)))
    pmi = ImageChops.subtract(pmi.filter(ImageFilter.MaxFilter(3)),
                              Image.frombytes("L", (rw, rh), bytes(255 if v == 2 else 0 for v in drop)))
    return VignetteGeometry((W, H), (px0, py0, px1, py1), crop, paper, source, bleed, img, ink, pmi)


def vignette(img: Image.Image, sci: str = "") -> Image.Image:
    """Crop a plate scan to its picture, erase captions and stains, and
    recolour the paper to CARD so it reads as printed on the card. Pillow
    only. Raises VignetteError when no usable picture is found."""
    g = _vignette_geometry(img, sci)
    qx0, qy0, qx1, qy1 = g.crop
    crop = g._img.crop(g.crop)
    # 10. background: non-picture cells reachable from the crop border
    cx0, cy0 = qx0 // CELL, qy0 // CELL
    cx1, cy1 = min(g._pm.width, -(-qx1 // CELL)), min(g._pm.height, -(-qy1 // CELL))
    sub = g._pm.crop((cx0, cy0, cx1, cy1))
    w2, h2 = sub.size
    pic = sub.tobytes()
    bg = bytearray(w2 * h2)
    stack = [j for j in itertools.chain(range(w2), range((h2 - 1) * w2, h2 * w2),
                                        range(0, h2 * w2, w2), range(w2 - 1, h2 * w2, w2))
             if not pic[j]]
    for j in stack:
        bg[j] = 1
    while stack:
        j = stack.pop()
        y, x = divmod(j, w2)
        for k, ok in ((j - 1, x > 0), (j + 1, x < w2 - 1), (j - w2, y > 0), (j + w2, y < h2 - 1)):
            if ok and not pic[k] and not bg[k]:
                bg[k] = 1
                stack.append(k)
    bgm = Image.frombytes("L", (w2, h2), bytes(255 if v else 0 for v in bg))
    ox, oy = qx0 - cx0 * CELL, qy0 - cy0 * CELL
    bgm = bgm.resize((w2 * CELL, h2 * CELL), Image.NEAREST).crop(
        (ox, oy, ox + crop.width, oy + crop.height)).filter(ImageFilter.GaussianBlur(2))
    # 11. colour: multiply paper -> CARD; flatten background only
    mult = Image.merge("RGB", [band.point(lambda v, k=CARD[i] / max(1, g.paper[i]): min(255, round(v * k)))
                               for i, band in enumerate(crop.split())])
    span = SOFT_HI - SOFT_LO
    soft = _darker_than(crop, g.paper).point(
        lambda v: 255 if v <= SOFT_LO else (0 if v >= SOFT_HI else round(255 * (SOFT_HI - v) / span)))
    ink_d = g._ink.crop(g.crop).filter(ImageFilter.MaxFilter(5))
    flat = ImageChops.multiply(bgm, ImageChops.lighter(soft, ink_d).filter(ImageFilter.GaussianBlur(1)))
    out = Image.composite(_solid(crop.size, CARD), mult, flat)
    # 12. aspect clamp by extending with CARD, then bound the size
    cw, ch = out.size
    a = cw / ch
    canvas = ((round(ASPECT[0] * ch), ch) if a < ASPECT[0]
              else (cw, round(cw / ASPECT[1])) if a > ASPECT[1] else (cw, ch))
    if canvas != (cw, ch):
        c2 = _solid(canvas, CARD)
        c2.paste(out, ((canvas[0] - cw) // 2, (canvas[1] - ch) // 2))
        out = c2
    return ImageOps.contain(out, (VIGNETTE_MAX, VIGNETTE_MAX), Image.LANCZOS).convert("RGB")


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
    x0, y0, x1, y1 = _placeholder_box(px, py, pw, ph)
    lw, ii = _frame_widths(pw)
    _card_frame(draw, x0, y0, x1, y1, lw, ii)
    _ornament(draw, (x0 + x1) // 2, (y0 + y1) // 2, round((x1 - x0) * 0.24), RULE, lw)


def _placeholder_box(px: int, py: int, pw: int, ph: int) -> tuple[int, int, int, int]:
    """The placeholder card, inclusive corners: 0.72 x 0.84 of the plate area, bottom-aligned."""
    ci = round(pw * 0.14)
    return px + ci, py + round(ph * 0.16), px + pw - ci - 1, py + ph - 1


def _frame_widths(pw: int) -> tuple[int, int]:
    """(rule width, inner-rule inset) for a card in a plate area pw wide."""
    return max(1, round(pw / 160)), max(3, round(pw * 0.03))


def _card_frame(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, lw: int, ii: int) -> None:
    """CARD fill, outer rule, inner rule inset by ii: shared by placeholders and vignettes."""
    draw.rectangle((x0, y0, x1, y1), fill=CARD, outline=RULE, width=lw)
    draw.rectangle((x0 + ii, y0 + ii, x1 - ii, y1 - ii), outline=RULE, width=lw)


def _vignette_frame_box(px: int, py: int, pw: int, ph: int, aspect: float) -> tuple[int, int, int, int]:
    """The largest box of `aspect` (w/h) no wider than the plate area, no
    taller than the placeholder and no larger in area than it, bottom-aligned
    and centred like the placeholder. Inclusive corners."""
    c0, c1, c2, c3 = _placeholder_box(px, py, pw, ph)
    cw, ch = c2 - c0 + 1, c3 - c1 + 1
    h = min(ch, pw / aspect, math.sqrt(cw * ch / aspect))
    w = min(pw, math.floor(aspect * h))
    h = math.floor(h)
    x0 = px + (pw - w) // 2
    y1 = py + ph - 1
    return x0, y1 - h + 1, x0 + w - 1, y1


def _vignette_card(img: Image.Image, draw: ImageDraw.ImageDraw, path: Path,
                   px: int, py: int, pw: int, ph: int) -> bool:
    """An Audubon vignette in the placeholder's double-rule frame, no lozenge.
    False (and the file deleted) when the cached vignette is unreadable, so
    the caller draws the placeholder instead."""
    try:
        with Image.open(path) as im:
            vig = im.convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- corrupt cached file
        log.warning("bad cached vignette %s: %s; deleting", path, exc)
        try:
            path.unlink()
        except OSError:
            pass
        return False
    x0, y0, x1, y1 = _vignette_frame_box(px, py, pw, ph, vig.width / vig.height)
    lw, ii = _frame_widths(pw)
    _card_frame(draw, x0, y0, x1, y1, lw, ii)
    room = (x1 - x0 + 1 - 4 * ii, y1 - y0 + 1 - 4 * ii)
    if min(room) > 0:
        vig = ImageOps.contain(vig, room, Image.LANCZOS)
        img.paste(vig, (x0 + 2 * ii + (room[0] - vig.width) // 2, y0 + 2 * ii + (room[1] - vig.height) // 2))
    return True


# ----------------------------------------------------------------- render
@dataclass(frozen=True)
class Rendered:
    png: bytes
    shown: int
    dropped: int
    quiet: bool              # True when the "all quiet" path was taken
    deferred: int            # plates not attempted because FETCH_BUDGET ran out
    boxes: tuple[Box, ...] = ()                   # one per shown species, as pack() made them
    names: tuple[tuple[str, str], ...] = ()       # (scientific, common) per box
    arts: tuple[str | None, ...] = ()             # what was actually drawn per box
    token: str = field(init=False, repr=False)    # content hash; names the PNG URL

    def __post_init__(self) -> None:
        # The names are part of the token: identical pixels with different
        # scientific names give different click targets.
        meta = json.dumps([self.names, self.arts], ensure_ascii=True).encode()
        object.__setattr__(self, "token", hashlib.sha256(self.png + meta).hexdigest()[:16])

    def layout(self, width: int, height: int) -> dict:
        """Click targets as percentages of the image, from the same boxes the
        pixels were drawn in."""
        pct = lambda v, d: round(100 * v / d, 4)  # noqa: E731
        return {
            "token": self.token, "w": width, "h": height,
            "shown": self.shown, "dropped": self.dropped,
            "targets": [
                {"scientific_name": sci, "common_name": com, "stem": stem(sci), "art": kind,
                 "x": pct(b.x, width), "y": pct(b.y, height), "w": pct(b.w, width), "h": pct(b.h, height)}
                for b, (sci, com), kind in zip(self.boxes, self.names, self.arts)
            ],
        }


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

    # Fetch (within budget) each shown species' art: Fugleramme first, then
    # Audubon. Past the deadline only what is on disk is drawn.
    deferred = 0
    aud = art.audubon
    for sp in species[:len(boxes)]:
        name = sp.scientific_name
        if not stem(name):
            continue
        if time.monotonic() < deadline:
            p = art.ensure_plate(name, now)
            if p is None and aud is not None and time.monotonic() < deadline:
                aud.ensure(name, now)
        if art.art_kind(name) is None and (
                not art.marker_path(name).exists()
                or (aud is not None and aud.entry(name) is not None and not aud.marker_path(name).exists())):
            deferred += 1   # a source was not tried yet
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
    arts: list[str | None] = []
    for sp, box, lines in zip(species, boxes, names):
        px, py, pw, ph = box.x + inset, box.y + inset, cell - 2 * inset, cell - 2 * inset
        name = sp.scientific_name
        kind = art.art_kind(name) if stem(name) else None
        drawn = False
        drew: str | None = None
        if kind == "fugleramme":
            path = art.plate_path(name)
            try:
                with Image.open(path) as im:
                    im = ImageOps.contain(im.convert("RGBA"), (pw, ph))
                img.paste(im, (px + (pw - im.width) // 2, py + ph - im.height), im)
                drawn = True
                drew = "fugleramme"
            except Exception as exc:  # noqa: BLE001 -- corrupt cached file
                log.warning("bad cached plate %s: %s; deleting", path, exc)
                try:
                    path.unlink()
                except OSError:
                    pass
        if not drawn and aud is not None and aud.has_art(name):
            drawn = _vignette_card(img, draw, aud.vignette_path(name), px, py, pw, ph)
            drew = "audubon" if drawn else None
        if not drawn:
            _plate_card(draw, px, py, pw, ph)
        arts.append(drew)
        # Names hang from the same line in every cell; a wrapped name adds a
        # second line below rather than shifting the first.
        y = box.y + cell - inset + round(cell * 0.05) + ascent
        for line in lines:
            draw.text((box.x + cell // 2, y), line, font=font, fill=INK, anchor="ms")
            y += line_h

    return Rendered(_png(img), len(boxes), dropped, False, deferred, tuple(boxes),
                    tuple((s.scientific_name, s.common_name) for s in species[:len(boxes)]), tuple(arts))


# ----------------------------------------------------------------- cache
def cache_key(species: list[Species], art: Artwork) -> tuple:
    """Everything that changes a pixel: species order, names, and which art
    is on disk (cut-out, vignette or none). Cameras, times and first_ever
    are JSON-only."""
    return tuple(
        (s.scientific_name, s.common_name, art.art_kind(s.scientific_name))
        for s in species
    )


@dataclass
class RenderCache:
    art: Artwork
    max_entries: int = 4
    max_tokens: int = 8                               # PNGs kept for /collage.png?v=
    renders: int = 0                                  # test observability
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _entries: dict[tuple[int, int, int], tuple[tuple, Rendered]] = field(default_factory=dict)
    _tok_lock: threading.Lock = field(default_factory=threading.Lock)
    _by_token: OrderedDict[str, Rendered] = field(default_factory=OrderedDict)

    def get(self, species: list[Species], width: int, height: int, hours: int,
            now: dt.datetime | None = None) -> bytes:
        return self.get_rendered(species, width, height, hours, now).png

    def get_rendered(self, species: list[Species], width: int, height: int, hours: int,
                     now: dt.datetime | None = None) -> Rendered:
        with self._lock:
            k = (width, height, hours)
            key = cache_key(species, self.art)
            hit = self._entries.get(k)
            if hit and hit[0] == key:
                out = hit[1]
            else:
                t0 = time.monotonic()
                out = render(species, self.art, width, height, hours, now=now)
                self.renders += 1
                log.info("rendered %sx%s in %d ms: %s species, %s dropped, %s deferred, quiet=%s",
                         width, height, (time.monotonic() - t0) * 1000,
                         out.shown, out.dropped, out.deferred, out.quiet)
                if k not in self._entries and len(self._entries) >= self.max_entries:
                    self._entries.pop(next(iter(self._entries)))   # oldest inserted
                self._entries[k] = (key, out)
        with self._tok_lock:
            self._by_token[out.token] = out
            self._by_token.move_to_end(out.token)
            while len(self._by_token) > self.max_tokens:
                self._by_token.popitem(last=False)
        return out

    def by_token(self, token: str) -> Rendered | None:
        with self._tok_lock:
            return self._by_token.get(token)
