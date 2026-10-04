"""HTTP server for the collage page. Started by loop.py when SERVE_PORT is set.

Routes (all GET, LAN only, one consumer: a Home Assistant Webpage card):
  /                    HTML page that shows /collage.png and refreshes it
  /collage.png         the collage; ?hours= ?w= ?h=
  /api/recent          JSON of the species behind the collage; ?hours=
  /attribution         Fugleramme credit plus its ATTRIBUTION.md, and the Audubon credit
  /favicon.ico         204

Environment (read once by load_serve_config):
  SERVE_PORT      unset/empty = no server at all (the default)
  COLLAGE_HOURS   window in hours (default 24, 1..720)
  ARTWORK_REF     fugleramme commit/branch for plate URLs (default: a pinned sha)
  ARTWORK_DIR     plate cache (default $DATA_DIR/artwork)
  AUDUBON_FALLBACK  1 (default) = Audubon plates for species Fugleramme lacks, 0 = off
  MIN_CONFIDENCE  rows below this are left off the page and /api/recent
                  (default 0.5, the same variable and default as the capture loop)

The server is a daemon thread beside the capture loop. Each request opens its
own read-only SQLite connection; nothing is shared with the loop.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import frame
from birdlisten import ConfigError

log = logging.getLogger("serve")
DEFAULT_W, DEFAULT_H = 1600, 1200
MIN_SIZE, MAX_SIZE = 200, 4000
MAX_HOURS = 24 * 30
REFRESH_SECONDS = 60
_REF_RE = re.compile(r"[A-Za-z0-9._/-]+")
_TOKEN_RE = re.compile(r"[0-9a-f]{16}")
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/static/page.js": ("page.js", "text/javascript; charset=utf-8"),
    "/static/page.css": ("page.css", "text/css; charset=utf-8"),
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; "
       "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'")
IMMUTABLE = "public, max-age=31536000, immutable"

CREDIT_HTML = (
    'Bird plates are from the <a href="https://github.com/arnegiacomo/fugleramme">Fugleramme</a> '
    "project (<code>assets/artwork/classic</code>), licensed "
    '<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. '
)


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class ServeConfig:
    port: int
    hours: int
    db_path: Path
    art: frame.Artwork
    min_confidence: float = 0.5


def load_serve_config(env=os.environ) -> ServeConfig | None:
    """None when SERVE_PORT is unset or empty (the server is off)."""
    raw_port = env.get("SERVE_PORT", "").strip()
    if not raw_port:
        return None
    try:
        port = int(raw_port)
        hours = int(env.get("COLLAGE_HOURS", "24"))
        min_conf = float(env.get("MIN_CONFIDENCE", "0.5"))
    except ValueError as exc:
        raise ConfigError(f"bad numeric setting: {exc}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"SERVE_PORT must be 1..65535, got {port}")
    if not 1 <= hours <= MAX_HOURS:
        raise ConfigError(f"COLLAGE_HOURS must be 1..{MAX_HOURS}, got {hours}")
    if not 0 <= min_conf <= 1:
        raise ConfigError(f"MIN_CONFIDENCE must be 0..1, got {min_conf}")
    ref = env.get("ARTWORK_REF", frame.DEFAULT_ARTWORK_REF).strip()
    if not ref or not _REF_RE.fullmatch(ref):
        raise ConfigError(f"ARTWORK_REF must be a git ref or sha ([A-Za-z0-9._/-]), got {ref!r}")
    fallback = env.get("AUDUBON_FALLBACK", "1").strip()
    if fallback not in ("0", "1"):
        raise ConfigError("AUDUBON_FALLBACK must be 0 or 1")
    data_dir = Path(env.get("DATA_DIR", "/data"))
    art_dir = Path(env.get("ARTWORK_DIR", "").strip() or data_dir / "artwork")
    # A table that fails to load logs an ERROR and leaves the fallback off.
    audubon = frame.Audubon.load(art_dir / "audubon") if fallback == "1" else None
    return ServeConfig(port=port, hours=hours, db_path=data_dir / "birdlisten.sqlite",
                       art=frame.Artwork(art_dir, ref, audubon), min_confidence=min_conf)


# ----------------------------------------------------------------- request helpers
class BadRequest(ValueError):
    pass


def int_param(qs: dict[str, list[str]], name: str, default: int, lo: int, hi: int) -> int:
    """Missing -> default; repeated -> first; non-int or out of range -> 400."""
    if name not in qs:
        return default
    try:
        value = int(qs[name][0])
    except ValueError:
        raise BadRequest(f"{name} must be {lo}..{hi}") from None
    if not lo <= value <= hi:
        raise BadRequest(f"{name} must be {lo}..{hi}")
    return value


def load_species(cfg: ServeConfig, hours: int, now: dt.datetime) -> list[frame.Species]:
    """Per-request read-only connection; nothing shared with the capture loop.
    No DB yet (server up before the first pass wrote anything) = no species."""
    if not cfg.db_path.exists():
        return []
    conn = frame.open_ro(cfg.db_path)
    try:
        return frame.recent_species(conn, now, hours, cfg.min_confidence)
    finally:
        conn.close()


def recent_json(cfg: ServeConfig, hours: int, now: dt.datetime | None = None) -> dict:
    now = now or frame.utcnow()
    return {
        "hours": hours,
        "generated_at": now.isoformat(timespec="seconds"),
        "species": [
            {
                "scientific_name": s.scientific_name,
                "common_name": s.common_name,
                "last_heard": s.last_heard,
                "count": s.count,
                "cameras": list(s.cameras),
                "first_ever": s.first_ever,
                # Fugleramme cut-out or Audubon plate on disk; never fetches
                "has_plate": cfg.art.has_art(s.scientific_name),
            }
            for s in load_species(cfg, hours, now)
        ],
    }


def layout_json(rendered: frame.Rendered, w: int, h: int, hours: int) -> dict:
    """What /api/layout returns and / embeds: the render's click targets and
    the URL of exactly the PNG they were built from."""
    return {**rendered.layout(w, h), "png": f"/collage.png?v={rendered.token}", "hours": hours}


def _json_block(obj: dict) -> str:
    """JSON safe inside <script type="application/json">: no name can close the element."""
    text = json.dumps(obj, ensure_ascii=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def index_html(layout: dict, w: int, h: int, refresh_ms: int = REFRESH_SECONDS * 1000,
               audubon: bool = False) -> str:
    """The collage <img> plus its click targets' layout as a JSON data block.
    static/page.js draws the targets, swaps image and targets together on
    refresh, and runs the pop-up; without JS the page is the plain image and
    the <noscript> meta refresh (only honoured in <head>). `w`, `h` and
    `refresh_ms` are validated ints and the token is hex, so only the JSON
    block needs escaping. `audubon` names the second artwork source."""
    credit = ("Plates from Fugleramme (CC BY-SA 4.0) and Audubon's <i>Birds of America</i>"
              if audubon else "Plates from Fugleramme, CC BY-SA 4.0")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<noscript><meta http-equiv="refresh" content="300"></noscript>
<title>birds heard recently</title>
<link rel="stylesheet" href="/static/page.css">
<script src="/static/page.js" defer></script>
</head>
<body data-refresh-ms="{int(refresh_ms)}">
<div id="stage">
<img id="c" src="{layout['png']}" width="{int(w)}" height="{int(h)}" alt="birds heard recently" tabindex="-1">
<div id="targets"></div>
</div>
<footer><a href="/attribution">{credit}</a></footer>
<div id="scrim" hidden></div>
<section id="card" role="dialog" aria-modal="true" aria-labelledby="card-title" hidden></section>
<script type="application/json" id="layout">{_json_block(layout)}</script>
</body>
</html>
"""


def attribution_html(art: frame.Artwork) -> str:
    """Always 200: the static credit does not depend on the fetched file."""
    art.ensure_meta()
    text = art.attribution_text()
    if text is None:
        tail = "(not fetched yet; it appears here after the first render)"
        body = ""
    else:
        tail = "Full per-plate sources are in the project's ATTRIBUTION.md, reproduced below."
        body = f"<pre>{html.escape(text)}</pre>"
    audubon = audubon_html(art.audubon) if art.audubon is not None else ""
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>artwork attribution</title>
<link rel="stylesheet" href="/static/page.css"></head>
<body class="attribution">
<p>{CREDIT_HTML}{tail}</p>
{body}
{audubon}</body>
</html>
"""


AUDUBON_EDITIONS = {
    "havell": ("(London, 1827&ndash;1838), Havell edition: hand-coloured engravings by Robert Havell Jr. "
               "(plates 1&ndash;10 first engraved by W. H. Lizars, Edinburgh)"),
    "octavo": "(Philadelphia, 1840&ndash;1844), octavo edition: hand-coloured lithographs by J. T. Bowen",
}


def audubon_html(aud: frame.Audubon) -> str:
    """The Audubon section of /attribution, from disk only (never fetches)."""
    edition = AUDUBON_EDITIONS.get(aud.edition, AUDUBON_EDITIONS["havell"])
    esc = lambda v: html.escape(str(v), quote=True)  # noqa: E731
    items = []
    for p in aud.cached_plates():
        item = f'<a href="{esc(p["page"])}">Plate {esc(p["plate"])}, {esc(p["title"] or "")}</a>'
        if p.get("credit"):
            credit = esc(p["credit"])
            if p.get("credit_url"):
                credit = f'<a href="{esc(p["credit_url"])}">{credit}</a>'
            item += f", {credit}"
        items.append(f"<li>{item}</li>")
    plates = "\n".join(items) or "<li>(none fetched yet)</li>"
    return f"""<h2>Audubon</h2>
<p>Some plates are from John James Audubon, <i>The Birds of America</i>
{edition}. Scans from <a href="https://commons.wikimedia.org/wiki/Category:The_Birds_of_America">Wikimedia
Commons</a>, credited there to the University of Pittsburgh. Public domain. The plates shown here are
cropped and recoloured.</p>
<ul>
{plates}
</ul>
"""


@lru_cache(maxsize=None)
def static_file(name: str) -> bytes:
    """page.js / page.css, read once; `name` only ever comes from STATIC_FILES."""
    return (STATIC_DIR / name).read_bytes()


# ----------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server: "CollageServer"

    def _send(self, status: int, ctype: str | None, body: bytes, cache: str = "no-store",
              extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        if ctype:
            self.send_header("Content-Type", ctype)
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", CSP)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _size_params(self, qs: dict[str, list[str]]) -> tuple[int, int, int]:
        cfg = self.server.cfg
        return (int_param(qs, "hours", cfg.hours, 1, MAX_HOURS),
                int_param(qs, "w", DEFAULT_W, MIN_SIZE, MAX_SIZE),
                int_param(qs, "h", DEFAULT_H, MIN_SIZE, MAX_SIZE))

    def do_GET(self) -> None:  # noqa: N802 -- http.server API
        url = urlparse(self.path)
        path = url.path
        cfg = self.server.cfg
        try:
            qs = parse_qs(url.query, keep_blank_values=False)
            now = frame.utcnow()   # once per request: window, generated_at, and marker checks agree
            if path == "/":
                hours, w, h = self._size_params(qs)
                refresh_ms = int_param(qs, "refresh_ms", REFRESH_SECONDS * 1000, 500, 3_600_000)
                r = self.server.cache.get_rendered(load_species(cfg, hours, now), w, h, hours, now=now)
                page = index_html(layout_json(r, w, h, hours), w, h, refresh_ms, cfg.art.audubon is not None)
                self._send(200, "text/html; charset=utf-8", page.encode())
            elif path == "/collage.png" and "v" in qs:
                token = qs["v"][0]
                if not _TOKEN_RE.fullmatch(token):
                    raise BadRequest("v must be 16 hex digits")
                r = self.server.cache.by_token(token)
                if r is None:
                    log.debug("GET %s: token gone", self.path)
                    self._send(404, "text/plain; charset=utf-8", b"gone\n")
                else:
                    self._send(200, "image/png", r.png, cache=IMMUTABLE)
            elif path == "/collage.png":
                hours, w, h = self._size_params(qs)
                species = load_species(cfg, hours, now)
                self._send(200, "image/png", self.server.cache.get(species, w, h, hours, now=now))
            elif path == "/api/layout":
                hours, w, h = self._size_params(qs)
                r = self.server.cache.get_rendered(load_species(cfg, hours, now), w, h, hours, now=now)
                body = json.dumps(layout_json(r, w, h, hours), ensure_ascii=False).encode()
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/api/recent":
                hours = int_param(qs, "hours", cfg.hours, 1, MAX_HOURS)
                body = json.dumps(recent_json(cfg, hours, now), ensure_ascii=False).encode()
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/attribution":
                self._send(200, "text/html; charset=utf-8", attribution_html(cfg.art).encode())
            elif path in STATIC_FILES:
                name, ctype = STATIC_FILES[path]
                self._send(200, ctype, static_file(name), cache="max-age=300")
            elif path == "/favicon.ico":
                self._send(204, None, b"")
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found\n")
        except BadRequest as exc:
            log.warning("GET %s: %s", self.path, exc)
            self._send(400, "text/plain; charset=utf-8", f"bad request: {exc}\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            return   # client went away, nothing to send
        except Exception:  # noqa: BLE001 -- one bad request must not take the thread down
            log.exception("GET %s failed", self.path)
            try:
                self._send(500, "text/plain; charset=utf-8", b"internal error\n")
            except OSError:
                pass

    def log_message(self, fmt, *args) -> None:
        log.debug(fmt, *args)   # 4xx/5xx are logged explicitly above


class CollageServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, cfg: ServeConfig, host: str = "") -> None:
        # host="" (wildcard) in production; tests pass host="127.0.0.1" and port=0.
        super().__init__((host, cfg.port), Handler)
        self.cfg = cfg
        self.cache = frame.RenderCache(cfg.art)


def start_server(cfg: ServeConfig) -> threading.Thread | None:
    """Bind and serve on a daemon thread. A bind failure disables the server
    and never touches the loop's exit code."""
    try:
        srv = CollageServer(cfg)
    except OSError as exc:
        log.error("cannot bind SERVE_PORT=%s: %s; running without the server", cfg.port, exc)
        return None
    t = threading.Thread(target=srv.serve_forever, name="serve", daemon=True)
    t.start()
    log.info("serving collage on port %s (window %sh, artwork %s@%s)",
             cfg.port, cfg.hours, cfg.art.dir, cfg.art.ref[:12])
    return t


def start_from_env(env=os.environ) -> threading.Thread | None:
    """What loop.py calls. None (silently) when SERVE_PORT is unset; None with
    an ERROR line when the config is bad or the port cannot be bound."""
    try:
        cfg = load_serve_config(env)
    except ConfigError as exc:
        log.error("config error: %s (server disabled)", exc)
        return None
    if cfg is None:
        return None
    return start_server(cfg)
