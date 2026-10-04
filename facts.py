"""Species facts for the pop-up card: Wikipedia blurb, Wikidata sizes, eBird
code and nearby reports. Fetched lazily in background threads, one upstream
call per species and source however many requests arrive, and cached per
source and species under FACTS_DIR (default $DATA_DIR/facts).

serve.py owns HTTP, frame.py owns pixels, this module owns outbound fact
fetching. frame.py never imports it, so a collage render cannot reach it.

Sources:
  Wikidata  taxon name (P225) -> enwiki article title, sizes (P2067 mass,
            P2043 length, P2050 wingspan), eBird taxon ID (P3444). CC0.
  Wikipedia REST summary of that article. CC BY-SA 4.0, credited per article.
  eBird     taxonomy (scientific/common name -> species code) and recent
            nearby observations, only when EBIRD_API_KEY is set. The key is
            sent as the X-eBirdApiToken header, built only in _ebird_headers().
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse

import frame

log = logging.getLogger("facts")

FACT_TIMEOUT = 4                  # s per request
FACT_MAX_BYTES = 2_000_000        # a Wikidata bird entity with the sitelink filter is <= ~280 KB
TAXONOMY_TIMEOUT = 30
TAXONOMY_MAX_BYTES = 8_000_000    # 3.5 MB today
FACTS_BUDGET = 2.5                # s a /api/species request may wait for fetches
MAX_CONCURRENT = 4                # outbound requests at once
OK_TTL = dt.timedelta(days=30)
NEARBY_TTL = dt.timedelta(hours=6)
TAXONOMY_TTL = dt.timedelta(days=30)
TAXONOMY_MIN = 5000               # fewer records than this is a broken response
EXTRACT_MAX = 600
NEARBY_DIST_KM = 25
NEARBY_BACK_DAYS = 7
SOURCES = ("wikidata", "wikipedia", "ebird", "nearby")

WIKIDATA_API = "https://www.wikidata.org/w/api.php?"
WIKIDATA_PAGE = "https://www.wikidata.org/wiki/"
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
EBIRD_TAXONOMY = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json&cat=species"
EBIRD_NEARBY = "https://api.ebird.org/v2/data/obs/geo/recent/"
EBIRD_SPECIES = "https://ebird.org/species/"
AAB_GUIDE = "https://www.allaboutbirds.org/guide/"
CC_BY_SA = "https://creativecommons.org/licenses/by-sa/4.0/"

BINOMIAL_RE = re.compile(r"[A-Z][a-z]+ [a-z]+(-[a-z]+)?( [a-z]+(-[a-z]+)?)?")
CODE_RE = re.compile(r"[a-z0-9]{3,12}")
QID_RE = re.compile(r"Q[0-9]{1,12}")


# ----------------------------------------------------------------- network
def http_get(url: str, headers: Mapping[str, str] | None = None,
             timeout: float = FACT_TIMEOUT, max_bytes: int = FACT_MAX_BYTES) -> bytes:
    """The only network call in this module; tests monkeypatch it. Sends the
    same descriptive User-Agent as frame.fetch_url (Wikimedia asks for one)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": frame.USER_AGENT, "Accept": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise frame.NotFound("404") from None
        raise
    if len(data) > max_bytes:
        raise ValueError("response too large")
    return data


# ----------------------------------------------------------------- config
class Secret:
    """A value that never prints. repr/str show ***; pickling is refused."""
    __slots__ = ("_v",)

    def __init__(self, v: str):
        self._v = v

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("Secret is not picklable")


@dataclass(frozen=True)
class FactsConfig:
    dir: Path
    fetch: bool = True                                   # FACTS_FETCH=0: cache only
    ebird_key: Secret | None = field(default=None, repr=False)
    lat: float | None = None
    lon: float | None = None

    @property
    def nearby_on(self) -> bool:
        return self.ebird_key is not None and self.lat is not None and self.lon is not None


@dataclass(frozen=True)
class FactsResult:
    facts: dict
    pending: list[str]


# ----------------------------------------------------------------- all about birds
def aab_url(common: str) -> str:
    """All About Birds guide URL from the common name. Never fetched."""
    name = common.replace("'", "").replace("\u2019", "").strip().replace(" ", "_")
    return AAB_GUIDE + quote(name, safe="-_")


# ----------------------------------------------------------------- wikidata
def search_url(sci: str) -> str:
    return WIKIDATA_API + urlencode({"action": "query", "list": "search",
                                     "srsearch": f'haswbstatement:"P225={sci}"',
                                     "srlimit": 5, "format": "json"})


def entities_url(qids: list[str]) -> str:
    return WIKIDATA_API + urlencode({"action": "wbgetentities", "ids": "|".join(qids),
                                     "props": "sitelinks|claims", "sitefilter": "enwiki",
                                     "format": "json"})


def parse_search(body: bytes) -> list[str]:
    hits = json.loads(body)["query"]["search"]
    return [h["title"] for h in hits if isinstance(h.get("title"), str) and QID_RE.fullmatch(h["title"])]


def _values(claims: Mapping, pid: str) -> list:
    out = []
    for c in claims.get(pid, []) or []:
        if c.get("rank") == "deprecated":
            continue
        ms = c.get("mainsnak") or {}
        if ms.get("snaktype") != "value":
            continue
        out.append(((ms.get("datavalue") or {}).get("value"), c))
    return out


def pick_entity(body: bytes, sci: str, qids: list[str]) -> tuple[str, dict] | None:
    """The first item (in search order) whose P225 equals `sci` exactly and
    that has an enwiki sitelink; else the first exact match; else None."""
    entities = json.loads(body)["entities"]
    exact = []
    for q in qids:
        e = entities.get(q)
        if not isinstance(e, dict):
            continue
        if any(v == sci for v, _ in _values(e.get("claims") or {}, "P225")):
            exact.append((q, e))
    for q, e in exact:
        if ((e.get("sitelinks") or {}).get("enwiki") or {}).get("title"):
            return q, e
    return exact[0] if exact else None


ADULT_WEIGHT, ADULT = "Q78101716", "Q80994"
UNITS = {"Q41803": ("g", 1.0), "Q11570": ("g", 1000.0), "Q174789": ("cm", 0.1),
         "Q174728": ("cm", 1.0), "Q11573": ("cm", 100.0)}
SIZE_PROPS = {"mass": ("P2067", "g", 1.0, 20000.0), "length": ("P2043", "cm", 5.0, 400.0),
              "wingspan": ("P2050", "cm", 5.0, 400.0)}


def _qual_ids(c: dict, pid: str) -> list[str] | None:
    quals = (c.get("qualifiers") or {}).get(pid)
    if quals is None:
        return None
    return [((q.get("datavalue") or {}).get("value") or {}).get("id") for q in quals]


def parse_sizes(claims: Mapping) -> dict[str, list[float] | None]:
    """{"mass": [min_g, max_g] | None, "length": [min_cm, max_cm] | None, "wingspan": ...}.
    Deprecated, non-adult and out-of-bounds values and unknown units are dropped;
    when any value is preferred rank, only preferred values count."""
    out: dict[str, list[float] | None] = {}
    for name, (pid, dim, lo, hi) in SIZE_PROPS.items():
        kept: list[tuple[str, float]] = []
        for val, c in _values(claims, pid):
            role = _qual_ids(c, "P3831")
            if role is not None and ADULT_WEIGHT not in role:
                continue
            part = _qual_ids(c, "P518")
            if part is not None and ADULT not in part:
                continue
            if not isinstance(val, dict):
                continue
            unit = UNITS.get(str(val.get("unit", "")).rsplit("/", 1)[-1])
            if unit is None or unit[0] != dim:
                continue
            try:
                v = float(val["amount"]) * unit[1]
            except (KeyError, TypeError, ValueError):
                continue
            if v != v or not lo <= v <= hi:
                continue
            kept.append((c.get("rank", "normal"), v))
        if any(r == "preferred" for r, _ in kept):
            kept = [(r, v) for r, v in kept if r == "preferred"]
        vals = [v for _, v in kept]
        out[name] = [min(vals), max(vals)] if vals else None
    return out


def _n(v: float, decimal_below: float) -> str:
    return f"{v:.1f}" if v < decimal_below else f"{round(v):d}"


def _range(lo: float, hi: float, fmt: Callable[[float], str]) -> str:
    if hi / lo < 1.05:
        return fmt((lo + hi) / 2)
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a}\u2013{b}"


def format_mass(r: list[float] | None) -> str | None:
    """US customary first: '1.1 oz (32 g)', '2.5 lb (1.1 kg)', en-dash ranges."""
    if not r:
        return None
    lo, hi = r
    if hi < 453.6:
        return f"{_range(lo, hi, lambda g: _n(g / 28.3495, 10))} oz ({_range(lo, hi, lambda g: _n(g, 10))} g)"
    return f"{_range(lo, hi, lambda g: _n(g / 453.592, 10))} lb ({_range(lo, hi, lambda g: f'{g / 1000:.1f}')} kg)"


def format_length(r: list[float] | None) -> str | None:
    """'47 in (119 cm)', '6.3 in (16 cm)'."""
    if not r:
        return None
    lo, hi = r
    return f"{_range(lo, hi, lambda c: _n(c / 2.54, 10))} in ({_range(lo, hi, lambda c: _n(c, 10))} cm)"


def ebird_code_of(claims: Mapping) -> str | None:
    for v, _ in _values(claims, "P3444"):
        if isinstance(v, str) and CODE_RE.fullmatch(v):
            return v
    return None


# ----------------------------------------------------------------- wikipedia
def summary_url(title: str) -> str:
    return SUMMARY_API + quote(title.replace(" ", "_"), safe="")


def trim_extract(text: str) -> tuple[str, bool]:
    """Cut at the last sentence end at or before EXTRACT_MAX; with none, keep
    the first sentence whatever its length."""
    if len(text) <= EXTRACT_MAX:
        return text, False
    ends = [m.start() + 1 for m in re.finditer(r"[.!?](?=\s)", text) if m.start() + 1 <= EXTRACT_MAX]
    if ends:
        cut = text[:ends[-1]]
    else:
        m = re.search(r"[.!?](?=\s|$)", text)
        cut = text[:m.start() + 1] if m else text
    return cut, len(cut) < len(text)


def parse_summary(body: bytes) -> dict | None:
    """{"title","extract","trimmed","url"} for a standard page with an https
    en.wikipedia.org URL and a non-empty extract; None (a miss) otherwise.
    Bad JSON raises (an error, retried sooner)."""
    doc = json.loads(body)
    if not isinstance(doc, dict) or doc.get("type") != "standard":
        return None
    extract = doc.get("extract")
    url = ((doc.get("content_urls") or {}).get("desktop") or {}).get("page")
    title = (doc.get("titles") or {}).get("normalized") or doc.get("title")
    if not isinstance(extract, str) or not extract.strip() or not isinstance(url, str) or not isinstance(title, str):
        return None
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != "en.wikipedia.org":
        return None   # CC BY-SA needs a link to the article; no safe link, no blurb
    text, trimmed = trim_extract(extract.strip())
    return {"title": title, "extract": text, "trimmed": trimmed, "url": url}


# ----------------------------------------------------------------- ebird
def slim_taxonomy(body: bytes, now: dt.datetime) -> dict:
    recs = json.loads(body)
    if not isinstance(recs, list):
        raise ValueError("taxonomy is not a list")
    sci: dict[str, str] = {}
    com: dict[str, str] = {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        code = r.get("speciesCode")
        if not isinstance(code, str) or not CODE_RE.fullmatch(code):
            continue
        if isinstance(r.get("sciName"), str):
            sci[r["sciName"].lower()] = code
        if isinstance(r.get("comName"), str):
            com[r["comName"].lower()] = code
    if len(sci) < TAXONOMY_MIN:
        raise ValueError(f"taxonomy has only {len(sci)} records")
    return {"fetched_at": now.isoformat(timespec="seconds"), "count": len(sci), "sci": sci, "com": com}


def match_code(tax: Mapping | None, sci: str, common: str, p3444: str | None) -> tuple[str, str] | None:
    """(code, source): taxonomy by scientific name, then by common name, then Wikidata P3444."""
    cands = []
    if tax:
        cands += [(tax.get("sci", {}).get(sci.lower()), "taxonomy"), (tax.get("com", {}).get(common.lower()), "taxonomy")]
    cands.append((p3444, "wikidata"))
    for code, src in cands:
        if isinstance(code, str) and CODE_RE.fullmatch(code):
            return code, src
    return None


def nearby_url(code: str, lat: float, lon: float) -> str:
    return EBIRD_NEARBY + quote(code, safe="") + "?" + urlencode(
        {"lat": round(lat, 2), "lng": round(lon, 2), "dist": NEARBY_DIST_KM, "back": NEARBY_BACK_DAYS})


def parse_nearby(body: bytes) -> dict:
    obs = json.loads(body)
    if not isinstance(obs, list):
        raise ValueError("nearby is not a list")
    newest = None
    for o in obs:
        raw = o.get("obsDt") if isinstance(o, dict) else None
        if not isinstance(raw, str):
            continue
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                d = dt.datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                d = None
        if d is not None and (newest is None or d > newest):
            newest = d
    return {"reported": newest is not None, "last_obs_date": newest.isoformat() if newest else None}


def _ebird_headers(key: Secret) -> dict[str, str]:
    """The only place the key leaves the Secret."""
    return {"X-eBirdApiToken": key.reveal()}


# ----------------------------------------------------------------- cache records
def rec_path(d: Path, source: str, s: str) -> Path:
    return d / "v1" / source / f"{s}.json"


def read_rec(d: Path, source: str, s: str) -> dict | None:
    p = rec_path(d, source, s)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        rec = json.loads(raw)
        if not isinstance(rec, dict) or rec.get("status") not in ("ok", "miss", "error"):
            raise ValueError("no status")
        dt.datetime.fromisoformat(rec["at"])
        if rec.get("error_at"):
            dt.datetime.fromisoformat(rec["error_at"])
    except (ValueError, KeyError, TypeError):
        log.warning("facts: corrupt cache file %s; deleting", p)
        try:
            p.unlink()
        except OSError:
            pass
        return None
    return rec


def write_rec(d: Path, source: str, s: str, status: str, now: dt.datetime, data=None,
              reason: str | None = None, error_at: dt.datetime | None = None) -> dict:
    rec = {"status": status, "at": now.isoformat(timespec="seconds"), "data": data}
    if reason:
        rec["reason"] = reason
    if error_at is not None:
        rec["error_at"] = error_at.isoformat(timespec="seconds")
    frame._write_atomic(rec_path(d, source, s), json.dumps(rec, ensure_ascii=False).encode())
    return rec


def needs_fetch(rec: dict | None, now: dt.datetime, source: str) -> bool:
    if rec is None:
        return True
    at = dt.datetime.fromisoformat(rec["at"])
    status = rec["status"]
    if status == "ok":
        if rec.get("error_at"):
            return now >= dt.datetime.fromisoformat(rec["error_at"]) + frame.ERROR_RETRY
        return now >= at + (NEARBY_TTL if source == "nearby" else OK_TTL)
    if status == "miss":
        return now >= at + (NEARBY_TTL if source == "nearby" else frame.MISSING_RETRY)
    return now >= at + frame.ERROR_RETRY


def _usable(rec: dict | None) -> bool:
    return rec is not None and rec["status"] == "ok" and rec.get("data") is not None


def _reason(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    return f"{type(exc).__name__} {code}" if isinstance(code, int) else type(exc).__name__


# ----------------------------------------------------------------- orchestrator
class Facts:
    def __init__(self, cfg: FactsConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, str], threading.Event] = {}
        self._sem = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._tax: dict | None = None
        self._tax_mtime: int | None = None
        self._tax_error_at: dt.datetime | None = None

    # -- plumbing
    def _get(self, url: str, **kw) -> bytes:
        with self._sem:
            return http_get(url, **kw)

    def _start(self, key: tuple[str, str], fn: Callable[[], None]) -> threading.Event:
        """Single flight: a job already running for `key` is reused."""
        with self._lock:
            ev = self._inflight.get(key)
            if ev is not None:
                return ev
            ev = threading.Event()
            self._inflight[key] = ev

        def run() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 -- a job bug must not kill anything
                log.error("facts: %s job failed: %s", key[0], type(exc).__name__)
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()

        threading.Thread(target=run, name=f"facts-{key[0]}", daemon=True).start()
        return ev

    def _running(self, job: str, s: str) -> threading.Event | None:
        with self._lock:
            return self._inflight.get((job, s))

    def _rec(self, source: str, s: str) -> dict | None:
        return read_rec(self.cfg.dir, source, s)

    def _fail(self, source: str, s: str, sci: str, old: dict | None, exc: BaseException,
              now: dt.datetime) -> dict:
        reason = _reason(exc)
        log.warning("facts: %s for %s failed: %s", source, sci, reason)
        if _usable(old):   # keep serving stale data; retry in ERROR_RETRY
            return write_rec(self.cfg.dir, source, s, "ok", dt.datetime.fromisoformat(old["at"]),
                             old["data"], reason, error_at=now)
        return write_rec(self.cfg.dir, source, s, "error", now, None, reason)

    def _miss(self, source: str, s: str, sci: str, now: dt.datetime) -> dict:
        log.info("facts: no %s for %s", source, sci)
        return write_rec(self.cfg.dir, source, s, "miss", now)

    # -- jobs
    def _wiki_job(self, sci: str, common: str, now: dt.datetime) -> None:
        s = frame.stem(sci)
        wd = self._rec("wikidata", s)
        if wd is None or needs_fetch(wd, now, "wikidata"):
            try:
                qids = parse_search(self._get(search_url(sci)))
                picked = pick_entity(self._get(entities_url(qids)), sci, qids) if qids else None
                if picked is None:
                    wd = self._miss("wikidata", s, sci, now)
                else:
                    qid, e = picked
                    claims = e.get("claims") or {}
                    data = {"qid": qid,
                            "enwiki": ((e.get("sitelinks") or {}).get("enwiki") or {}).get("title"),
                            "sizes": parse_sizes(claims), "ebird_code": ebird_code_of(claims)}
                    wd = write_rec(self.cfg.dir, "wikidata", s, "ok", now, data)
            except Exception as exc:  # noqa: BLE001
                wd = self._fail("wikidata", s, sci, wd, exc, now)
        # Without a Wikidata title (miss, or an error with nothing cached) fall
        # back to the summary for the scientific name, then the common name.
        enwiki = (wd.get("data") or {}).get("enwiki") if _usable(wd) else None

        wp = self._rec("wikipedia", s)
        changed = _usable(wp) and enwiki and wp["data"].get("requested") != enwiki
        if not (wp is None or needs_fetch(wp, now, "wikipedia") or changed):
            return
        cands = [t for t in dict.fromkeys([enwiki, sci, common]) if t]
        try:
            for title in cands:
                try:
                    data = parse_summary(self._get(summary_url(title)))
                except frame.NotFound:
                    continue
                if data is not None:
                    write_rec(self.cfg.dir, "wikipedia", s, "ok", now, {**data, "requested": title})
                    return
            self._miss("wikipedia", s, sci, now)
        except Exception as exc:  # noqa: BLE001
            self._fail("wikipedia", s, sci, wp, exc, now)

    def _load_tax(self) -> dict | None:
        p = self.cfg.dir / "ebird-taxonomy.json"
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            return None
        with self._lock:
            if self._tax is not None and self._tax_mtime == mtime:
                return self._tax
        try:
            tax = json.loads(p.read_bytes())
            dt.datetime.fromisoformat(tax["fetched_at"])
            if not isinstance(tax.get("sci"), dict) or not isinstance(tax.get("com"), dict):
                raise ValueError("bad taxonomy")
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("facts: corrupt taxonomy cache; deleting")
            try:
                p.unlink()
            except OSError:
                pass
            return None
        with self._lock:
            self._tax, self._tax_mtime = tax, mtime
        return tax

    def _fetch_taxonomy(self, now: dt.datetime) -> None:
        try:
            body = self._get(EBIRD_TAXONOMY, headers=_ebird_headers(self.cfg.ebird_key),
                             timeout=TAXONOMY_TIMEOUT, max_bytes=TAXONOMY_MAX_BYTES)
            tax = slim_taxonomy(body, now)
        except Exception as exc:  # noqa: BLE001
            self._tax_error_at = now
            log.warning("facts: ebird taxonomy failed: %s", _reason(exc))
            return
        self._tax_error_at = None
        frame._write_atomic(self.cfg.dir / "ebird-taxonomy.json", json.dumps(tax).encode())

    def _taxonomy(self, now: dt.datetime) -> dict | None:
        tax = self._load_tax()
        if tax is not None and now - dt.datetime.fromisoformat(tax["fetched_at"]) < TAXONOMY_TTL:
            return tax
        if self._tax_error_at is not None and now - self._tax_error_at < frame.ERROR_RETRY:
            return tax
        self._start(("taxonomy", ""), lambda: self._fetch_taxonomy(now)).wait(TAXONOMY_TIMEOUT + 5)
        return self._load_tax() or tax

    def _ebird_job(self, sci: str, common: str, now: dt.datetime) -> None:
        s = frame.stem(sci)
        old = self._rec("ebird", s)
        tax = self._taxonomy(now)
        wd = self._rec("wikidata", s)
        p3444 = wd["data"].get("ebird_code") if _usable(wd) else None
        m = match_code(tax, sci, common, p3444)
        if m is None:
            if tax is None:
                self._fail("ebird", s, sci, old, RuntimeError("no taxonomy"), now)
            else:
                self._miss("ebird", s, sci, now)
            return
        write_rec(self.cfg.dir, "ebird", s, "ok", now, {"code": m[0], "via": m[1]})
        if self.cfg.nearby_on and needs_fetch(self._rec("nearby", s), now, "nearby"):
            self._start(("nearby", s), lambda: self._nearby_job(sci, m[0], now))

    def _nearby_job(self, sci: str, code: str, now: dt.datetime) -> None:
        s = frame.stem(sci)
        old = self._rec("nearby", s)
        try:
            body = self._get(nearby_url(code, self.cfg.lat, self.cfg.lon),
                             headers=_ebird_headers(self.cfg.ebird_key))
            write_rec(self.cfg.dir, "nearby", s, "ok", now, parse_nearby(body))
        except Exception as exc:  # noqa: BLE001
            self._fail("nearby", s, sci, old, exc, now)

    # -- the entry point
    def lookup(self, sci: str, common: str, now: dt.datetime, budget: float = FACTS_BUDGET,
               tz: dt.tzinfo = dt.timezone.utc) -> FactsResult:
        """Cached facts, after starting (and briefly waiting for) any fetch
        that is due. Only validated binomials may reach this."""
        assert BINOMIAL_RE.fullmatch(sci), "facts lookup needs a validated binomial"
        s = frame.stem(sci)
        cfg = self.cfg
        if cfg.fetch:
            deadline = time.monotonic() + budget
            evs = []
            if (needs_fetch(self._rec("wikidata", s), now, "wikidata")
                    or needs_fetch(self._rec("wikipedia", s), now, "wikipedia")):
                evs.append(self._start(("wiki", s), lambda: self._wiki_job(sci, common, now)))
            if cfg.ebird_key is not None:
                eb = self._rec("ebird", s)
                if needs_fetch(eb, now, "ebird"):
                    evs.append(self._start(("ebird", s), lambda: self._ebird_job(sci, common, now)))
                elif _usable(eb) and cfg.nearby_on and needs_fetch(self._rec("nearby", s), now, "nearby"):
                    code = eb["data"]["code"]
                    self._start(("nearby", s), lambda: self._nearby_job(sci, code, now))
            for ev in evs:
                ev.wait(max(0.0, deadline - time.monotonic()))
            nb = self._running("nearby", s)
            if nb is not None:
                nb.wait(max(0.0, deadline - time.monotonic()))
        recs = {src: self._rec(src, s) for src in SOURCES}
        pending: list[str] = []
        if cfg.fetch:
            jobs = {"wikidata": "wiki", "wikipedia": "wiki", "ebird": "ebird", "nearby": "nearby"}
            for src in SOURCES:
                if src in ("ebird", "nearby") and cfg.ebird_key is None:
                    continue
                if src == "nearby" and not cfg.nearby_on:
                    continue
                if self._running(jobs[src], s) is not None and not _usable(recs[src]):
                    pending.append(src)
        return FactsResult(build_facts(recs, cfg, now, tz), pending)


def build_facts(recs: Mapping[str, dict | None], cfg: FactsConfig, now: dt.datetime,
                tz: dt.tzinfo) -> dict:
    """The response's `facts` object, field by field from cached data only."""
    out: dict = {}
    wp, wd, eb, nb = (recs.get("wikipedia"), recs.get("wikidata"), recs.get("ebird"), recs.get("nearby"))
    if _usable(wp):
        d = wp["data"]
        out["wikipedia"] = {"title": d["title"], "extract": d["extract"], "trimmed": bool(d["trimmed"]),
                            "url": d["url"], "license": "CC BY-SA 4.0", "license_url": CC_BY_SA}
    wdd = wd["data"] if _usable(wd) else None
    if wdd:
        sizes = wdd.get("sizes") or {}
        size = {"mass": format_mass(sizes.get("mass")), "length": format_length(sizes.get("length")),
                "wingspan": format_length(sizes.get("wingspan"))}
        if any(size.values()) and QID_RE.fullmatch(str(wdd.get("qid", ""))):
            out["size"] = {**size, "url": WIKIDATA_PAGE + wdd["qid"], "license": "CC0"}
    code = None
    if cfg.ebird_key is not None and _usable(eb):
        code = eb["data"].get("code")
    if code is None and wdd:
        code = wdd.get("ebird_code")
    if isinstance(code, str) and CODE_RE.fullmatch(code):
        out["ebird"] = {"code": code, "url": EBIRD_SPECIES + code}
        if cfg.nearby_on and _usable(nb):
            d = nb["data"]
            near = {"reported": bool(d.get("reported")), "dist_km": NEARBY_DIST_KM,
                    "back_days": NEARBY_BACK_DAYS, "url": EBIRD_SPECIES + code}
            if near["reported"] and d.get("last_obs_date"):
                last = dt.date.fromisoformat(d["last_obs_date"])
                near["days_ago"] = max(0, (now.astimezone(tz).date() - last).days)
            out["nearby"] = near
    return out
