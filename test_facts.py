"""Unit tests for facts.py. No network: an autouse fixture makes facts.http_get
and frame.fetch_url raise; tests install fake upstreams (URL prefix -> bytes
or exception) that log every call. Fixtures are hand-built JSON shaped like
the live Wikidata, Wikipedia and eBird responses."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import urllib.error
from pathlib import Path

import pytest

import facts
import frame

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 3, 19, 0, tzinfo=UTC)
REAL_HTTP_GET = facts.http_get        # captured before no_network replaces it
KEY = "SENTINELKEY123"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, *a, **kw):
        raise AssertionError(f"network call in a test: {url}")
    monkeypatch.setattr(facts, "http_get", refuse)
    monkeypatch.setattr(frame, "fetch_url", refuse)


class Upstream:
    """Fake facts.http_get: the first matching URL prefix answers."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, url, headers=None, timeout=None, max_bytes=None):
        with self._lock:
            self.calls.append((url, dict(headers or {})))
        for prefix, v in self.routes.items():
            if url.startswith(prefix):
                if callable(v):
                    v = v(url)
                if isinstance(v, BaseException):
                    raise v
                return v
        raise frame.NotFound(url)

    def count(self, prefix: str) -> int:
        return sum(1 for u, _ in self.calls if u.startswith(prefix))


# ----------------------------------------------------------------- fixture builders
def _snak(value):
    return {"snaktype": "value", "datavalue": {"value": value}}


def p225(name):
    return {"mainsnak": _snak(name), "rank": "normal"}


def qty(amount, unit, rank="normal", **quals):
    c = {"mainsnak": _snak({"amount": amount, "unit": f"http://www.wikidata.org/entity/{unit}"}), "rank": rank}
    if quals:
        c["qualifiers"] = {p: [{"datavalue": {"value": {"id": q}}} for q in qs] for p, qs in quals.items()}
    return c


def entity(qid, sci, enwiki=None, mass=(), length=(), wingspan=(), code=None):
    claims = {"P225": [p225(sci)]}
    if mass:
        claims["P2067"] = list(mass)
    if length:
        claims["P2043"] = list(length)
    if wingspan:
        claims["P2050"] = list(wingspan)
    if code:
        claims["P3444"] = [{"mainsnak": _snak(code), "rank": "normal"}]
    return {"id": qid, "claims": claims,
            "sitelinks": {"enwiki": {"site": "enwiki", "title": enwiki}} if enwiki else {}}


def search_body(*qids):
    return json.dumps({"query": {"search": [{"ns": 0, "title": q} for q in qids]}}).encode()


def entities_body(*ents):
    return json.dumps({"entities": {e["id"]: e for e in ents}}).encode()


def summary_body(title, extract, type_="standard", url=None):
    url = url or "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return json.dumps({"type": type_, "title": title.replace(" ", "_"),
                       "titles": {"normalized": title}, "extract": extract,
                       "content_urls": {"desktop": {"page": url}}}).encode()


SEARCH = facts.WIKIDATA_API + "action=query"
ENTITIES = facts.WIKIDATA_API + "action=wbgetentities"
SUMMARY = facts.SUMMARY_API
BUSHTIT = entity("Q2746307", "Psaltriparus minimus", "American bushtit",
                 mass=[qty("+5.3", "Q41803")], code="bushti")
BUSHTIT_TEXT = "The American bushtit is a tiny songbird. It lives in the west."


def bushtit_routes(**over):
    routes = {SEARCH: search_body("Q2746307"), ENTITIES: entities_body(BUSHTIT),
              SUMMARY + "American_bushtit": summary_body("American bushtit", BUSHTIT_TEXT)}
    routes.update(over)
    return routes


def taxonomy_body(n=5200, extra=()):
    recs = [{"sciName": f"Genus s{i}", "comName": f"Bird {i}", "speciesCode": f"code{i}",
             "category": "species"} for i in range(n)]
    recs += [{"sciName": s, "comName": c, "speciesCode": k, "category": "species"} for s, c, k in extra]
    return json.dumps(recs).encode()


def make(tmp_path, monkeypatch, routes, **cfg) -> tuple[facts.Facts, Upstream]:
    up = Upstream(routes)
    monkeypatch.setattr(facts, "http_get", up)
    cfg.setdefault("sizes", None)        # the shipped table is tested on its own below
    return facts.Facts(facts.FactsConfig(tmp_path / "facts", **cfg)), up


def settle(f: facts.Facts, timeout=5.0):
    """Wait until no background job is running."""
    end = time.monotonic() + timeout
    while f._inflight and time.monotonic() < end:
        time.sleep(0.01)
    assert not f._inflight


# ----------------------------------------------------------------- http_get
def test_http_get_user_agent_cap_and_404(monkeypatch):
    seen = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return self.data[:n]

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("User-agent"), req.get_header("Accept"),
                     req.get_header("X-test"), timeout))
        if "missing" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
        if "boom" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return Resp(b"x" * (11 if "big" in req.full_url else 10))

    monkeypatch.setattr(facts.urllib.request, "urlopen", fake_urlopen)
    assert REAL_HTTP_GET("https://a/ok", {"X-Test": "1"}, max_bytes=10) == b"x" * 10
    assert seen[0] == ("https://a/ok", frame.USER_AGENT, "application/json", "1", facts.FACT_TIMEOUT)
    with pytest.raises(ValueError, match="too large"):
        REAL_HTTP_GET("https://a/big", max_bytes=10)
    with pytest.raises(frame.NotFound):
        REAL_HTTP_GET("https://a/missing")
    with pytest.raises(urllib.error.HTTPError):
        REAL_HTTP_GET("https://a/boom")


# ----------------------------------------------------------------- #18 wikidata resolution
def test_pick_entity_exact_p225_and_first_with_enwiki():
    wrong = entity("Q1", "Psaltriparus minimus minimus", "Subspecies")
    no_link = entity("Q2", "Psaltriparus minimus")
    body = entities_body(wrong, no_link, BUSHTIT)
    q, e = facts.pick_entity(body, "Psaltriparus minimus", ["Q1", "Q2", "Q2746307"])
    assert q == "Q2746307" and e["sitelinks"]["enwiki"]["title"] == "American bushtit"
    q, _ = facts.pick_entity(entities_body(wrong, no_link), "Psaltriparus minimus", ["Q1", "Q2"])
    assert q == "Q2"                               # exact match without a sitelink beats none
    assert facts.pick_entity(entities_body(wrong), "Psaltriparus minimus", ["Q1"]) is None
    assert facts.parse_search(search_body("Q5", "Property:P1", "Q7")) == ["Q5", "Q7"]
    url = facts.search_url("Psaltriparus minimus")
    assert url.startswith(SEARCH) and "haswbstatement%3A%22P225%3DPsaltriparus+minimus%22" in url


def test_wiki_job_resolves_bushtit(tmp_path, monkeypatch):
    f, up = make(tmp_path, monkeypatch, bushtit_routes())
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert r.pending == []
    w = r.facts["wikipedia"]
    assert w == {"title": "American bushtit", "extract": BUSHTIT_TEXT, "trimmed": False,
                 "url": "https://en.wikipedia.org/wiki/American_bushtit",
                 "license": "CC BY-SA 4.0", "license_url": facts.CC_BY_SA}
    assert r.facts["size"] == {"mass": "0.2 oz (5.3 g)", "length": None, "wingspan": None, "sources": [
        {"name": "Wikidata", "url": "https://www.wikidata.org/wiki/Q2746307", "license": "CC0",
         "license_url": facts.CC0, "fields": ["mass"]}]}
    assert r.facts["ebird"] == {"code": "bushti", "url": "https://ebird.org/species/bushti"}   # P3444, no key
    assert "nearby" not in r.facts
    for url, headers in up.calls:
        assert "X-eBirdApiToken" not in headers and "ebird.org" not in url


def test_wiki_redirected_title_is_credited(tmp_path, monkeypatch):
    """Glaucidium gnoma: the sitelink redirects; the card credits what the summary returned."""
    owl = entity("Q1034956", "Glaucidium gnoma", "Mountain pygmy owl")
    routes = {SEARCH: search_body("Q1034956"), ENTITIES: entities_body(owl),
              SUMMARY + "Mountain_pygmy_owl": summary_body("Northern pygmy owl", "A small owl.")}
    f, _ = make(tmp_path, monkeypatch, routes)
    w = f.lookup("Glaucidium gnoma", "Northern Pygmy-Owl", NOW).facts["wikipedia"]
    assert w["title"] == "Northern pygmy owl" and w["url"].endswith("/Northern_pygmy_owl")


def test_wiki_falls_back_to_sci_then_common(tmp_path, monkeypatch):
    routes = {SEARCH: search_body(), SUMMARY + "Bird_name": summary_body("Bird name", "Common one.")}
    f, up = make(tmp_path, monkeypatch, routes)
    r = f.lookup("Genus species", "Bird name", NOW)
    assert r.facts["wikipedia"]["title"] == "Bird name"
    assert [u for u, _ in up.calls if u.startswith(SUMMARY)] == [SUMMARY + "Genus_species", SUMMARY + "Bird_name"]
    assert up.count(ENTITIES) == 0 and "size" not in r.facts and "ebird" not in r.facts


# ----------------------------------------------------------------- #19 summary
def test_parse_summary_rules():
    d = facts.parse_summary(summary_body("American bushtit", BUSHTIT_TEXT))
    assert d["title"] == "American bushtit" and d["trimmed"] is False
    assert facts.parse_summary(summary_body("Jay", "x", type_="disambiguation")) is None
    assert facts.parse_summary(summary_body("Jay", "x", url="http://en.wikipedia.org/wiki/Jay")) is None
    assert facts.parse_summary(summary_body("Jay", "x", url="https://evil.example/wiki/Jay")) is None
    assert facts.parse_summary(summary_body("Jay", "   ")) is None
    with pytest.raises(ValueError):
        facts.parse_summary(b"not json")


def test_trim_extract_at_sentence():
    s = "A" * 300 + ". " + "B" * 250 + ". " + "C" * 200 + "."
    text, trimmed = facts.trim_extract(s)
    assert trimmed and text == "A" * 300 + ". " + "B" * 250 + "." and len(text) <= 600
    long_one = "D" * 700 + ". Next."
    assert facts.trim_extract(long_one) == ("D" * 700 + ".", True)
    assert facts.trim_extract("short.") == ("short.", False)


def test_summary_404_and_non_standard_are_misses(tmp_path, monkeypatch):
    routes = bushtit_routes(**{SUMMARY + "American_bushtit": frame.NotFound("404"),
                               SUMMARY + "Psaltriparus_minimus": summary_body("x", "y", type_="disambiguation")})
    f, _ = make(tmp_path, monkeypatch, routes)
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert "wikipedia" not in r.facts and "size" in r.facts
    assert facts.read_rec(tmp_path / "facts", "wikipedia", "psaltriparus-minimus")["status"] == "miss"


# ----------------------------------------------------------------- #20 sizes
def test_parse_sizes_units_filters_and_rank():
    claims = {
        "P2067": [qty("+0.56", "Q41803", P3831=["Q4128476"]),         # birth weight: dropped
                  qty("+4.3", "Q41803", P3831=["Q78101716"]),         # adult weight: kept
                  qty("+3", "Q41803", P518=["Q1130"]),                # hatchling part: dropped
                  qty("+9", "Q41803", rank="deprecated")],
        "P2043": [qty("+90", "Q174789")],                               # 90 mm = 9 cm
        "P2050": [qty("+1.2", "Q11573"), qty("+5", "Q12345")],          # m kept, unknown unit dropped
    }
    assert facts.parse_sizes(claims) == {"mass": [4.3, 4.3], "length": [9.0, 9.0], "wingspan": [120.0, 120.0]}
    raven = {"P2067": [qty("+689", "Q41803", rank="preferred"), qty("+1625", "Q41803", rank="preferred"),
                       qty("+800", "Q41803"), qty("+1.5", "Q11570")]}
    assert facts.parse_sizes(raven)["mass"] == [689.0, 1625.0]
    assert facts.parse_sizes({"P2067": [qty("+1.5", "Q11570")]})["mass"] == [1500.0, 1500.0]   # kg
    assert facts.parse_sizes({"P2067": [qty("+50000", "Q41803")], "P2043": [qty("x", "Q174728")]}) == \
        {"mass": None, "length": None, "wingspan": None}
    assert facts.parse_sizes({"P2043": [qty("+20", "Q174728")]})["length"] == [20.0, 20.0]   # cm


def test_size_formatting():
    assert facts.format_mass([32, 32]) == "1.1 oz (32 g)"
    assert facts.format_mass([5.3, 5.3]) == "0.2 oz (5.3 g)"
    assert facts.format_mass([1134, 1134]) == "2.5 lb (1.1 kg)"
    assert facts.format_mass([10300, 11400]) == "23\u201325 lb (10.3\u201311.4 kg)"
    assert facts.format_mass([240, 260]) == "8.5\u20139.2 oz (240\u2013260 g)"
    assert facts.format_mass([100, 104]) == "3.6 oz (102 g)"        # max/min < 1.05: one value
    assert facts.format_length([119, 119]) == "47 in (119 cm)"
    assert facts.format_length([16, 16]) == "6.3 in (16 cm)"
    assert facts.format_mass(None) is None and facts.format_length(None) is None


# ----------------------------------------------------------------- #21 ebird
def test_match_code_order():
    tax = {"sci": {"aphelocoma californica": "cowscj1"}, "com": {"bushtit": "bushti"}}
    assert facts.match_code(tax, "Aphelocoma californica", "California Scrub-Jay", "zzz") == ("cowscj1", "taxonomy")
    assert facts.match_code(tax, "Psaltriparus minimus", "Bushtit", None) == ("bushti", "taxonomy")
    assert facts.match_code(tax, "Genus x", "None", "abc1") == ("abc1", "wikidata")
    assert facts.match_code(tax, "Genus x", "None", "BAD CODE") is None
    assert facts.match_code(None, "Genus x", "None", None) is None


def test_slim_taxonomy():
    tax = facts.slim_taxonomy(taxonomy_body(extra=[("Psaltriparus minimus", "Bushtit", "bushti")]), NOW)
    assert tax["count"] == 5201 and tax["sci"]["psaltriparus minimus"] == "bushti"
    assert tax["com"]["bushtit"] == "bushti" and tax["fetched_at"] == "2026-10-03T19:00:00+00:00"
    with pytest.raises(ValueError):
        facts.slim_taxonomy(taxonomy_body(n=100), NOW)


def test_no_key_means_no_ebird_requests(tmp_path, monkeypatch):
    f, up = make(tmp_path, monkeypatch, bushtit_routes(), lat=47.6, lon=-122.3)
    f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    settle(f)
    assert up.calls and all("ebird.org" not in u and "X-eBirdApiToken" not in h for u, h in up.calls)
    assert not (tmp_path / "facts" / "v1" / "nearby").exists()


def test_key_sent_as_header_only(tmp_path, monkeypatch):
    nearby = json.dumps([{"obsDt": "2026-10-01 07:30", "locName": "secret yard", "howMany": 2}]).encode()
    routes = bushtit_routes(**{facts.EBIRD_TAXONOMY: taxonomy_body(extra=[("Psaltriparus minimus", "Bushtit", "bushti")]),
                               facts.EBIRD_NEARBY: nearby})
    f, up = make(tmp_path, monkeypatch, routes, ebird_key=facts.Secret(KEY), lat=47.61234, lon=-122.33333)
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert r.facts["ebird"]["code"] == "bushti"
    assert r.facts["nearby"] == {"reported": True, "days_ago": 2, "dist_km": 25, "back_days": 7,
                                 "url": "https://ebird.org/species/bushti"}
    ebird_calls = [(u, h) for u, h in up.calls if "ebird.org" in u]
    assert len(ebird_calls) == 2
    for u, h in ebird_calls:
        assert h["X-eBirdApiToken"] == KEY and KEY not in u
    assert any("lat=47.61&lng=-122.33&dist=25&back=7" in u for u, _ in ebird_calls)
    for u, h in up.calls:
        if "ebird.org" not in u:
            assert "X-eBirdApiToken" not in h
    for p in (tmp_path / "facts").rglob("*"):
        if p.is_file():
            text = p.read_text()
            assert KEY not in text and "secret yard" not in text
    assert (tmp_path / "facts" / "ebird-taxonomy.json").exists()


# ----------------------------------------------------------------- #21a wiki job mapping
def test_wiki_job_refetches_only_stale_source(tmp_path, monkeypatch):
    d = tmp_path / "facts"
    s = "psaltriparus-minimus"
    facts.write_rec(d, "wikidata", s, "ok", NOW - dt.timedelta(days=1),
                    {"qid": "Q2746307", "enwiki": "American bushtit", "sizes": {}, "ebird_code": "bushti"})
    facts.write_rec(d, "wikipedia", s, "error", NOW - dt.timedelta(hours=2), None, "TimeoutError")
    f, up = make(tmp_path, monkeypatch, bushtit_routes())
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert up.count(SUMMARY) == 1 and up.count(facts.WIKIDATA_API) == 0
    assert r.facts["wikipedia"]["title"] == "American bushtit"
    # Both stale: one of each.
    later = NOW + dt.timedelta(days=40)
    f.lookup("Psaltriparus minimus", "Bushtit", later)
    assert up.count(SEARCH) == 1 and up.count(ENTITIES) == 1 and up.count(SUMMARY) == 2


def test_pending_names_sources_while_wiki_job_runs(tmp_path, monkeypatch):
    gate = threading.Event()
    d = tmp_path / "facts"
    s = "psaltriparus-minimus"

    def slow(url):
        gate.wait(5)
        return summary_body("American bushtit", BUSHTIT_TEXT)

    facts.write_rec(d, "wikidata", s, "ok", NOW, {"qid": "Q2746307", "enwiki": "American bushtit",
                                                    "sizes": {}, "ebird_code": None})
    f, up = make(tmp_path, monkeypatch, bushtit_routes(**{SUMMARY + "American_bushtit": slow}))
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW, budget=0.2)
    assert r.pending == ["wikipedia"]                 # wikidata is cached, so not pending
    gate.set()
    settle(f)
    # Nothing cached and Wikidata still running: both wiki sources pending.
    gate.clear()
    slow_search = lambda url: (gate.wait(5), search_body("Q2746307"))[1]  # noqa: E731
    f2, _ = make(tmp_path / "b", monkeypatch, bushtit_routes(**{SEARCH: slow_search}))
    assert f2.lookup("Psaltriparus minimus", "Bushtit", NOW, budget=0.2).pending == ["wikidata", "wikipedia"]
    gate.set()
    settle(f2)


# ----------------------------------------------------------------- #22 nearby
def test_parse_nearby():
    body = json.dumps([{"obsDt": "2026-09-29 08:00"}, {"obsDt": "2026-10-01"}, {"obsDt": "bad"}]).encode()
    assert facts.parse_nearby(body) == {"reported": True, "last_obs_date": "2026-10-01"}
    assert facts.parse_nearby(b"[]") == {"reported": False, "last_obs_date": None}
    with pytest.raises(ValueError):
        facts.parse_nearby(b'{"errors": []}')
    url = facts.nearby_url("bushti", 47.61234, -122.33333)
    assert url == facts.EBIRD_NEARBY + "bushti?lat=47.61&lng=-122.33&dist=25&back=7"


def test_nearby_none_reported_and_no_coordinates(tmp_path, monkeypatch):
    routes = bushtit_routes(**{facts.EBIRD_TAXONOMY: taxonomy_body(extra=[("Psaltriparus minimus", "Bushtit", "bushti")]),
                               facts.EBIRD_NEARBY: b"[]"})
    f, _ = make(tmp_path, monkeypatch, routes, ebird_key=facts.Secret(KEY), lat=47.6, lon=-122.3)
    near = f.lookup("Psaltriparus minimus", "Bushtit", NOW).facts["nearby"]
    assert near["reported"] is False and "days_ago" not in near
    f2, up2 = make(tmp_path / "b", monkeypatch, routes, ebird_key=facts.Secret(KEY))
    r = f2.lookup("Psaltriparus minimus", "Bushtit", NOW)
    settle(f2)
    assert "nearby" not in r.facts and up2.count(facts.EBIRD_NEARBY) == 0


def test_cached_code_starts_nearby_directly(tmp_path, monkeypatch):
    d = tmp_path / "facts"
    s = "psaltriparus-minimus"
    facts.write_rec(d, "ebird", s, "ok", NOW, {"code": "bushti", "via": "taxonomy"})
    routes = bushtit_routes(**{facts.EBIRD_NEARBY: json.dumps([{"obsDt": "2026-10-03 06:00"}]).encode()})
    f, up = make(tmp_path, monkeypatch, routes, ebird_key=facts.Secret(KEY), lat=47.6, lon=-122.3)
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert r.facts["nearby"]["days_ago"] == 0
    assert up.count(facts.EBIRD_TAXONOMY) == 0 and up.count(facts.EBIRD_NEARBY) == 1


def test_days_ago_uses_local_zone(tmp_path, monkeypatch):
    from zoneinfo import ZoneInfo
    d = tmp_path / "facts"
    s = "psaltriparus-minimus"
    facts.write_rec(d, "ebird", s, "ok", NOW, {"code": "bushti", "via": "taxonomy"})
    facts.write_rec(d, "nearby", s, "ok", NOW, {"reported": True, "last_obs_date": "2026-10-03"})
    f, _ = make(tmp_path, monkeypatch, bushtit_routes(), ebird_key=facts.Secret(KEY), lat=47.6, lon=-122.3)
    late = dt.datetime(2026, 10, 4, 3, 0, tzinfo=UTC)      # 8 PM Oct 3 in Seattle
    assert f.lookup("Psaltriparus minimus", "Bushtit", late, tz=ZoneInfo("America/Los_Angeles")).facts["nearby"]["days_ago"] == 0
    assert f.lookup("Psaltriparus minimus", "Bushtit", late).facts["nearby"]["days_ago"] == 1


# ----------------------------------------------------------------- #23 all about birds
def test_aab_url():
    assert facts.aab_url("Townsend's Solitaire") == "https://www.allaboutbirds.org/guide/Townsends_Solitaire"
    assert facts.aab_url("Black-capped Chickadee") == "https://www.allaboutbirds.org/guide/Black-capped_Chickadee"
    assert facts.aab_url("Steller\u2019s Jay") == "https://www.allaboutbirds.org/guide/Stellers_Jay"
    assert facts.aab_url("A/B?c") == "https://www.allaboutbirds.org/guide/A%2FB%3Fc"


# ----------------------------------------------------------------- #24 cache
def test_needs_fetch_table():
    def rec(status, at, **kw):
        return {"status": status, "at": at.isoformat(), **kw}
    h, d = dt.timedelta(hours=1), dt.timedelta(days=1)
    assert facts.needs_fetch(None, NOW, "wikidata")
    assert not facts.needs_fetch(rec("ok", NOW - 29 * d), NOW, "wikidata")
    assert facts.needs_fetch(rec("ok", NOW - 30 * d), NOW, "wikidata")
    assert not facts.needs_fetch(rec("ok", NOW - 5 * h), NOW, "nearby")
    assert facts.needs_fetch(rec("ok", NOW - 6 * h), NOW, "nearby")
    assert not facts.needs_fetch(rec("ok", NOW - 40 * d, error_at=(NOW - 0.5 * h).isoformat()), NOW, "wikipedia")
    assert facts.needs_fetch(rec("ok", NOW - 40 * d, error_at=(NOW - h).isoformat()), NOW, "wikipedia")
    assert not facts.needs_fetch(rec("miss", NOW - 23 * h), NOW, "wikipedia")
    assert facts.needs_fetch(rec("miss", NOW - 24 * h), NOW, "wikipedia")
    assert facts.needs_fetch(rec("miss", NOW - 6 * h), NOW, "nearby")
    assert not facts.needs_fetch(rec("error", NOW - 0.9 * h), NOW, "ebird")
    assert facts.needs_fetch(rec("error", NOW - h), NOW, "ebird")


def test_second_lookup_makes_no_call(tmp_path, monkeypatch):
    f, up = make(tmp_path, monkeypatch, bushtit_routes())
    a = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    n = len(up.calls)
    assert n == 3
    b = f.lookup("Psaltriparus minimus", "Bushtit", NOW + dt.timedelta(days=29))
    assert len(up.calls) == n and a.facts == b.facts
    # A new Facts (a restart) reads the same cache.
    f2, up2 = make(tmp_path, monkeypatch, {})
    assert f2.lookup("Psaltriparus minimus", "Bushtit", NOW).facts == a.facts and up2.calls == []


def test_404_waits_a_day_and_5xx_an_hour(tmp_path, monkeypatch):
    f, up = make(tmp_path, monkeypatch, {SEARCH: search_body(), SUMMARY: frame.NotFound("404")})
    f.lookup("Genus species", "Bird", NOW)
    n = len(up.calls)
    f.lookup("Genus species", "Bird", NOW + dt.timedelta(hours=23))
    assert len(up.calls) == n
    f.lookup("Genus species", "Bird", NOW + dt.timedelta(hours=24))
    assert len(up.calls) > n

    boom = urllib.error.HTTPError("u", 503, "busy", {}, None)
    f, up = make(tmp_path / "b", monkeypatch, {SEARCH: boom, SUMMARY: frame.NotFound("404")})
    f.lookup("Genus species", "Bird", NOW)
    rec = facts.read_rec(tmp_path / "b" / "facts", "wikidata", "genus-species")
    assert rec["status"] == "error" and rec["reason"] == "HTTPError 503"
    n = up.count(SEARCH)
    f.lookup("Genus species", "Bird", NOW + dt.timedelta(minutes=59))
    assert up.count(SEARCH) == n
    f.lookup("Genus species", "Bird", NOW + dt.timedelta(minutes=61))
    assert up.count(SEARCH) == n + 1


def test_stale_ok_served_on_failure_and_30_day_refetch(tmp_path, monkeypatch, caplog):
    f, up = make(tmp_path, monkeypatch, bushtit_routes())
    good = f.lookup("Psaltriparus minimus", "Bushtit", NOW).facts
    up.routes = {SEARCH: TimeoutError("slow"), SUMMARY: TimeoutError("slow")}
    later = NOW + dt.timedelta(days=31)
    with caplog.at_level("WARNING", logger="facts"):
        stale = f.lookup("Psaltriparus minimus", "Bushtit", later)
    assert stale.facts == good and stale.pending == []
    assert "facts: wikidata for Psaltriparus minimus failed: TimeoutError" in caplog.text
    rec = facts.read_rec(tmp_path / "facts", "wikidata", "psaltriparus-minimus")
    assert rec["status"] == "ok" and rec["error_at"] == later.isoformat(timespec="seconds")
    n = len(up.calls)
    f.lookup("Psaltriparus minimus", "Bushtit", later + dt.timedelta(minutes=30))
    assert len(up.calls) == n                       # error_at holds retries for an hour


def test_corrupt_file_deleted_with_one_warning(tmp_path, caplog):
    d = tmp_path / "facts"
    p = facts.rec_path(d, "wikipedia", "genus-species")
    p.parent.mkdir(parents=True)
    for bad in (b"{not json", b'{"status": "ok"}', b'{"status": "weird", "at": "2026-01-01T00:00:00+00:00"}'):
        p.write_bytes(bad)
        caplog.clear()
        with caplog.at_level("WARNING", logger="facts"):
            assert facts.read_rec(d, "wikipedia", "genus-species") is None
        assert not p.exists() and len(caplog.records) == 1


def test_fetch_off_reads_cache_only(tmp_path, monkeypatch):
    f, up = make(tmp_path, monkeypatch, bushtit_routes())
    good = f.lookup("Psaltriparus minimus", "Bushtit", NOW).facts
    off, up2 = make(tmp_path, monkeypatch, bushtit_routes(), fetch=False)
    r = off.lookup("Psaltriparus minimus", "Bushtit", NOW + dt.timedelta(days=90))
    assert r.facts == good and r.pending == [] and up2.calls == []
    r = off.lookup("Turdus migratorius", "American Robin", NOW)
    assert r.facts == {} and r.pending == [] and up2.calls == []


def test_lookup_requires_binomial(tmp_path, monkeypatch):
    f, _ = make(tmp_path, monkeypatch, {})
    for bad in ("Dog", "../etc passwd", "turdus migratorius"):
        with pytest.raises(AssertionError):
            f.lookup(bad, "x", NOW)


# ----------------------------------------------------------------- #25 budget and single flight
def test_budget_returns_pending_then_facts(tmp_path, monkeypatch):
    gate = threading.Event()

    def blocked(body):
        return lambda url: (gate.wait(10), body)[1]

    routes = {SEARCH: blocked(search_body("Q2746307")), ENTITIES: entities_body(BUSHTIT),
              SUMMARY: summary_body("American bushtit", BUSHTIT_TEXT)}
    f, up = make(tmp_path, monkeypatch, routes)
    t0 = time.monotonic()
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW, budget=0.5)
    assert time.monotonic() - t0 < 0.5 + 0.5
    assert r.facts == {} and r.pending == ["wikidata", "wikipedia"]
    gate.set()
    settle(f)
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert r.pending == [] and r.facts["wikipedia"]["title"] == "American bushtit"
    for url, headers in up.calls:
        assert url.startswith(("https://www.wikidata.org/", "https://en.wikipedia.org/"))


def test_concurrent_lookups_single_flight(tmp_path, monkeypatch):
    gate = threading.Event()

    def blocked(body):
        return lambda url: (gate.wait(10), body)[1]

    routes = bushtit_routes(**{SEARCH: blocked(search_body("Q2746307"))})
    f, up = make(tmp_path, monkeypatch, routes)
    out = []
    threads = [threading.Thread(target=lambda: out.append(f.lookup("Psaltriparus minimus", "Bushtit", NOW, budget=0.3)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gate.set()
    settle(f)
    assert up.count(SEARCH) == 1 and up.count(ENTITIES) == 1 and up.count(SUMMARY) == 1
    assert all(r.pending for r in out)


def test_user_agent_on_wikimedia_calls(tmp_path, monkeypatch):
    """facts.http_get always sends frame.USER_AGENT; check the real function builds it."""
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        seen.append(req.get_header("User-agent"))
        return Resp()

    monkeypatch.setattr(facts.urllib.request, "urlopen", fake_urlopen)
    REAL_HTTP_GET(facts.search_url("Psaltriparus minimus"))
    REAL_HTTP_GET(facts.summary_url("American bushtit"))
    assert seen == [frame.USER_AGENT, frame.USER_AGENT]


# ----------------------------------------------------------------- #26 the collage never fetches facts
def test_render_never_calls_facts(tmp_path, monkeypatch):
    def loud(*a, **kw):
        raise AssertionError("render reached facts")
    monkeypatch.setattr(facts, "http_get", loud)
    cache = frame.RenderCache(frame.Artwork(tmp_path / "art"))
    sp = frame.Species("Psaltriparus minimus", "Bushtit", "2026-10-03T18:00:00+00:00", 1, ("back",), False)
    assert cache.get([sp], 400, 300, 24, now=NOW)[:8] == b"\x89PNG\r\n\x1a\n"
    import ast
    tree = ast.parse(Path(frame.__file__).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in getattr(n, "names", [])} | {getattr(n, "module", None) for n in ast.walk(tree)
                                                      if isinstance(n, ast.ImportFrom)}
    assert "facts" not in imported and "serve" not in imported


# ----------------------------------------------------------------- #5 size table
def _coverage_species() -> list[str]:
    """The species tools/facts_coverage.py measures (the issue's acceptance list)."""
    import re
    src = (Path(facts.__file__).parent / "tools" / "facts_coverage.py").read_text()
    return re.findall(r'\("([A-Z][a-z]+ [a-z]+)", "', src.split("NAME_RE")[0])


def test_shipped_size_table_loads_and_is_credited():
    t = facts.SizeTable.load()
    assert t is not None and len(t.masses) > 6000
    assert t.name == "AVONET" and t.license == "CC BY 4.0"
    assert t.url == "https://doi.org/10.1111/ele.13898" and t.license_url.startswith("https://creativecommons.org/")
    assert t.masses["Psaltriparus minimus"] == 5.3 and t.masses["Corvus brachyrhynchos"] == 448.8
    # #5 acceptance: >= 90% of the coverage list get a mass line from the table alone.
    sp = _coverage_species()
    assert len(sp) >= 25
    have = [s for s in sp if s in t.masses]
    assert len(have) / len(sp) >= 0.9, sorted(set(sp) - set(have))


def test_shipped_size_table_has_birds_only():
    import taxa
    t = facts.SizeTable.load()
    assert all(taxa.is_bird(s) for s in t.masses)


def _table(tmp_path, species, **source):
    src = {"name": "T", "url": "https://example.org/t", "license": "CC BY 4.0",
           "license_url": "https://creativecommons.org/licenses/by/4.0/", **source}
    p = tmp_path / "sizes.json"
    p.write_text(json.dumps({"source": src, "species": species}))
    return p


def test_size_table_rejects_bad_files_and_rows(tmp_path, caplog):
    assert facts.SizeTable.load(tmp_path / "missing.json") is None
    (tmp_path / "junk.json").write_text("not json")
    assert facts.SizeTable.load(tmp_path / "junk.json") is None
    assert facts.SizeTable.load(_table(tmp_path, {}, url="http://example.org/t")) is None
    assert facts.SizeTable.load(_table(tmp_path, {}, license="")) is None
    assert "Wikidata sizes only" in caplog.text
    t = facts.SizeTable.load(_table(tmp_path, {
        "Turdus migratorius": {"mass_g": 77.3}, "Bad name!": {"mass_g": 5}, "Genus zero": {"mass_g": 0},
        "Genus huge": {"mass_g": 1e9}, "Genus text": {"mass_g": "5"}, "Genus bool": {"mass_g": True},
        "Genus row": 5}))
    assert dict(t.masses) == {"Turdus migratorius": 77.3}


def test_build_size_prefers_wikidata_field_by_field(tmp_path):
    t = facts.SizeTable.load(_table(tmp_path, {"Turdus migratorius": {"mass_g": 77.3}}))
    wd = lambda **s: {"qid": "Q1", "sizes": {"mass": None, "length": None, "wingspan": None, **s}}  # noqa: E731
    sci = "Turdus migratorius"
    # Wikidata has a mass: the table is not used.
    s = facts.build_size(sci, wd(mass=[80.0, 80.0]), t)
    assert s["mass"] == "2.8 oz (80 g)" and [x["name"] for x in s["sources"]] == ["Wikidata"]
    # Wikidata has only a wingspan: the table adds the mass and both are credited.
    s = facts.build_size(sci, wd(wingspan=[31.0, 40.0]), t)
    assert s["mass"] == "2.7 oz (77 g)" and s["wingspan"] == "12\u201316 in (31\u201340 cm)"
    assert [(x["name"], x["fields"]) for x in s["sources"]] == [("Wikidata", ["wingspan"]), ("T", ["mass"])]
    # No Wikidata record, or one with an invalid QID: the table alone.
    for w in (None, {"qid": "bad", "sizes": {"mass": [1.0, 1.0]}}):
        s = facts.build_size(sci, w, t)
        assert s["mass"] == "2.7 oz (77 g)" and s["length"] is None
        assert s["sources"] == [{"name": "T", "url": "https://example.org/t", "license": "CC BY 4.0",
                                 "license_url": "https://creativecommons.org/licenses/by/4.0/", "fields": ["mass"]}]
    # Neither: no size at all; no table configured: Wikidata only.
    assert facts.build_size("Genus species", None, t) is None
    assert facts.build_size(sci, None, None) is None


def test_lookup_uses_table_offline(tmp_path, monkeypatch):
    t = facts.SizeTable.load(_table(tmp_path, {"Psaltriparus minimus": {"mass_g": 5.3}}))
    f, up = make(tmp_path, monkeypatch, {}, fetch=False, sizes=t)
    r = f.lookup("Psaltriparus minimus", "Bushtit", NOW)
    assert up.calls == [] and r.pending == []
    assert r.facts == {"size": {"mass": "0.2 oz (5.3 g)", "length": None, "wingspan": None,
                                "sources": [{**t.credit(), "fields": ["mass"]}]}}
