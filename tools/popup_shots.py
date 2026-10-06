"""Browser checks and screenshots for the pop-up card (manual; not in the image).

  uv run --no-project --python 3.11 --with playwright==1.59.0 --with pillow==12.3.0 python tools/popup_shots.py

Builds a temp DATA_DIR: a db seeded from the live server's /api/recent (or
a fixed list if it is unreachable) plus the approved species, "Dog" (must stay hidden, #10), "Larus sp." (a
name that is not a binomial), an
aged-out species and an XSS species, with detections spread over the day;
the cached art from .agents/artwork; and facts cache records copied from
.agents/facts-cache (written by tools/facts_coverage.py) plus synthetic ones.
FACTS_FETCH=0 and frame.fetch_url refuses, so the run is offline apart from
the one /api/recent call. Starts serve.CollageServer on 127.0.0.1 and drives
it with Playwright Chromium.

Prints one PASS/FAIL line per check; exits 1 on any FAIL, 2 if Chromium
cannot launch. Screenshots go to .agents/shots/ and .agents/popup-*.png.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import random
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageChops  # noqa: E402

import birdlisten as bl  # noqa: E402
import facts  # noqa: E402
import frame  # noqa: E402
import serve  # noqa: E402

AGENTS = ROOT / ".agents"
LIVE = "http://192.168.0.103:8085"
UTC = dt.timezone.utc
XSS_CAM = "<script>alert(1)</script>"
XSS_EXTRACT = 'Plain words <script>alert(1)</script> and "><img onerror=alert(1)> more words.'
APPROVED = [
    ("Psaltriparus minimus", "Bushtit"), ("Ixoreus naevius", "Varied Thrush"),
    ("Cyanocitta stelleri", "Steller's Jay"), ("Aphelocoma californica", "California Scrub-Jay"),
    ("Zonotrichia albicollis", "White-throated Sparrow"), ("Myadestes townsendi", "Townsend's Solitaire"),
    ("Cygnus buccinator", "Trumpeter Swan"), ("Meleagris gallopavo", "Wild Turkey"),
    ("Tachycineta bicolor", "Tree Swallow"), ("Melospiza melodia", "Song Sparrow"),
    ("Poecile atricapillus", "Black-capped Chickadee"), ("Junco hyemalis", "Dark-eyed Junco"),
    ("Calypte anna", "Anna's Hummingbird"), ("Pipilo maculatus", "Spotted Towhee"),
]
FALLBACK_LIVE = [("Corvus brachyrhynchos", "American Crow", 19), ("Haemorhous mexicanus", "House Finch", 4)]
AGED = ("Mergus merganser", "Common Merganser")
PLACEHOLDER = ("Calypte anna", "Anna's Hummingbird")
NON_BINOMIAL = ("Larus sp.", "gull sp.")
XSS = ("Pipilo maculatus", "Spotted Towhee")
ALL_FACTS = ("Cygnus buccinator", "Trumpeter Swan")
NO_FACTS = ("Junco hyemalis", "Dark-eyed Junco")

results: list[tuple[bool, str, str]] = []


def check(cid: str, ok: bool, detail: str = "") -> bool:
    results.append((bool(ok), cid, detail))
    print(f"{'PASS' if ok else 'FAIL'} {cid} {detail}".rstrip(), flush=True)
    return bool(ok)


# ----------------------------------------------------------------- demo data
def live_species() -> list[tuple[str, str, int]]:
    try:
        with urllib.request.urlopen(f"{LIVE}/api/recent?hours=24", timeout=8) as resp:
            data = json.loads(resp.read())
        out = [(s["scientific_name"], s["common_name"], s["count"]) for s in data["species"]]
        print(f"seeded from {LIVE}/api/recent: {len(out)} species")
        return out or FALLBACK_LIVE
    except Exception as exc:  # noqa: BLE001
        print(f"live server unreachable ({type(exc).__name__}); using a fixed list")
        return FALLBACK_LIVE


DAY_WEIGHT = [1, 1, 1, 1, 2, 5, 9, 10, 8, 6, 4, 3, 3, 3, 3, 4, 5, 7, 8, 5, 2, 1, 1, 1]   # dawn and dusk


def spread(rng: random.Random, now: dt.datetime, n: int, tz) -> list[dt.datetime]:
    """n times in the last 24 h, denser at dawn and dusk (local hours)."""
    out = []
    while len(out) < n:
        t = now - dt.timedelta(minutes=rng.randrange(5, 24 * 60 - 5))
        if rng.random() * 10 < DAY_WEIGHT[t.astimezone(tz).hour]:
            out.append(t)
    return out


def seed_db(data_dir: Path, now: dt.datetime, tz) -> list[tuple[str, str]]:
    rng = random.Random(3)
    conn = bl.open_db(data_dir)
    cams = ["doorbell", "garage", "east-rooftop", "west-rooftop"]
    shown: list[tuple[str, str]] = []
    rows = [(sci, com, max(3, n)) for sci, com, n in live_species()]
    rows += [(sci, com, rng.randrange(2, 12)) for sci, com in APPROVED if sci not in {r[0] for r in rows}]
    for sci, com, n in rows:
        for t in spread(rng, now, n, tz):
            cam = XSS_CAM if sci == XSS[0] and rng.random() < 0.5 else rng.choice(cams)
            bl.record(conn, t, bl.Camera(cam, "x"), bl.Detection(com, sci, round(rng.uniform(0.9, 0.99), 3), 0, 3), None)
        shown.append((sci, com))
    bl.record(conn, now - dt.timedelta(minutes=50), bl.Camera("garage", "x"), bl.Detection("Dog", "Dog", 0.93, 0, 3), None)
    bl.record(conn, now - dt.timedelta(minutes=30), bl.Camera("garage", "x"), bl.Detection("Dog", "Dog", 0.95, 0, 3), None)
    for m in (40, 20):
        bl.record(conn, now - dt.timedelta(minutes=m), bl.Camera("garage", "x"),
                  bl.Detection(NON_BINOMIAL[1], NON_BINOMIAL[0], 0.94, 0, 3), None)
    bl.record(conn, now - dt.timedelta(days=3), bl.Camera("garage", "x"), bl.Detection(AGED[1], AGED[0], 0.95, 0, 3), None)
    conn.close()
    return shown


def copy_art(art_dir: Path) -> None:
    src = AGENTS / "artwork"
    for p in (src / "birds").glob("*"):
        if p.suffix in (".webp", ".missing"):
            (art_dir / "birds").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, art_dir / "birds" / p.name)
    for p in (src / "audubon" / frame.VIGNETTE_VERSION).glob("*.webp"):
        (art_dir / "audubon" / frame.VIGNETTE_VERSION).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, art_dir / "audubon" / frame.VIGNETTE_VERSION / p.name)
    for name in frame.META_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, art_dir / name)


def seed_facts(d: Path, now: dt.datetime) -> None:
    cache = AGENTS / "facts-cache"
    if cache.exists():
        shutil.copytree(cache, d, dirs_exist_ok=True)
    else:
        print("no .agents/facts-cache (run tools/facts_coverage.py --keep .agents/facts-cache); facts are synthetic only")
    s = frame.stem(NO_FACTS[0])
    for src in facts.SOURCES:
        facts.rec_path(d, src, s).unlink(missing_ok=True)
    s = frame.stem(XSS[0])
    facts.write_rec(d, "wikipedia", s, "ok", now, {"title": 'Spotted <b>towhee</b>"><img onerror=alert(1)>',
                                                   "extract": XSS_EXTRACT, "trimmed": False,
                                                   "url": "https://en.wikipedia.org/wiki/Spotted_towhee"})
    s = frame.stem(ALL_FACTS[0])
    wd = facts.read_rec(d, "wikidata", s)
    if wd is None or wd["status"] != "ok":
        facts.write_rec(d, "wikidata", s, "ok", now, {"qid": "Q733375", "enwiki": "Trumpeter swan",
                        "sizes": {"mass": [10300, 11400], "length": None, "wingspan": [245, 245]}, "ebird_code": "truswa"})
    if facts.read_rec(d, "wikipedia", s) is None:
        facts.write_rec(d, "wikipedia", s, "ok", now, {"title": "Trumpeter swan", "trimmed": False,
                        "extract": "The trumpeter swan is a species of swan found in North America.",
                        "url": "https://en.wikipedia.org/wiki/Trumpeter_swan"})
    facts.write_rec(d, "ebird", s, "ok", now, {"code": "truswa", "via": "taxonomy"})
    facts.write_rec(d, "nearby", s, "ok", now, {"reported": True,
                                                 "last_obs_date": (now - dt.timedelta(days=2)).date().isoformat()})


def build(tmp: Path, empty: bool = False):
    data = tmp / "data"
    data.mkdir(parents=True)
    env = {"SERVE_PORT": "1", "DATA_DIR": str(data), "MIN_CONFIDENCE": "0.9", "FACTS_FETCH": "0",
           # A dummy key with FACTS_FETCH=0: no request is ever made, but the
           # cached eBird code and nearby record of the all-facts species show.
           "EBIRD_API_KEY": "DEMOKEY0", "LATITUDE": "47.6", "LONGITUDE": "-122.3"}
    cfg = dataclasses.replace(serve.load_serve_config(env), port=0)
    now = frame.utcnow()
    shown = [] if empty else seed_db(data, now, cfg.tz)
    copy_art(cfg.art.dir)
    if not empty:
        seed_facts(cfg.facts.dir, now)
    srv = serve.CollageServer(cfg, host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", shown


def add_species(srv, sci: str, com: str) -> None:
    conn = bl.open_db(srv.cfg.db_path.parent)
    bl.record(conn, frame.utcnow(), bl.Camera("garage", "x"), bl.Detection(com, sci, 0.97, 0, 3), None)
    conn.close()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


# ----------------------------------------------------------------- page helpers
class Pages:
    def __init__(self, browser, base: str):
        self.browser, self.base = browser, base
        self.foreign: list[str] = []
        self.errors: list[str] = []
        self.dialogs: list[str] = []

    def open(self, path="/", w=1280, h=900, js=True, wait="load"):
        ctx = self.browser.new_context(viewport={"width": w, "height": h}, java_script_enabled=js)
        page = ctx.new_page()
        page.reqs = []
        page.on("request", lambda r: (page.reqs.append(r.url),
                                      None if r.url.startswith(self.base) else self.foreign.append(r.url)))
        page.on("pageerror", lambda e: self.errors.append(f"{path}: {e}"))
        page.on("console", lambda m: self.errors.append(f"{path}: console {m.text}") if m.type == "error"
                and "Failed to load resource" not in m.text else None)
        page.on("dialog", lambda dlg: (self.dialogs.append(dlg.message), dlg.dismiss()))
        page.goto(self.base + path, wait_until=wait)
        return page


def token_of(page) -> str:
    return page.evaluate("() => new URL(document.getElementById('c').src).searchParams.get('v')")


def target_styles(page) -> list[list[float]]:
    return page.evaluate("""() => Array.from(document.querySelectorAll('#targets .t')).map(b =>
        [b.getAttribute('data-sci'), parseFloat(b.style.left), parseFloat(b.style.top),
         parseFloat(b.style.width), parseFloat(b.style.height)])""")


def matches(styles, layout, tol=0.01) -> bool:
    ts = layout["targets"]
    return len(styles) == len(ts) and all(
        s[0] == t["scientific_name"] and all(abs(a - t[k]) <= tol for a, k in zip(s[1:], "xywh"))
        for s, t in zip(styles, ts))


def geom(page):
    return page.evaluate("""() => { const a = document.getElementById('targets').getBoundingClientRect(),
        b = document.getElementById('c').getBoundingClientRect();
        return {d: Math.max(Math.abs(a.left - b.left), Math.abs(a.top - b.top), Math.abs(a.width - b.width),
                            Math.abs(a.height - b.height)), complete: document.getElementById('c').complete,
                w: b.width, h: b.height}; }""")


def wait_until(page, fn, timeout=8.0, step=100) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        page.wait_for_timeout(step)
    return fn()


def card_open(page) -> bool:
    return page.evaluate("() => !document.getElementById('card').hidden")


def card_title(page) -> str:
    return page.evaluate("() => document.getElementById('card-title').textContent")


def about_loading(page) -> int:
    return page.locator("#card .about .loading").count()


def wait_loaded(page):
    page.wait_for_function("() => !document.querySelector('#card .heard .loading')", timeout=10000)


def layout_of(page) -> dict:
    return page.evaluate("() => JSON.parse(document.getElementById('layout').textContent)")


# ----------------------------------------------------------------- checks
def run_checks(pages: Pages, srv, base: str, shown) -> None:
    # ---- AC4-swap (also AC5-geom after a swap)
    page = pages.open("/?refresh_ms=1000")
    old = layout_of(page)
    held, mode = [], {"m": "hold"}

    def png_route(route):
        if mode["m"] == "hold":
            held.append(route)
        elif mode["m"] == "404":
            route.fulfill(status=404, body="gone")
        else:
            route.continue_()
    page.route(lambda u: "/collage.png?v=" in u, png_route)
    add_species(srv, "Anas platyrhynchos", "Mallard")
    got = wait_until(page, lambda: bool(held), 6)
    new = get_json(base + f"/api/layout?hours={old['hours']}&w={old['w']}&h={old['h']}")
    during_ok = got and token_of(page) == old["token"] and matches(target_styles(page), old)
    page.wait_for_timeout(2000)
    during_ok = during_ok and token_of(page) == old["token"] and matches(target_styles(page), old)
    mode["m"] = "pass"
    for r in held:
        r.continue_()
    swapped = wait_until(page, lambda: token_of(page) == new["token"], 6)
    n_req = sum(1 for u in page.reqs if f"v={new['token']}" in u)
    check("AC4-swap", during_ok and swapped and matches(target_styles(page), new) and n_req == 1,
          f"held={len(held)} old-kept={during_ok} swapped={swapped} requests-for-new-token={n_req}")
    g = geom(page)
    check("AC5-geom", g["d"] <= 0.5, f"after swap 1280x900 max-diff={g['d']:.3f}px")

    # ---- AC4-404
    mode["m"] = "404"
    before = get_json(base + "/api/layout")
    add_species(srv, "Branta canadensis", "Canada Goose")
    page.wait_for_timeout(2600)
    new404 = get_json(base + "/api/layout")
    tries = sum(1 for u in page.reqs if f"v={new404['token']}" in u)
    kept = token_of(page) == before["token"] and matches(target_styles(page), before)
    mode["m"] = "pass"
    page.unroute_all()
    rec = wait_until(page, lambda: token_of(page) == new404["token"], 6)
    check("AC4-404", tries >= 2 and kept and rec and matches(target_styles(page), new404),
          f"404s={tries} old-kept={kept} recovered={rec}")
    page.context.close()

    # ---- AC4-hang (held /api/layout)
    page = pages.open("/?refresh_ms=1000")
    held_l, failed = [], []
    lmode = {"m": "hold"}

    def layout_route(route):
        if lmode["m"] == "hold":
            held_l.append(route)
        else:
            route.continue_()
    page.on("requestfailed", lambda r: failed.append(r.url) if "/api/layout" in r.url else None)
    page.route(lambda u: "/api/layout" in u, layout_route)
    before = layout_of(page)
    add_species(srv, "Bucephala albeola", "Bufflehead")
    wait_until(page, lambda: bool(held_l), 5)
    t0 = time.monotonic()
    page.wait_for_timeout(20000)
    overlap = len(held_l)
    aborted = wait_until(page, lambda: bool(failed), 15, 250)
    waited = time.monotonic() - t0
    lmode["m"] = "pass"
    newh = get_json(base + "/api/layout")
    rec = wait_until(page, lambda: token_of(page) == newh["token"], 8)
    check("AC4-hang", overlap == 1 and aborted and rec and before["token"] != newh["token"],
          f"layout-requests-during-hold={overlap} aborted-after={waited + 2:.0f}s recovered={rec}")
    page.context.close()

    # ---- AC4-hang-png (held token PNG)
    page = pages.open("/?refresh_ms=1000")
    held_p = []
    pmode = {"m": "hold"}

    def png_hold(route):
        if pmode["m"] == "hold":
            held_p.append(route)
        else:
            route.continue_()
    page.route(lambda u: "/collage.png?v=" in u, png_hold)
    before = layout_of(page)
    add_species(srv, "Mareca americana", "American Wigeon")
    wait_until(page, lambda: bool(held_p), 5)
    page.wait_for_timeout(31000)
    kept = token_of(page) == before["token"] and matches(target_styles(page), before)
    pmode["m"] = "pass"
    for r in held_p:
        try:
            r.continue_()
        except Exception:  # noqa: BLE001 -- the page already gave up on it
            pass
    newp = get_json(base + "/api/layout")
    rec = wait_until(page, lambda: token_of(page) == newp["token"], 8)
    check("AC4-hang-png", bool(held_p) and kept and rec, f"held={len(held_p)} old-kept={kept} recovered={rec}")
    page.context.close()

    # ---- AC5 and AC5-geom at several viewports
    for w, h in ((360, 800), (533, 400), (800, 600), (1600, 900), (390, 844)):
        page = pages.open(f"/?refresh_ms=3600000", w, h, wait="commit")
        held_g = []
        page.route(lambda u: "/collage.png?v=" in u, lambda r: held_g.append(r))
        page.goto(base + "/?refresh_ms=3600000", wait_until="domcontentloaded")
        g0 = geom(page)
        for r in held_g:
            r.continue_()
        page.unroute_all()
        page.wait_for_load_state("load")
        page.wait_for_function("() => document.getElementById('c').complete")
        g1 = geom(page)
        check("AC5-geom", (not g0["complete"]) and g0["d"] <= 0.5 and g1["d"] <= 0.5,
              f"{w}x{h} before-load complete={g0['complete']} diff={g0['d']:.3f}px, after-load diff={g1['d']:.3f}px")
        if w in (360, 533, 800, 1600):
            lay = layout_of(page)
            bad = []
            for t in lay["targets"]:
                ay = page.evaluate("() => document.getElementById('c').getBoundingClientRect().top + window.scrollY")
                ch = page.evaluate("() => document.getElementById('c').getBoundingClientRect().height")
                cy = ay + ch * (t["y"] + t["h"] / 2) / 100
                page.evaluate(f"() => window.scrollTo(0, {cy} - window.innerHeight / 2)")
                r = page.evaluate("() => { const r = document.getElementById('c').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }")
                x = r[0] + r[2] * (t["x"] + t["w"] / 2) / 100
                y = r[1] + r[3] * (t["y"] + t["h"] / 2) / 100
                page.mouse.click(x, y)
                if not (card_open(page) and card_title(page) == t["common_name"]):
                    bad.append(t["common_name"])
                page.keyboard.press("Escape")
            check("AC5", not bad and lay["targets"], f"{w}px: {len(lay['targets'])} targets, misses={bad}")
        page.context.close()

    # ---- AC6: the overlay is invisible
    page = pages.open("/", 1280, 900)
    page.mouse.move(0, 0)
    a = Image.open(io.BytesIO(page.locator("#stage").screenshot())).convert("RGB")
    page.evaluate("() => document.getElementById('targets').remove()")
    b = Image.open(io.BytesIO(page.locator("#stage").screenshot())).convert("RGB")
    check("AC6", a.size == b.size and ImageChops.difference(a, b).getbbox() is None, f"stage {a.size}")
    page.context.close()

    # ---- AC7: no JS
    page = pages.open("/", 1280, 900, js=False)
    n_t = page.locator("#targets > *").count()
    hidden = page.locator("#card").is_hidden()
    nat = page.locator("#c").evaluate("e => e.naturalWidth")
    check("AC7", n_t == 0 and hidden and nat > 0, f"targets={n_t} card-hidden={hidden} img-width={nat}")
    page.context.close()

    # ---- AC8: footer link
    page = pages.open("/", 1280, 900)
    page.locator("footer a").click()
    page.wait_for_load_state("load")
    check("AC8", page.url.endswith("/attribution"), page.url.replace(base, ""))
    page.context.close()

    # ---- AC9, AC10, AC11, AC12, font, AC44, art, AC14/15, AC13
    page = pages.open("/", 1280, 900)
    lay = layout_of(page)
    first = lay["targets"][0]
    ms = page.evaluate("""(sci) => { const b = Array.from(document.querySelectorAll('#targets .t'))
        .find(x => x.getAttribute('data-sci') === sci); const t0 = performance.now(); b.click();
        return new Promise(res => requestAnimationFrame(() => res([!document.getElementById('card').hidden,
        performance.now() - t0]))); }""", first["scientific_name"])
    check("AC9", ms[0] and ms[1] < 100, f"visible={ms[0]} after {ms[1]:.1f} ms (one frame)")
    wait_loaded(page)
    page.evaluate("() => document.fonts.ready")
    fonts_ok = page.evaluate("""() => document.fonts.check('16px "Libre Baskerville"') &&
        Array.from(document.fonts).some(f => f.family.indexOf('Libre Baskerville') >= 0 && f.status === 'loaded')""")
    fam = page.evaluate("() => getComputedStyle(document.getElementById('card-title')).fontFamily")
    check("font", fonts_ok, f"loaded, card font-family: {fam}")
    # AC11: clicks inside stay open
    page.locator("#card-title").click()
    inside = card_open(page)
    if page.locator("#card svg.chart").count():
        page.locator("#card svg.chart").click()
    check("AC11", inside and card_open(page), "title and chart clicks keep the card open")
    page.keyboard.press("Escape")
    # AC10: three ways to close, focus back to the opener
    closers = {"Esc": lambda: page.keyboard.press("Escape"),
               "close-button": lambda: page.locator("#card .close").click(),
               "scrim": lambda: page.mouse.click(8, 8)}
    for name, act in closers.items():
        sci = first["scientific_name"]
        page.locator(f'#targets .t[data-sci="{sci}"]').click()
        wait_loaded(page)
        act()
        fsci = page.evaluate("() => document.activeElement && document.activeElement.getAttribute('data-sci')")
        check("AC10", not card_open(page) and fsci == sci, f"{name}: closed, focus on {fsci}")
    # AC12: tab order and names, Enter and Space
    page.reload(wait_until="load")      # a fresh sequential-focus starting point
    order = []
    for _ in range(len(lay["targets"])):
        page.keyboard.press("Tab")
        order.append(page.evaluate("() => [document.activeElement.getAttribute('data-sci'), document.activeElement.getAttribute('aria-label')]"))
    exp = [[t["scientific_name"], t["common_name"]] for t in lay["targets"]]
    keys_ok = True
    for key in ("Enter", " "):
        page.locator(f'#targets .t[data-sci="{first["scientific_name"]}"]').focus()
        page.keyboard.press(key)
        keys_ok = keys_ok and card_open(page) and card_title(page) == first["common_name"]
        wait_loaded(page)
        page.keyboard.press("Escape")
    diff = next(([i, o, e] for i, (o, e) in enumerate(zip(order, exp)) if o != e), None)
    check("AC12", order == exp and keys_ok,
          f"{len(order)} targets in species order with common-name labels; Enter/Space open={keys_ok}; first diff={diff}")
    page.context.close()

    # AC44: XSS species
    page = pages.open("/#species=" + XSS[0].replace(" ", "%20"), 1280, 900)
    wait_loaded(page)
    page.wait_for_timeout(300)
    bad = page.evaluate("() => document.querySelectorAll('#card script, #card img[onerror], #card [onerror]').length")
    text = page.evaluate("() => document.getElementById('card').textContent")
    check("AC44", bad == 0 and "<script>alert(1)</script>" in text and '"><img onerror=' in text and not pages.dialogs,
          f"injected-elements={bad} literal-text-shown={'<script>alert(1)</script>' in text} dialogs={len(pages.dialogs)}")
    page.context.close()

    # #10: Dog rows are in the db but get no tap target, and its pop-up is a 404
    page = pages.open("/", 1280, 900)
    dog_targets = page.locator('#targets .t[data-sci="Dog"]').count()
    status = page.request.get(page.url.split("#")[0] + "api/species/Dog").status
    check("#10", dog_targets == 0 and status == 404, f"Dog tap targets={dog_targets} /api/species/Dog={status}")
    page.context.close()

    # art: the placeholder species and a non-binomial name never request /plate/
    for sci in (PLACEHOLDER[0], NON_BINOMIAL[0]):
        page = pages.open("/", 1280, 900)
        page.locator(f'#targets .t[data-sci="{sci}"]').click()
        wait_loaded(page)
        imgs = page.locator("#card .plate img").count()
        ph = page.locator("#card .plate .plate-ph").count()
        plate_reqs = [u for u in page.reqs if f"/plate/{frame.stem(sci)}" in u]
        check("art", imgs == 0 and ph == 1 and not plate_reqs, f"{sci}: plate img={imgs} placeholder={ph} /plate/ requests={len(plate_reqs)}")
        page.context.close()
    # art: a species with art shows its plate
    sci = "Psaltriparus minimus"
    page = pages.open("/", 1280, 900)
    page.locator(f'#targets .t[data-sci="{sci}"]').click()
    wait_loaded(page)
    page.wait_for_function("() => { const i = document.querySelector('#card .plate img'); return i && i.complete; }")
    nat = page.evaluate("() => document.querySelector('#card .plate img').naturalWidth")
    check("art", nat > 0, f"{sci}: plate loaded ({nat}px wide)")
    page.context.close()

    # AC14/15: fit at 533x400 and the bottom sheet at 390x844
    page = pages.open("/#species=Psaltriparus%20minimus", 533, 400)
    wait_loaded(page)
    box = page.evaluate("""() => { const r = e => e.getBoundingClientRect().toJSON();
        return {card: r(document.getElementById('card')), title: r(document.getElementById('card-title')),
                close: r(document.querySelector('#card .close')), vw: innerWidth, vh: innerHeight,
                scroll: document.querySelector('#card .body').scrollHeight > document.querySelector('#card .body').clientHeight}; }""")
    inside = lambda r: r["left"] >= -0.5 and r["top"] >= -0.5 and r["right"] <= box["vw"] + 0.5 and r["bottom"] <= box["vh"] + 0.5  # noqa: E731
    check("AC14/15", inside(box["card"]) and inside(box["title"]) and inside(box["close"]),
          f"533x400 card {box['card']['width']:.0f}x{box['card']['height']:.0f} at ({box['card']['left']:.0f},{box['card']['top']:.0f}), body scrolls={box['scroll']}")
    page.context.close()
    page = pages.open("/#species=Psaltriparus%20minimus", 390, 844)
    wait_loaded(page)
    r = page.evaluate("() => document.getElementById('card').getBoundingClientRect().toJSON()")
    check("AC14/15", abs(r["bottom"] - 844) <= 1 and abs(r["left"]) <= 1 and abs(r["width"] - 390) <= 1
          and r["height"] <= 0.85 * 844 + 1, f"390x844 sheet {r['width']:.0f}x{r['height']:.0f}, bottom={r['bottom']:.0f}")
    page.context.close()
    # #8: the bottom sheet on a short phone viewport (85vh = 340 px). The header
    # stays on screen and the body scrolls under it.
    page = pages.open("/#species=Psaltriparus%20minimus", 390, 400)
    wait_loaded(page)
    page.evaluate("() => document.fonts.ready")
    m = page.evaluate("""() => { const r = e => e.getBoundingClientRect().toJSON();
        const b = document.querySelector('#card .body');
        const out = {card: r(document.getElementById('card')), title: r(document.getElementById('card-title')),
                     close: r(document.querySelector('#card .close')), vw: innerWidth, vh: innerHeight,
                     body_h: b.clientHeight, body_scroll_h: b.scrollHeight};
        b.scrollTop = 120;
        out.scrolled = b.scrollTop;
        out.title_after = r(document.getElementById('card-title'));
        out.close_after = r(document.querySelector('#card .close'));
        return out; }""")
    vis = lambda r: (r["width"] > 0 and r["height"] > 0 and r["left"] >= -0.5 and r["top"] >= -0.5  # noqa: E731
                     and r["right"] <= m["vw"] + 0.5 and r["bottom"] <= m["vh"] + 0.5)
    c = m["card"]
    sheet = abs(c["bottom"] - 400) <= 1 and abs(c["left"]) <= 1 and abs(c["width"] - 390) <= 1 and c["height"] <= 0.85 * 400 + 1
    header = vis(m["title"]) and vis(m["close"]) and vis(m["title_after"]) and vis(m["close_after"])
    scrolls = m["body_scroll_h"] > m["body_h"] and m["scrolled"] > 0
    check("#8", sheet and header and scrolls and m["body_h"] >= 150,
          f"390x400 sheet {c['width']:.0f}x{c['height']:.0f} bottom={c['bottom']:.0f}; title/close visible={header}; "
          f"body {m['body_h']}px of {m['body_scroll_h']}px, scrolled to {m['scrolled']}")
    page.context.close()

    # AC13: refresh while the card is open and scrolled
    page = pages.open("/?refresh_ms=1000&w=533&h=400", 533, 400)
    t = layout_of(page)["targets"][0]
    page.locator(f'#targets .t[data-sci="{t["scientific_name"]}"]').click()
    wait_loaded(page)
    page.evaluate("() => { document.querySelector('#card .body').scrollTop = 60; }")
    snap = lambda: page.evaluate("""() => [document.querySelector('#card .body').scrollTop,
        document.activeElement.className, document.getElementById('card').textContent]""")  # noqa: E731
    s0 = snap()
    tok0 = token_of(page)
    add_species(srv, "Aythya collaris", "Ring-necked Duck")
    swapped = wait_until(page, lambda: token_of(page) != tok0, 8)
    s1 = snap()
    check("AC13", swapped and s0 == s1 and card_open(page) and s0[0] > 0,
          f"swapped={swapped} scrollTop={s0[0]}->{s1[0]} focus={s1[1]} text-unchanged={s0[2] == s1[2]}")
    page.context.close()

    # AC16: failures show "Couldn't load details" and close normally
    for how in ("500", "abort", "404"):
        page = pages.open("/", 1280, 900)
        if how == "abort":
            page.route(lambda u: "/api/species/" in u, lambda route: route.abort())
        else:
            code = int(how)
            page.route(lambda u: "/api/species/" in u, lambda route: route.fulfill(status=code, body="x"))
        page.locator("#targets .t").first.click()
        page.wait_for_function("() => document.querySelector('#card .heard').textContent.indexOf('load details') >= 0", timeout=5000)
        txt = page.evaluate("() => document.querySelector('#card .heard').textContent")
        page.keyboard.press("Escape")
        check("AC16", "Couldn\u2019t load details" in txt and not card_open(page), f"{how}: shown and closed")
        page.context.close()

    # AC17: pending causes exactly one follow-up
    page = pages.open("/", 1280, 900)
    sci = ALL_FACTS[0]     # has a blurb, so #6 below can check it is kept
    real = get_json(base + "/api/species/" + sci.replace(" ", "%20"))
    real["pending"] = ["wikipedia"]
    body = json.dumps(real)
    want_blurb = 1
    assert real["facts"].get("wikipedia"), f"{sci} needs a Wikipedia blurb for the #6 checks"
    page.route(lambda u: "/api/species/" in u, lambda route: route.fulfill(status=200, body=body,
                                                                         content_type="application/json"))
    page.locator(f'#targets .t[data-sci="{sci}"]').click()
    wait_loaded(page)
    before = about_loading(page)
    page.wait_for_timeout(9500)
    n = sum(1 for u in page.reqs if "/api/species/" in u)
    check("AC17", n == 2, f"requests to /api/species in 9.5 s: {n}")
    # #6: both responses pending; the follow-up is the last pass, so no loading line after it
    after, blurb = about_loading(page), page.locator("#card .about .blurb").count()
    check("#6", before == 1 and after == 0 and blurb == want_blurb,
          f"both pending: loading line before={before} after={after}, blurb kept={blurb}")
    page.context.close()

    # #6: pending, then the follow-up fails: keep what arrived, drop the loading line
    page = pages.open("/", 1280, 900)
    calls = []

    def first_ok_then_abort(route):
        calls.append(1)
        if len(calls) == 1:
            route.fulfill(status=200, body=body, content_type="application/json")
        else:
            route.abort()
    page.route(lambda u: "/api/species/" in u, first_ok_then_abort)
    page.locator(f'#targets .t[data-sci="{sci}"]').click()
    wait_loaded(page)
    page.wait_for_timeout(5500)
    after, blurb = about_loading(page), page.locator("#card .about .blurb").count()
    heard = page.evaluate("() => document.querySelector('#card .heard').textContent")
    check("#6", len(calls) == 2 and after == 0 and blurb == want_blurb and "load details" not in heard,
          f"follow-up aborted: requests={len(calls)} loading line={after} blurb kept={blurb}")
    page.context.close()

    # #6: pending with nothing yet, twice: the follow-up says there are no notes
    page = pages.open("/", 1280, 900)
    empty = dict(real, facts={}, links={}, pending=["wikidata", "wikipedia"])
    ebody = json.dumps(empty)
    page.route(lambda u: "/api/species/" in u, lambda route: route.fulfill(status=200, body=ebody,
                                                                          content_type="application/json"))
    page.locator(f'#targets .t[data-sci="{sci}"]').click()
    wait_loaded(page)
    page.wait_for_timeout(5500)
    text = page.evaluate("() => document.querySelector('#card .about').textContent")
    check("#6", about_loading(page) == 0 and "No notes for this bird yet." in text,
          f"nothing arrived: loading line={about_loading(page)} quiet line shown={'No notes' in text}")
    page.context.close()

    # deeplink
    page = pages.open("/#species=%E0%A4%A", 1280, 900)
    page.wait_for_timeout(500)
    bad_open = card_open(page)
    errs = [e for e in pages.errors if "%E0%A4%A" in e]
    page.evaluate("() => { location.hash = '#species=Psaltriparus%20minimus'; }")
    page.wait_for_function("() => !document.getElementById('card').hidden")
    check("deeplink", not bad_open and not errs and card_title(page) == "Bushtit",
          f"bad escape opened={bad_open} errors={len(errs)}; Bushtit opened via hashchange")
    page.context.close()
    page = pages.open("/#species=" + AGED[0].replace(" ", "%20"), 1280, 900)
    wait_loaded(page)
    page.wait_for_function("() => { const i = document.querySelector('#card .plate img'); return i && i.complete; }", timeout=5000)
    txt = page.evaluate("() => document.querySelector('#card .heard').textContent")
    check("deeplink", "Not heard in the last 24 hours" in txt and card_title(page) == AGED[1],
          f"off-page {AGED[0]}: '{txt.strip()}', plate from the response")
    page.context.close()


def shots(pages: Pages, base: str, empty_base: str) -> None:
    out = AGENTS / "shots"
    out.mkdir(exist_ok=True)
    picks = {"fugleramme": ("Melospiza melodia", base), "audubon": ("Psaltriparus minimus", base),
             "placeholder": (PLACEHOLDER[0], base), "all-facts": (ALL_FACTS[0], base),
             "no-facts": (NO_FACTS[0], base), "non-binomial": (NON_BINOMIAL[0], base), "empty-page": ("Psaltriparus minimus", empty_base)}
    for name, (sci, b) in picks.items():
        for w, h in ((1600, 1200), (533, 400), (390, 844), (390, 400)):
            ctx = pages.browser.new_context(viewport={"width": w, "height": h})
            page = ctx.new_page()
            page.goto(b + "/#species=" + sci.replace(" ", "%20"), wait_until="load")
            wait_loaded(page)
            page.evaluate("() => document.fonts.ready")
            page.wait_for_timeout(400)
            page.screenshot(path=str(out / f"{name}-{w}x{h}.png"))
            ctx.close()
    for name, sci, (w, h) in (("popup-desktop", "Melospiza melodia", (1280, 900)),
                              ("popup-mobile", "Corvus brachyrhynchos", (390, 800)),
                              ("popup-vignette", "Psaltriparus minimus", (1280, 900))):
        ctx = pages.browser.new_context(viewport={"width": w, "height": h})
        page = ctx.new_page()
        page.goto(base + "/#species=" + sci.replace(" ", "%20"), wait_until="load")
        wait_loaded(page)
        page.evaluate("() => document.fonts.ready")
        page.wait_for_timeout(400)
        page.screenshot(path=str(AGENTS / f"{name}.png"))
        ctx.close()
    print(f"screenshots: {out}/ (28) and .agents/popup-desktop.png, popup-mobile.png, popup-vignette.png")


def main() -> int:
    def refuse(url, timeout=None):
        raise frame.NotFound(url)
    frame.fetch_url = refuse

    def no_facts(url, *a, **kw):
        raise AssertionError("facts fetch in an offline run")
    facts.http_get = no_facts
    from playwright.sync_api import sync_playwright

    tmp = Path(tempfile.mkdtemp(prefix="popup-shots-"))
    srv, base, shown = build(tmp / "a")
    esrv, ebase, _ = build(tmp / "b", empty=True)
    print(f"demo server {base} with {len(shown)} species + Dog + Larus sp.; empty server {ebase}")
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            print(f"chromium.launch() failed: {exc}")
            return 2
        pages = Pages(browser, base)
        try:
            run_checks(pages, srv, base, shown)
            pages.base = base
            shots(pages, base, ebase)
        except Exception as exc:  # noqa: BLE001
            check("run", False, f"{type(exc).__name__}: {exc}")
        foreign = [u for u in pages.foreign if not u.startswith(ebase) and not u.startswith("data:")]
        check("AC47", not foreign, f"foreign requests: {foreign[:3]}")
        errs = [e for e in pages.errors if "%E0%A4%A" not in e]
        check("page-errors", not errs, f"{errs[:3]}")
        browser.close()
    srv.shutdown()
    esrv.shutdown()
    failed = [r for r in results if not r[0]]
    print(f"{len(results) - len(failed)} PASS, {len(failed)} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
