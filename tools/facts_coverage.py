"""Real-network coverage check for the pop-up facts (manual; not in the image).

Resolves facts for the species heard on the live server plus a fixed list
of approved and common species, with the real facts.Facts code and a temp
(or --keep) FACTS_DIR, and prints a coverage table:

  uv run --no-project --python 3.11 --with pillow==12.3.0 python tools/facts_coverage.py \
      [--base http://192.168.0.103:8085] [--hours 168] [--keep DIR]

EBIRD_API_KEY is used from the environment only if set, and never printed.
All About Birds blocks scripted requests (Cloudflare 403 for every URL), so
its column reports the direct status and the Internet Archive availability
API, labelled as such.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import facts  # noqa: E402
import frame  # noqa: E402

HARD = [
    ("Zonotrichia albicollis", "White-throated Sparrow"), ("Aphelocoma californica", "California Scrub-Jay"),
    ("Psaltriparus minimus", "Bushtit"), ("Myadestes townsendi", "Townsend's Solitaire"),
    ("Cygnus buccinator", "Trumpeter Swan"), ("Meleagris gallopavo", "Wild Turkey"),
    ("Tachycineta bicolor", "Tree Swallow"), ("Ixoreus naevius", "Varied Thrush"),
    ("Cyanocitta stelleri", "Steller's Jay"),
]
COMMON = [
    ("Corvus brachyrhynchos", "American Crow"), ("Melospiza melodia", "Song Sparrow"),
    ("Poecile atricapillus", "Black-capped Chickadee"), ("Junco hyemalis", "Dark-eyed Junco"),
    ("Calypte anna", "Anna's Hummingbird"),
]
APPROVED = [   # the rest of the species in the approved screenshots (.agents/artwork)
    ("Bombycilla cedrorum", "Cedar Waxwing"), ("Bubo virginianus", "Great Horned Owl"),
    ("Columba livia", "Rock Pigeon"), ("Corvus corax", "Common Raven"),
    ("Glaucidium gnoma", "Northern Pygmy-Owl"), ("Haemorhous mexicanus", "House Finch"),
    ("Mergus merganser", "Common Merganser"), ("Pipilo maculatus", "Spotted Towhee"),
    ("Poecile rufescens", "Chestnut-backed Chickadee"), ("Setophaga coronata", "Yellow-rumped Warbler"),
    ("Streptopelia decaocto", "Eurasian Collared-Dove"), ("Zenaida asiatica", "White-winged Dove"),
    ("Zonotrichia atricapilla", "Golden-crowned Sparrow"), ("Turdus migratorius", "American Robin"),
]
NAME_RE = re.compile(r"[A-Za-z .'-]{1,100}")


def live_species(base: str, hours: int) -> list[tuple[str, str]]:
    req = urllib.request.Request(f"{base}/api/recent?hours={hours}", headers={"User-Agent": frame.USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return [(s["scientific_name"], s["common_name"]) for s in data["species"]]


def aab_status(url: str) -> tuple[str, str]:
    """(direct HTTP status, Wayback availability) for an All About Birds URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": frame.USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            direct = str(resp.status)
    except urllib.error.HTTPError as exc:
        direct = str(exc.code)
    except Exception as exc:  # noqa: BLE001
        direct = type(exc).__name__
    path = url.split("://", 1)[1]
    try:
        req = urllib.request.Request("https://archive.org/wayback/available?url=" + quote(path, safe=""),
                                     headers={"User-Agent": frame.USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            snap = json.loads(resp.read()).get("archived_snapshots", {}).get("closest") or {}
        wb = "y" if snap.get("status") == "200" else "n"
    except Exception as exc:  # noqa: BLE001
        wb = "?" + type(exc).__name__
    if wb != "y":   # the availability API often answers empty; the CDX index is authoritative
        try:
            req = urllib.request.Request("https://web.archive.org/cdx/search/cdx?url=" + quote(path, safe="")
                                         + "&filter=statuscode:200&limit=1", headers={"User-Agent": frame.USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                wb = "y" if resp.read().strip() else "n"
        except Exception as exc:  # noqa: BLE001
            wb = "?" + type(exc).__name__
    return direct, wb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.0.103:8085")
    ap.add_argument("--hours", type=int, default=168)
    ap.add_argument("--keep", type=Path, help="facts dir to fill and keep (default: a temp dir)")
    ap.add_argument("--no-aab", action="store_true", help="skip the All About Birds checks")
    args = ap.parse_args()

    try:
        heard = live_species(args.base, args.hours)
    except Exception as exc:  # noqa: BLE001
        print(f"live server unreachable ({type(exc).__name__}); using the fixed list only")
        heard = []
    species: dict[str, tuple[str, str]] = {}
    for sci, com in HARD + COMMON + APPROVED + heard:
        species.setdefault(sci, (com, "heard" if (sci, com) in heard else "list"))
    rejected = [sci for sci, _ in heard if not NAME_RE.fullmatch(sci)]

    raw = os.environ.get("EBIRD_API_KEY", "").strip()
    key = facts.Secret(raw) if raw else None
    del raw
    d = args.keep or Path(tempfile.mkdtemp(prefix="facts-coverage-"))
    f = facts.Facts(facts.FactsConfig(d, True, key))
    now = frame.utcnow()

    rows = []
    for sci, (com, src) in species.items():
        if not facts.BINOMIAL_RE.fullmatch(sci):
            rows.append((sci, com, src, None, None, None))
            continue
        t0 = time.monotonic()
        r = f.lookup(sci, com, now, budget=60)
        deadline = time.monotonic() + 60
        while f._inflight and time.monotonic() < deadline:
            time.sleep(0.1)
        r = f.lookup(sci, com, now, budget=0)
        wd = facts.read_rec(d, "wikidata", frame.stem(sci))
        aab = ("-", "-") if args.no_aab else aab_status(facts.aab_url(com))
        rows.append((sci, com, src, r.facts, wd, aab, time.monotonic() - t0))

    print(f"facts coverage, {len(rows)} species, {now.isoformat(timespec='seconds')}, "
          f"eBird key {'set' if key else 'not set'} (eBird code from "
          f"{'taxonomy' if key else 'Wikidata P3444'})")
    print("AAB direct = HEAD status of the All About Birds URL; AAB wayback = Internet Archive has a 200 snapshot (availability API, then CDX index)")
    hdr = ("scientific name", "common name", "src", "WD title", "blurb", "mass", "length", "wingspan",
           "eBird code", "AAB direct", "AAB wayback")
    print(" | ".join(hdr))
    tot = {"title": 0, "blurb": 0, "mass": 0, "length": 0, "wingspan": 0, "anysize": 0, "code": 0, "aab": 0}
    misses = []
    n = 0
    for row in rows:
        sci, com, src = row[:3]
        if row[3] is None:
            print(f"{sci} | {com} | {src} | non-binomial: stats only")
            continue
        n += 1
        fa, wd, aab = row[3], row[4], row[5]
        title = (wd or {}).get("data", {}) or {}
        title = title.get("enwiki") if wd and wd["status"] == "ok" else None
        w = fa.get("wikipedia")
        size = fa.get("size") or {}
        code = (fa.get("ebird") or {}).get("code")
        vals = {"title": bool(title), "blurb": bool(w), "mass": bool(size.get("mass")),
                "length": bool(size.get("length")), "wingspan": bool(size.get("wingspan")),
                "anysize": bool(size), "code": bool(code), "aab": aab[1] == "y"}
        for k, v in vals.items():
            tot[k] += v
        if not w or not code:
            misses.append(sci)
        print(" | ".join([sci, com, src, title or "-", (w or {}).get("title", "n") if w else "n",
                          size.get("mass") or "n", size.get("length") or "n", size.get("wingspan") or "n",
                          code or "n", aab[0], aab[1]]))
    print()
    pc = lambda k: f"{tot[k]}/{n} ({100 * tot[k] / n:.0f}%)" if n else "0/0"  # noqa: E731
    print(f"Wikipedia title resolved via Wikidata: {pc('title')}")
    print(f"blurb: {pc('blurb')} (target >= 90%)")
    print(f"eBird code: {pc('code')} (target >= 95%)")
    print(f"mass: {pc('mass')}, length: {pc('length')}, wingspan: {pc('wingspan')}, any size: {pc('anysize')}")
    print(f"All About Birds archived 200 (Wayback): {pc('aab')}")
    print("missing blurb or code: " + (", ".join(misses) or "none"))
    print("heard names the /api/species validator would reject: " + (", ".join(rejected) or "none"))
    print(f"facts dir: {d}")
    ok = n and tot["blurb"] / n >= 0.9 and tot["code"] / n >= 0.95
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
