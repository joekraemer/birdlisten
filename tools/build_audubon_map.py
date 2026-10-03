#!/usr/bin/env python3
"""Build audubon.json: BirdNET v2.4 scientific name -> one Havell plate of
Audubon's *The Birds of America* on Wikimedia Commons.

Run by hand from the repo root (needs the network, stdlib only):

  python3.11 tools/build_audubon_map.py [--labels PATH|URL] [--out audubon.json]

Sources, unioned: Wikidata plate items (`part of` Q377817, `depicts` P180 ->
taxon P225), Commons `Category:<Genus species> (illustrations)` on each
plate's canonical file, and the manual tables below. The printed report is the
number of record and is reviewed in the CR, including every chosen pair that
only Commons supports. The runtime (frame.py) never queries Wikidata or the
Commons API; it only fetches thumbnails of the files named here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "birdlisten/1.0 (https://github.com/joekraemer/birdlisten)"
LABELS_URL = ("https://raw.githubusercontent.com/joeweiss/birdnetlib/"
              "8746db66d1605882b80abcf2ac7706fd5c3125f9/"   # tag 0.18.0, matches pyproject
              "src/birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Labels.txt")
LABELS_FILE = "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
MIN_LABELS = 6000
SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_PAGE = "https://commons.wikimedia.org/wiki/"
WORK = "Q377817"            # The Birds of America
AVES = "Q5113"
CATEGORY = "Category:The Birds of America"
PLATES = range(1, 436)       # Havell plates 1..435
SPARQL_BATCH = 150
COMMONS_BATCH = 50
PAUSE = 1.0                  # seconds between SPARQL batches
TIMEOUT = 120

# Commons or Wikidata name -> BirdNET v2.4 name, for splits and renames that
# BirdNET's 2021-era list does not have yet.
NAME_ALIASES = {
    "Setophaga aestiva": "Setophaga petechia",           # Yellow Warbler, plates 35, 65, 95
    "Setophaga auduboni": "Setophaga coronata",          # Yellow-rumped Warbler, 395
    "Tyto furcata": "Tyto alba",                         # Barn Owl, 171
    "Numenius hudsonicus": "Numenius phaeopus",          # Whimbrel, 237
    "Larus smithsonianus": "Larus argentatus",           # Herring Gull, 291
    "Passerella unalaschcensis": "Passerella iliaca",    # Fox Sparrow, 424
    "Accipiter atricapillus": "Accipiter gentilis",      # Northern Goshawk, 141
}

# Figures no machine source lists.
PLATE_ADDITIONS = {
    # Octavo vol. 2 p. 161: Townsend's Parus minimus, basionym of Psaltriparus minimus.
    353: "Psaltriparus minimus",
    # Octavo vol. 4 pp. 115-116: "Ultramarine Jay" from Fort Vancouver, a
    # California Scrub-Jay under the 2016 split (identification inferred).
    362: "Aphelocoma californica",
}

# (plate, species) pairs a machine source supplies wrongly.
PAIR_DENY = {
    (46, "Sciurus carolinensis"),    # the Barred Owl's prey, not a subject
    (275, "Anous minutus"),          # Commons mis-tag; the plate is the Brown Noddy
    (353, "Poecile hudsonicus"),     # Commons mis-tag; Boreal Chickadee is on 194
}

TARGETS = ("Zonotrichia albicollis", "Aphelocoma californica", "Psaltriparus minimus",
           "Myadestes townsendi", "Cygnus buccinator", "Meleagris gallopavo",
           "Tachycineta bicolor", "Ixoreus naevius", "Cyanocitta stelleri")

FILE_RE = re.compile(r"^File:(\d{1,3}) (.+)\.jpe?g$", re.I)
SKIP_RE = re.compile(r"\(cropped\)|restored|detail", re.I)
CAT_RE = re.compile(r"^Category:([A-Z][a-z]+ [a-z]+( [a-z]+)?) \(illustrations\)$")
HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)
TAG_RE = re.compile(r"<[^>]+>")


class BuildError(Exception):
    pass


class LabelsError(Exception):
    pass


class FetchError(Exception):
    """A network failure, with the URL that failed."""


# ----------------------------------------------------------------- fetch layer
def _fetch(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    """The only network call; tests monkeypatch it."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"{url[:300]}: {exc}") from exc


def sparql(query: str) -> list[dict]:
    body = urllib.parse.urlencode({"query": query}).encode()
    raw = _fetch(SPARQL_URL, body, {"Accept": "application/sparql-results+json"})
    return json.loads(raw)["results"]["bindings"]


def commons(params: dict) -> list[dict]:
    """All pages of a Commons API query, following `continue`. Returns the raw
    page dicts; one title may appear in several (categories are paged too)."""
    pages: list[dict] = []
    cont: dict = {}
    while True:
        q = {"action": "query", "format": "json", **params, **cont}
        r = json.loads(_fetch(COMMONS_API + "?" + urllib.parse.urlencode(q)))
        pages += list(r.get("query", {}).get("pages", {}).values())
        if "continue" not in r:
            return pages
        cont = r["continue"]
        time.sleep(0.2)


# ----------------------------------------------------------------- labels
def parse_labels(text: str) -> dict[str, str]:
    """'Genus species_Common Name' lines -> {scientific: common}."""
    out = {}
    for line in text.splitlines():
        if "_" in line:
            sci, com = line.strip().split("_", 1)
            out[sci] = com
    return out


def installed_labels() -> Path | None:
    spec = importlib.util.find_spec("birdnetlib")
    if spec is None or not spec.submodule_search_locations:
        return None
    for base in spec.submodule_search_locations:
        hits = sorted(glob.glob(str(Path(base) / "**" / LABELS_FILE), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def load_labels(arg: str | None, fetch=None, find_installed=installed_labels) -> tuple[dict[str, str], str]:
    """Labels from --labels (a path or an http(s) URL), else the installed
    birdnetlib, else LABELS_URL. Raises LabelsError when unreadable or short."""
    fetch = fetch or _fetch
    if arg:
        src = arg
    else:
        p = find_installed()
        src = str(p) if p else LABELS_URL
    try:
        if src.startswith(("http://", "https://")):
            text = fetch(src).decode("utf-8")
        else:
            text = Path(src).read_text(encoding="utf-8")
    except (OSError, FetchError, UnicodeDecodeError) as exc:
        raise LabelsError(f"cannot read labels from {src}: {exc}") from exc
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < MIN_LABELS:
        raise LabelsError(f"labels from {src} have {len(lines)} lines, expected >= {MIN_LABELS}")
    return parse_labels(text), src


# ----------------------------------------------------------------- pure rules
def plate_of(title: str) -> int | None:
    """'File:8 White throated Sparrow.jpg' -> 8, for a usable plate file."""
    m = FILE_RE.match(title)
    if not m or SKIP_RE.search(title):
        return None
    n = int(m[1])
    return n if n in PLATES else None


def category_taxon(cat: str) -> str | None:
    m = CAT_RE.match(cat)
    return m[1] if m else None


def file_from_p18(url: str) -> str:
    """Wikidata P18 '.../Special:FilePath/8%20White%20throated%20Sparrow.jpg' -> 'File:8 White ...'."""
    return "File:" + urllib.parse.unquote(url.rsplit("/", 1)[1]).replace("_", " ")


def canonical_files(titles, p18: dict[int, str]) -> dict[int, str]:
    """One file per plate: Wikidata's P18 if it is a candidate, else the single
    candidate. Zero or several candidates is a BuildError naming the plate."""
    cand: dict[int, list[str]] = defaultdict(list)
    for t in titles:
        n = plate_of(t)
        if n is not None:
            cand[n].append(t)
    out, bad = {}, []
    for n in PLATES:
        c = sorted(set(cand.get(n, [])))
        if p18.get(n) in c:
            out[n] = p18[n]
        elif len(c) == 1:
            out[n] = c[0]
        else:
            bad.append(f"plate {n}: {len(c)} candidates {c}")
    if bad:
        raise BuildError("no canonical file for " + "; ".join(bad))
    return out


@dataclass
class Taxon:
    names: set[str] = field(default_factory=set)      # P225
    synonyms: set[str] = field(default_factory=set)   # P225 of P1420 in either direction
    common: set[str] = field(default_factory=set)     # en label + P1843 en


def resolve(names, synonyms, common, sci: set[str], com: dict[str, str]) -> tuple[str | None, str | None]:
    """First matching rule wins: alias, exact, trinomial -> binomial,
    Wikidata synonym, English common name (case-insensitive)."""
    names = sorted(names)
    for n in names:
        if n in NAME_ALIASES:
            return NAME_ALIASES[n], "alias"
    for n in names:
        if n in sci:
            return n, "exact"
    for n in names:
        b = " ".join(n.split()[:2])
        if b != n and b in sci:
            return b, "trinomial"
    for n in sorted(synonyms):
        if n in sci:
            return n, "synonym"
    for c in sorted(common):
        if c.lower() in com:
            return com[c.lower()], "common"
    return None, None


@dataclass
class Contribution:
    source: str          # "wikidata" | "commons" | "manual"
    rule: str
    taxon: str           # the source's name or QID, for the report
    bird: bool | None    # False = resolves only to non-Aves items; None = unknown to Wikidata


def choose(pairs: dict[tuple[int, str], list[Contribution]]) -> tuple[dict[str, int], dict[int, int]]:
    """Plate per species by (-sources agreeing, on_plate, plate)."""
    on_plate: Counter = Counter(n for n, _ in pairs)
    best: dict[str, tuple] = {}
    for (n, s), cs in pairs.items():
        key = (-len({c.source for c in cs}), on_plate[n], n)
        if s not in best or key < best[s]:
            best[s] = key
    return {s: k[2] for s, k in best.items()}, dict(on_plate)


def apply_deny(pairs: dict, deny=None) -> list[tuple[int, str]]:
    """Remove PAIR_DENY pairs; every one must be present (else it is stale)."""
    deny = PAIR_DENY if deny is None else deny
    absent = sorted(k for k in deny if k not in pairs)
    if absent:
        raise BuildError(f"PAIR_DENY pairs not among the resolved pairs: {absent}")
    for k in deny:
        del pairs[k]
    return sorted(deny)


def bird_filter(pairs: dict) -> list[tuple[int, str, str]]:
    """Drop contributions whose taxon resolves only to non-Aves items, and
    pairs left with none. Returns the dropped (plate, species, taxon)."""
    dropped = []
    for k in list(pairs):
        keep = [c for c in pairs[k] if c.bird is not False]
        dropped += [(k[0], k[1], c.taxon) for c in pairs[k] if c.bird is False]
        if keep:
            pairs[k] = keep
        else:
            del pairs[k]
    return sorted(dropped)


def validate_overrides(labels: dict[str, str], files: dict[int, str]) -> None:
    errs = []
    for src, dst in NAME_ALIASES.items():
        if dst not in labels:
            errs.append(f"NAME_ALIASES {src} -> {dst}: {dst} not in BirdNET labels")
    for n, s in PLATE_ADDITIONS.items():
        if s not in labels:
            errs.append(f"PLATE_ADDITIONS {n}: {s} not in BirdNET labels")
        if n not in files:
            errs.append(f"PLATE_ADDITIONS {n}: no file for plate {n}")
    if errs:
        raise BuildError("; ".join(errs))


def credit_of(meta: dict) -> tuple[str, str | None]:
    raw = (meta.get("Credit") or {}).get("value", "")
    m = HREF_RE.search(raw)
    text = " ".join(html.unescape(TAG_RE.sub(" ", raw)).split())
    return text, (html.unescape(m[1]) if m else None)


def page_url(file_title: str) -> str:
    return COMMONS_PAGE + urllib.parse.quote(file_title.replace(" ", "_"), safe=":'(),")


def plate_title(file_title: str, label: str | None) -> str:
    if label:
        return label
    m = FILE_RE.match(file_title)
    return m[2] if m else file_title


@dataclass
class Inputs:
    """Everything fetched, so the mapping itself is a pure function."""
    labels: dict[str, str]
    files: dict[int, str]                      # plate -> canonical 'File:...'
    wd_taxa: dict[int, set[str]]               # plate -> depicted QIDs (P180)
    wd_labels: dict[int, str]                  # plate -> Wikidata plate label
    cats: dict[int, set[str]]                  # plate -> Commons (illustrations) names
    meta: dict[int, dict]                      # plate -> extmetadata
    taxa: dict[str, Taxon]                     # QID -> details
    name_qids: dict[str, set[str]]             # Commons name -> QIDs with that P225
    aves: set[str]                             # QIDs in Aves


@dataclass
class Result:
    species: dict[str, dict]
    report: dict


def build(inp: Inputs, edition: str = "havell") -> Result:
    sci = set(inp.labels)
    com = {c.lower(): s for s, c in inp.labels.items()}
    validate_overrides(inp.labels, inp.files)
    pairs: dict[tuple[int, str], list[Contribution]] = defaultdict(list)
    unresolved: set[tuple[int, str]] = set()
    unknown: set[str] = set()
    nonbird_raw: set[tuple[int, str]] = set()
    for n in sorted(inp.files):
        for q in sorted(inp.wd_taxa.get(n, ())):
            t = inp.taxa.get(q, Taxon())
            bird = q in inp.aves
            if not bird:
                nonbird_raw.add((n, "/".join(sorted(t.names)) or q))
            s, rule = resolve(t.names, t.synonyms, t.common, sci, com)
            if s:
                pairs[(n, s)].append(Contribution("wikidata", rule, q, bird))
            else:
                unresolved.add((n, "/".join(sorted(t.names)) or q))
        for name in sorted(inp.cats.get(n, ())):
            qs = inp.name_qids.get(name, set())
            if qs:
                bird = bool(qs & inp.aves)
                if not bird:
                    nonbird_raw.add((n, name))
            else:
                bird = None
                unknown.add(name)
            names, syns, common = {name}, set(), set()
            for q in qs:
                t = inp.taxa.get(q, Taxon())
                names |= t.names
                syns |= t.synonyms
                common |= t.common
            s, rule = resolve(names, syns, common, sci, com)
            if s:
                pairs[(n, s)].append(Contribution("commons", rule, name, bird))
            else:
                unresolved.add((n, name))
    for n, s in PLATE_ADDITIONS.items():
        pairs[(n, s)].append(Contribution("manual", "manual", s, True))

    denied = apply_deny(pairs)
    dropped = bird_filter(pairs)
    chosen, on_plate = choose(pairs)

    species = {}
    for s, n in sorted(chosen.items()):
        f = inp.files[n]
        credit, credit_url = credit_of(inp.meta.get(n, {}))
        species[s] = {
            "plate": n,
            "title": plate_title(f, inp.wd_labels.get(n)),
            "file": f[len("File:"):],
            "page": page_url(f),
            "credit": credit,
            "credit_url": credit_url,
            "on_plate": on_plate[n],
            "via": sorted({c.source for c in pairs[(n, s)]}),
        }

    commons_only = sorted((n, s) for (n, s), cs in pairs.items() if {c.source for c in cs} == {"commons"})
    chosen_co = [(n, s) for n, s in commons_only if chosen.get(s) == n]
    rules = Counter(r for cs in pairs.values() for r in {c.rule for c in cs})
    report = {
        "species": len(species),
        "plates": len(set(chosen.values())),
        "edition": {edition: len(species)},
        "pairs": len(pairs),
        "rules": dict(sorted(rules.items())),
        "single_species_plates": sum(1 for n in chosen.values() if on_plate[n] == 1),
        "denied": denied,
        "nonbird_dropped": sorted(nonbird_raw),
        "nonbird_contributions_dropped": dropped,
        "unknown_to_wikidata": sorted(unknown),
        "unresolved": sorted(unresolved),
        "chosen_commons_only": chosen_co,
        "other_commons_only": [p for p in commons_only if p not in chosen_co],
        "credits": dict(Counter(e["credit"] for e in species.values()).most_common()),
        "targets": {t: (species[t]["plate"], species[t]["on_plate"]) if t in species else None
                    for t in TARGETS},
    }
    return Result(species, report)


def format_report(rep: dict) -> str:
    out = [f"species mapped: {rep['species']} on {rep['plates']} distinct plates",
           f"edition split: {rep['edition']}",
           f"(plate, species) pairs after filtering: {rep['pairs']}",
           f"pairs by rule (a pair may count under two): {rep['rules']}",
           f"species whose plate shows only them: {rep['single_species_plates']}",
           f"PAIR_DENY applied: {rep['denied']}",
           f"non-bird pairs dropped ({len(rep['nonbird_dropped'])}): {rep['nonbird_dropped']}",
           f"names unknown to Wikidata, kept ({len(rep['unknown_to_wikidata'])}): {rep['unknown_to_wikidata']}",
           f"unresolved taxa ({len(rep['unresolved'])}):"]
    out += [f"  {n} {t}" for n, t in rep["unresolved"]]
    out.append(f"CHOSEN Commons-only pairs, check each by eye ({len(rep['chosen_commons_only'])}):")
    out += [f"  {n} {s}" for n, s in rep["chosen_commons_only"]]
    out.append(f"other Commons-only pairs ({len(rep['other_commons_only'])}):")
    out += [f"  {n} {s}" for n, s in rep["other_commons_only"]]
    out.append(f"credits: {rep['credits']}")
    out.append("targets (plate, on_plate):")
    out += [f"  {t}: {v}" for t, v in rep["targets"].items()]
    return "\n".join(out)


def write_json(path: Path, species: dict, edition: str, generated: str) -> None:
    doc = {"edition": edition, "generated": generated, "species": species}
    path.write_text(json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- gathering (network)
def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def gather(labels: dict[str, str]) -> Inputs:
    # 1. Wikidata plate items.
    rows = sparql(f"""SELECT ?item ?img ?taxon ?label WHERE {{
      ?item wdt:P361 wd:{WORK} .
      OPTIONAL {{ ?item wdt:P18 ?img }}
      OPTIONAL {{ ?item wdt:P180 ?taxon . ?taxon wdt:P225 ?tn }}
      OPTIONAL {{ ?item rdfs:label ?label FILTER(lang(?label) = "en") }} }}""")
    p18: dict[int, str] = {}
    wd_taxa: dict[int, set[str]] = defaultdict(set)
    wd_labels: dict[int, str] = {}
    for b in rows:
        if "img" not in b:
            continue
        f = file_from_p18(b["img"]["value"])
        n = plate_of(f)
        if n is None:
            continue
        p18[n] = f
        if "taxon" in b:
            wd_taxa[n].add(_qid(b["taxon"]["value"]))
        if "label" in b:
            wd_labels[n] = b["label"]["value"]
    # 2. Commons files and the canonical one per plate.
    titles = {p["title"] for p in commons({"generator": "categorymembers", "gcmtitle": CATEGORY,
                                           "gcmtype": "file", "gcmlimit": "max"})}
    files = canonical_files(titles, p18)
    # 3. Categories and credit of the canonical files.
    by_title = {f: n for n, f in files.items()}
    cats: dict[int, set[str]] = defaultdict(set)
    meta: dict[int, dict] = {}
    flist = sorted(by_title)
    for i in range(0, len(flist), COMMONS_BATCH):
        for p in commons({"titles": "|".join(flist[i:i + COMMONS_BATCH]), "prop": "categories|imageinfo",
                          "cllimit": "max", "iiprop": "extmetadata",
                          "iiextmetadatafilter": "Credit|LicenseShortName"}):
            n = by_title.get(p["title"])
            if n is None:
                continue
            for c in p.get("categories", []):
                t = category_taxon(c["title"])
                if t:
                    cats[n].add(t)
            for ii in p.get("imageinfo", []):
                if ii.get("extmetadata"):
                    meta[n] = ii["extmetadata"]
    # 4. Taxon details and the Aves set.
    names = sorted({t for s in cats.values() for t in s})
    name_qids: dict[str, set[str]] = defaultdict(set)
    for i in range(0, len(names), SPARQL_BATCH):
        vals = " ".join(json.dumps(x) for x in names[i:i + SPARQL_BATCH])
        for b in sparql(f"SELECT ?t ?n WHERE {{ VALUES ?n {{ {vals} }} ?t wdt:P225 ?n }}"):
            name_qids[b["n"]["value"]].add(_qid(b["t"]["value"]))
        time.sleep(PAUSE)
    qids = sorted({q for s in wd_taxa.values() for q in s} | {q for s in name_qids.values() for q in s})
    taxa: dict[str, Taxon] = defaultdict(Taxon)
    aves: set[str] = set()
    for i in range(0, len(qids), SPARQL_BATCH):
        vals = " ".join("wd:" + q for q in qids[i:i + SPARQL_BATCH])
        for b in sparql(f"""SELECT ?t ?name ?syn ?lab ?cn WHERE {{ VALUES ?t {{ {vals} }}
          OPTIONAL {{ ?t wdt:P225 ?name }}
          OPTIONAL {{ {{ ?t wdt:P1420 ?s }} UNION {{ ?s wdt:P1420 ?t }} ?s wdt:P225 ?syn }}
          OPTIONAL {{ ?t rdfs:label ?lab FILTER(lang(?lab) = "en") }}
          OPTIONAL {{ ?t wdt:P1843 ?cn FILTER(lang(?cn) = "en") }} }}"""):
            t = taxa[_qid(b["t"]["value"])]
            for k, attr in (("name", "names"), ("syn", "synonyms"), ("lab", "common"), ("cn", "common")):
                if k in b:
                    getattr(t, attr).add(b[k]["value"])
        time.sleep(PAUSE)
        for b in sparql(f"SELECT ?t WHERE {{ VALUES ?t {{ {vals} }} "
                        f"FILTER EXISTS {{ ?t wdt:P171* wd:{AVES} }} }}"):
            aves.add(_qid(b["t"]["value"]))
        time.sleep(PAUSE)
    return Inputs(labels, files, dict(wd_taxa), wd_labels, dict(cats), meta, dict(taxa),
                  dict(name_qids), aves)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", help=f"BirdNET labels file or URL (default: installed birdnetlib, then {LABELS_URL})")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "audubon.json"),
                    help="output path (default: audubon.json at the repo root)")
    args = ap.parse_args(argv)
    try:
        labels, src = load_labels(args.labels)
    except LabelsError as exc:
        print(f"error: {exc}\nlabels default: {LABELS_URL}\npass --labels PATH|URL to override",
              file=sys.stderr)
        return 2
    print(f"labels: {len(labels)} from {src}")
    try:
        inp = gather(labels)
        res = build(inp)
    except FetchError as exc:
        print(f"error: fetch failed: {exc}", file=sys.stderr)
        return 1
    except BuildError as exc:
        print(f"build error: {exc}", file=sys.stderr)
        return 1
    write_json(Path(args.out), res.species, "havell", dt.date.today().isoformat())
    print(format_report(res.report))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
