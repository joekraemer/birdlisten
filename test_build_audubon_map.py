"""Unit tests for tools/build_audubon_map.py on small inline fixtures.
No network: the fetch layer is monkeypatched to refuse."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_audubon_map", Path(__file__).resolve().parent / "tools" / "build_audubon_map.py")
bam = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bam      # dataclasses look the module up by name
_SPEC.loader.exec_module(bam)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, data=None, headers=None):
        raise AssertionError(f"network in a test: {url}")
    monkeypatch.setattr(bam, "_fetch", refuse)


LABELS = {
    "Meleagris gallopavo": "Wild Turkey",
    "Zonotrichia albicollis": "White-throated Sparrow",
    "Accipiter cooperii": "Cooper's Hawk",
    "Setophaga petechia": "Yellow Warbler",
    "Setophaga coronata": "Yellow-rumped Warbler",
    "Tyto alba": "Barn Owl",
    "Numenius phaeopus": "Whimbrel",
    "Larus argentatus": "Herring Gull",
    "Passerella iliaca": "Fox Sparrow",
    "Accipiter gentilis": "Northern Goshawk",
    "Psaltriparus minimus": "Bushtit",
    "Aphelocoma californica": "California Scrub-Jay",
    "Poecile hudsonicus": "Boreal Chickadee",
    "Poecile atricapillus": "Black-capped Chickadee",
    "Strix varia": "Barred Owl",
    "Sciurus carolinensis": "Eastern Gray Squirrel",
    "Anous stolidus": "Brown Noddy",
    "Anous minutus": "Black Noddy",
    "Cyanocitta stelleri": "Steller's Jay",
    "Catharus fuscescens": "Veery",
}
SCI = set(LABELS)
COM = {c.lower(): s for s, c in LABELS.items()}


def files_for(*plates):
    return {n: f"File:{n} Plate {n}.jpg" for n in plates}


def inputs(**kw):
    """A minimal consistent world: plates 1, 6, 46, 275, 353, 362, 164."""
    base = dict(
        labels=LABELS,
        files=files_for(1, 6, 46, 164, 275, 353, 362),
        wd_taxa={1: {"Q1"}, 46: {"Q46"}, 275: {"Q275"}, 353: {"Q353a"}, 362: {"Q362"}},
        wd_labels={1: "Wild Turkey"},
        cats={1: {"Meleagris gallopavo"}, 6: {"Meleagris gallopavo"},
              46: {"Strix varia", "Sciurus carolinensis"}, 164: {"Catharus fuscescens"},
              275: {"Anous stolidus", "Anous minutus"},
              353: {"Poecile atricapillus", "Poecile hudsonicus"}},
        meta={1: {"Credit": {"value": '<a href="http://pitt.example/1">University of Pittsburgh</a>'}}},
        taxa={"Q1": bam.Taxon({"Meleagris gallopavo"}), "Q46": bam.Taxon({"Strix varia"}),
              "Q275": bam.Taxon({"Anous stolidus"}), "Q353a": bam.Taxon({"Poecile atricapillus"}),
              "Q362": bam.Taxon({"Cyanocitta stelleri"}), "Qsq": bam.Taxon({"Sciurus carolinensis"}),
              "Qam": bam.Taxon({"Anous minutus"}), "Qph": bam.Taxon({"Poecile hudsonicus"}),
              "Qcf": bam.Taxon({"Catharus fuscescens"})},
        name_qids={"Meleagris gallopavo": {"Q1"}, "Strix varia": {"Q46"}, "Sciurus carolinensis": {"Qsq"},
                   "Anous stolidus": {"Q275"}, "Anous minutus": {"Qam"},
                   "Poecile atricapillus": {"Q353a"}, "Poecile hudsonicus": {"Qph"},
                   "Catharus fuscescens": {"Qcf"}},
        aves={"Q1", "Q46", "Q275", "Q353a", "Q362", "Qam", "Qph", "Qcf"},
    )
    base.update(kw)
    return bam.Inputs(**base)


# ----------------------------------------------------------------- regexes
@pytest.mark.parametrize("cat,expected", [
    ("Category:Meleagris gallopavo (illustrations)", "Meleagris gallopavo"),
    ("Category:Passerella iliaca unalaschcensis (illustrations)", "Passerella iliaca unalaschcensis"),
    ("Category:Meleagris gallopavo", None),
    ("Category:Birds of America (illustrations)", None),
    ("Category:meleagris gallopavo (illustrations)", None),
])
def test_category_regex(cat, expected):
    assert bam.category_taxon(cat) == expected


@pytest.mark.parametrize("title,plate", [
    ("File:8 White throated Sparrow.jpg", 8),
    ("File:435 American Dipper.JPEG", 435),
    ("File:1 Wild Turkey (cropped).jpg", None),
    ("File:1 Wild Turkey restored.jpg", None),
    ("File:1 Wild Turkey detail.jpg", None),
    ("File:1863 17 26 Parakeet.jpg", None),
    ("File:0 Cover.jpg", None),
    ("File:Wild Turkey.jpg", None),
    ("File:1 Wild Turkey.svg", None),
])
def test_plate_file_regex(title, plate):
    assert bam.plate_of(title) == plate


def test_file_from_p18():
    url = "http://commons.wikimedia.org/wiki/Special:FilePath/8%20White%20throated%20Sparrow.jpg"
    assert bam.file_from_p18(url) == "File:8 White throated Sparrow.jpg"


# ----------------------------------------------------------------- canonical file
def _all_plates(extra=()):
    return [f"File:{n} Bird.jpg" for n in bam.PLATES] + list(extra)


def test_canonical_file_prefers_p18_then_single():
    titles = _all_plates(["File:1 Wild Turkey (cropped).jpg", "File:2 Other.jpg"])
    out = bam.canonical_files(titles, {2: "File:2 Other.jpg"})
    assert out[1] == "File:1 Bird.jpg" and out[2] == "File:2 Other.jpg" and len(out) == 435


def test_canonical_file_ambiguous_plate_is_an_error():
    with pytest.raises(bam.BuildError, match="plate 2: 2 candidates"):
        bam.canonical_files(_all_plates(["File:2 Other.jpg"]), {})
    with pytest.raises(bam.BuildError, match="plate 2: 2 candidates"):
        bam.canonical_files(_all_plates(["File:2 Other.jpg"]), {2: "File:2 Not listed.jpg"})


def test_canonical_file_zero_candidates_is_an_error():
    titles = [t for t in _all_plates() if not t.startswith("File:7 ")]
    with pytest.raises(bam.BuildError, match="plate 7: 0 candidates"):
        bam.canonical_files(titles, {})


# ----------------------------------------------------------------- resolution
def test_resolution_order():
    r = lambda names, syn=(), com=(): bam.resolve(set(names), set(syn), set(com), SCI, COM)  # noqa: E731
    assert r(["Setophaga aestiva"]) == ("Setophaga petechia", "alias")
    # alias beats exact when both names are present
    assert r(["Tyto furcata", "Tyto alba"]) == ("Tyto alba", "alias")
    assert r(["Meleagris gallopavo"]) == ("Meleagris gallopavo", "exact")
    # exact beats trinomial
    assert r(["Passerella iliaca", "Meleagris gallopavo silvestris"]) == ("Passerella iliaca", "exact")
    assert r(["Meleagris gallopavo silvestris"]) == ("Meleagris gallopavo", "trinomial")
    # trinomial beats synonym
    assert r(["Meleagris gallopavo silvestris"], ["Tyto alba"]) == ("Meleagris gallopavo", "trinomial")
    assert r(["Astur cooperii"], ["Accipiter cooperii"]) == ("Accipiter cooperii", "synonym")
    # synonym beats common name
    assert r(["Astur cooperii"], ["Accipiter cooperii"], ["Wild Turkey"]) == ("Accipiter cooperii", "synonym")
    assert r(["Nomen novum"], [], ["veery"]) == ("Catharus fuscescens", "common")
    assert r(["Ectopistes migratorius"], [], ["Passenger Pigeon"]) == (None, None)


def test_aliases_are_the_design_list():
    assert len(bam.NAME_ALIASES) == 7
    assert bam.PLATE_ADDITIONS == {353: "Psaltriparus minimus", 362: "Aphelocoma californica"}
    assert bam.PAIR_DENY == {(46, "Sciurus carolinensis"), (275, "Anous minutus"),
                             (353, "Poecile hudsonicus")}
    assert "8746db66d1605882b80abcf2ac7706fd5c3125f9" in bam.LABELS_URL
    assert bam.USER_AGENT == "birdlisten/1.0 (https://github.com/joekraemer/birdlisten)"


# ----------------------------------------------------------------- plate choice
def test_plate_choice_sort():
    C = bam.Contribution
    wd, co = C("wikidata", "exact", "Q", True), C("commons", "exact", "n", True)
    pairs = {
        (1, "Meleagris gallopavo"): [wd, co],          # two sources
        (6, "Meleagris gallopavo"): [co],              # one source, also alone on its plate
        (369, "Ixoreus naevius"): [wd, co], (369, "Oreoscoptes montanus"): [wd],
        (433, "Ixoreus naevius"): [wd, co],            # same sources, more species
        (433, "Icterus galbula"): [co], (433, "Icterus bullockii"): [co],
        (500, "Tyto alba"): [wd], (499, "Tyto alba"): [wd],   # tie -> lower plate
    }
    chosen, on_plate = bam.choose(pairs)
    assert chosen["Meleagris gallopavo"] == 1
    assert chosen["Ixoreus naevius"] == 369 and on_plate[369] == 2 and on_plate[433] == 3
    assert chosen["Tyto alba"] == 499


def test_deny_before_on_plate_and_bird_filter():
    res = bam.build(inputs())
    s = res.species
    assert s["Psaltriparus minimus"]["plate"] == 353
    assert s["Psaltriparus minimus"]["on_plate"] == 2          # chickadee + bushtit, not the denied boreal
    assert "Anous minutus" not in s and s["Anous stolidus"]["plate"] == 275
    assert s["Anous stolidus"]["on_plate"] == 1
    assert "Sciurus carolinensis" not in s and s["Strix varia"]["on_plate"] == 1
    assert s["Meleagris gallopavo"]["plate"] == 1
    assert sorted(res.report["denied"]) == sorted(bam.PAIR_DENY)
    assert res.report["nonbird_dropped"] == [(46, "Sciurus carolinensis")]


def test_absent_deny_pair_is_a_build_error():
    cats = dict(inputs().cats)
    cats[275] = {"Anous stolidus"}       # the mis-tag is gone: the deny entry is stale
    with pytest.raises(bam.BuildError, match="275, 'Anous minutus'"):
        bam.build(inputs(cats=cats))


def test_bird_only_rule_drops_non_aves_keeps_unknown():
    cats = dict(inputs().cats)
    cats[6] = {"Meleagris gallopavo", "Alligator mississippiensis", "Astur cooperii"}
    labels = {**LABELS, "Alligator mississippiensis": "American Alligator", "Astur cooperii": "x"}
    name_qids = {**inputs().name_qids, "Alligator mississippiensis": {"Qal"}}
    taxa = {**inputs().taxa, "Qal": bam.Taxon({"Alligator mississippiensis"})}
    res = bam.build(inputs(cats=cats, labels=labels, name_qids=name_qids, taxa=taxa))
    assert "Alligator mississippiensis" not in res.species
    assert (6, "Alligator mississippiensis") in res.report["nonbird_dropped"]
    assert "Astur cooperii" in res.species                       # no Wikidata item: kept
    assert res.report["unknown_to_wikidata"] == ["Astur cooperii"]


def test_report_lists_chosen_commons_only():
    res = bam.build(inputs())
    assert res.report["chosen_commons_only"] == [(164, "Catharus fuscescens")]
    assert (6, "Meleagris gallopavo") in res.report["other_commons_only"]
    assert res.species["Catharus fuscescens"]["via"] == ["commons"]
    assert res.report["targets"]["Meleagris gallopavo"] == (1, 1)
    assert res.report["targets"]["Myadestes townsendi"] is None
    text = bam.format_report(res.report)
    assert "CHOSEN Commons-only pairs" in text and "164 Catharus fuscescens" in text


def test_entry_fields_and_json(tmp_path):
    res = bam.build(inputs())
    e = res.species["Meleagris gallopavo"]
    assert e == {"plate": 1, "title": "Wild Turkey", "file": "1 Plate 1.jpg",
                 "page": "https://commons.wikimedia.org/wiki/File:1_Plate_1.jpg",
                 "credit": "University of Pittsburgh", "credit_url": "http://pitt.example/1",
                 "on_plate": 1, "via": ["commons", "wikidata"]}
    assert res.species["Strix varia"]["title"] == "Plate 46"       # no Wikidata label
    out = tmp_path / "a.json"
    bam.write_json(out, res.species, "havell", "2026-10-03")
    text = out.read_text(encoding="utf-8")
    assert text.startswith('{\n "edition": "havell",\n "generated": "2026-10-03",\n "species": {')
    assert text.index('"Anous stolidus"') < text.index('"Meleagris gallopavo"')   # sorted keys


def test_additions_validated_against_labels():
    labels = {k: v for k, v in LABELS.items() if k != "Psaltriparus minimus"}
    with pytest.raises(bam.BuildError, match="Psaltriparus minimus not in BirdNET labels"):
        bam.build(inputs(labels=labels))
    files = {k: v for k, v in inputs().files.items() if k != 362}
    with pytest.raises(bam.BuildError, match="no file for plate 362"):
        bam.build(inputs(files=files))
    labels = {k: v for k, v in LABELS.items() if k != "Tyto alba"}
    with pytest.raises(bam.BuildError, match="Tyto furcata -> Tyto alba"):
        bam.build(inputs(labels=labels))


# ----------------------------------------------------------------- labels source
GOOD = "\n".join(f"Genus sp{i}_Bird {i}" for i in range(6100)) + "\n"


def test_labels_fallback_order(tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        return GOOD.encode()
    local = tmp_path / "labels.txt"
    local.write_text(GOOD)
    installed = tmp_path / "installed.txt"
    installed.write_text(GOOD.replace("Bird", "Inst"))

    labels, src = bam.load_labels(str(local), fetch, lambda: installed)
    assert src == str(local) and seen == [] and labels["Genus sp0"] == "Bird 0"
    labels, src = bam.load_labels(None, fetch, lambda: installed)
    assert src == str(installed) and seen == [] and labels["Genus sp0"] == "Inst 0"
    labels, src = bam.load_labels(None, fetch, lambda: None)
    assert src == bam.LABELS_URL and seen == [bam.LABELS_URL] and len(labels) == 6100
    labels, src = bam.load_labels("https://example.org/l.txt", fetch, lambda: installed)
    assert seen[-1] == "https://example.org/l.txt"


def test_labels_short_or_unreadable(tmp_path):
    short = tmp_path / "short.txt"
    short.write_text("A b_C\n" * 10)
    with pytest.raises(bam.LabelsError, match="10 lines"):
        bam.load_labels(str(short), None, lambda: None)
    with pytest.raises(bam.LabelsError, match="cannot read"):
        bam.load_labels(str(tmp_path / "nope.txt"), None, lambda: None)

    def broken(url):
        raise bam.FetchError(f"{url}: boom")
    with pytest.raises(bam.LabelsError, match="cannot read"):
        bam.load_labels(None, broken, lambda: None)


def test_main_exits_2_on_bad_labels(tmp_path, capsys):
    assert bam.main(["--labels", str(tmp_path / "nope.txt"), "--out", str(tmp_path / "o.json")]) == 2
    err = capsys.readouterr().err
    assert bam.LABELS_URL in err and "--labels" in err
    assert not (tmp_path / "o.json").exists()


def test_main_names_failing_url(tmp_path, capsys, monkeypatch):
    local = tmp_path / "labels.txt"
    local.write_text(GOOD)

    def broken(url, data=None, headers=None):
        raise bam.FetchError(f"{url}: HTTP Error 503")
    monkeypatch.setattr(bam, "_fetch", broken)
    assert bam.main(["--labels", str(local), "--out", str(tmp_path / "o.json")]) == 1
    assert bam.SPARQL_URL in capsys.readouterr().err
