"""tools/build_cutouts.py: run selection, pruning and override checks (no torch
needed; Models() is refused, so a test that would load the 4 GB of weights fails)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

_SPEC = importlib.util.spec_from_file_location("build_cutouts", Path(__file__).parent / "tools" / "build_cutouts.py")
bc = importlib.util.module_from_spec(_SPEC)
sys.modules["build_cutouts"] = bc
_SPEC.loader.exec_module(bc)

PAGE = "https://commons.wikimedia.org/wiki/File:"


def _entry(plate: int) -> dict:
    return {"plate": plate, "title": f"Plate {plate}", "file": f"{plate}.jpg", "page": f"{PAGE}{plate}.jpg",
            "credit": "University of Pittsburgh", "on_plate": 1}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch):
    """A fake audubon.json (3 species, one now in Fugleramme), cutouts/ and overrides."""
    aud = tmp_path / "audubon.json"
    aud.write_text(json.dumps({"species": {"Aa aa": _entry(1), "Bb bb": _entry(2), "Cc cc": _entry(3)}}))
    out = tmp_path / "cutouts"
    out.mkdir()
    over = tmp_path / "overrides.json"
    over.write_text("{}")
    monkeypatch.setattr(bc.frame, "AUDUBON_MAP", aud)
    monkeypatch.setattr(bc, "OUT", out)
    monkeypatch.setattr(bc, "OVERRIDES", over)
    monkeypatch.setattr(bc, "fugleramme_stems", lambda ref: {"cc-cc"})
    monkeypatch.setattr(bc, "Models", lambda: pytest.fail("models loaded"))
    return tmp_path


def _index(out: Path, names: list[str]) -> None:
    species = {}
    for n in names:
        s = bc.frame.stem(n)
        Image.new("RGBA", (4, 4)).save(out / f"{s}.webp", "WEBP")
        species[n] = {"file": f"{s}.webp", "source": "audubon", "plate": 1, "page": PAGE + "x"}
    (out / "index.json").write_text(json.dumps({"generated": "2026-01-01", "species": species}))


def test_nothing_to_build_loads_no_models_and_leaves_index_alone(repo: Path):
    _index(repo / "cutouts", ["Aa aa", "Bb bb"])
    before = (repo / "cutouts" / "index.json").read_text()
    assert bc.main(["--work", str(repo / "w")]) == 0
    assert (repo / "cutouts" / "index.json").read_text() == before        # no date churn
    assert bc.main(["--work", str(repo / "w"), "--review"]) == 0           # still no models


def test_prunes_species_fugleramme_gained_or_skipped(repo: Path, capsys):
    out = repo / "cutouts"
    _index(out, ["Aa aa", "Bb bb", "Cc cc"])           # Cc cc: Fugleramme has it now
    (repo / "overrides.json").write_text(json.dumps({"Bb bb": {"skip": True}}))
    Image.new("RGBA", (4, 4)).save(out / "stray.webp", "WEBP")
    assert bc.main(["--work", str(repo / "w")]) == 0
    doc = json.loads((out / "index.json").read_text())
    assert set(doc["species"]) == {"Aa aa"}
    assert sorted(p.name for p in out.glob("*.webp")) == ["aa-aa.webp"]
    assert "removed stray.webp" in capsys.readouterr().out


def test_species_and_full_are_exclusive_and_names_checked(repo: Path):
    with pytest.raises(SystemExit):
        bc.main(["--work", str(repo / "w"), "--full", "--species", "Aa aa"])
    with pytest.raises(SystemExit, match="not an Audubon-only species: Cc cc, Zz zz"):
        bc.main(["--work", str(repo / "w"), "--species", "Cc cc", "Zz zz"])


@pytest.mark.parametrize("override, msg", [
    ({"Zz zz": {"skip": True}}, "stale overrides"),
    ({"Aa aa": []}, "must be an object"),
    ({"Aa aa": {"box": [10, 10, 5, 20]}}, "needs a box"),
    ({"Aa aa": {"box": [1, 2, "3", 4]}}, "needs a box"),
    ({"Aa aa": {"box": [1, 2, 3, 4], "neg": [[1]]}}, "neg must be a list"),
    ({"Aa aa": {"box": [1, 2, 3, 4], "source": {"file": "x.jpg", "page": "https://example.com/x"}}},
     "source needs"),
])
def test_bad_overrides_fail(repo: Path, override, msg):
    (repo / "overrides.json").write_text(json.dumps(override))
    with pytest.raises(SystemExit, match=msg):
        bc.main(["--work", str(repo / "w")])


def test_repo_overrides_are_valid():
    """The committed overrides pass the same checks against the real table
    (Fugleramme membership is not checked here: no network)."""
    aud = json.loads(bc.frame.AUDUBON_MAP.read_text())["species"]
    doc = bc.load_overrides(bc.OVERRIDES, aud)
    assert "Larus delawarensis" in doc and doc["Larus delawarensis"]["source"]["credit_url"].startswith("http")
