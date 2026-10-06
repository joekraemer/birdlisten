"""tools/build_sizes.py: filtering and output format (no openpyxl needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import facts

_SPEC = importlib.util.spec_from_file_location("build_sizes", Path(__file__).parent / "tools" / "build_sizes.py")
bs = importlib.util.module_from_spec(_SPEC)
sys.modules["build_sizes"] = bs
_SPEC.loader.exec_module(bs)


def test_build_keeps_measured_masses_only():
    avonet = {
        "Turdus migratorius": (77.34, "Dunning"),
        "Junco hyemalis": (19.6, "Updated_literature"),
        "Genus generic": (10.0, "EltonTraits_GenAvg"),     # genus average: an estimate
        "Genus modelled": (10.0, "EltonTraits_Model"),
        "Genus inferred": (10.0, "Inferred"),
        "Genus split": (10.0, "DataFromSplit"),
        "Genus broken": ("NA", "Dunning"),
    }
    birds = list(avonet) + ["Genus absent", "Turdus migratorius"]
    species, skipped = bs.build(avonet, birds)
    assert species == {"Junco hyemalis": {"mass_g": 19.6, "ref": "Updated_literature"},
                       "Turdus migratorius": {"mass_g": 77.3, "ref": "Dunning"}}
    assert skipped == {"no row": 1, "estimate": 4, "bad value": 1}


def test_read_labels_drops_non_birds(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("Turdus migratorius_American Robin\nDog_Dog\nLithobates clamitans_Green Frog\n\n")
    assert bs.read_labels(p) == ["Turdus migratorius"]


def test_dump_round_trips_and_loads(tmp_path):
    species = {"Junco hyemalis": {"mass_g": 19.6, "ref": "Dunning"},
               "Turdus migratorius": {"mass_g": 77.3, "ref": "Dunning"}}
    text = bs.dump(species, "2026-10-06")
    doc = json.loads(text)
    assert doc["species"] == species and doc["source"]["license"] == "CC BY 4.0"
    assert all(r in bs.MEASURED for r in (v["ref"] for v in doc["species"].values()))
    assert '\n "Junco hyemalis": {"mass_g": 19.6, "ref": "Dunning"},\n' in text   # one species per line
    p = tmp_path / "sizes.json"
    p.write_text(text)
    t = facts.SizeTable.load(p)
    assert dict(t.masses) == {"Junco hyemalis": 19.6, "Turdus migratorius": 77.3}


def test_shipped_table_refs_are_measured():
    doc = json.loads(bs.OUT.read_text(encoding="utf-8"))
    assert set(doc["source"]["refs"]) == set(bs.MEASURED)
    assert {v["ref"] for v in doc["species"].values()} <= set(bs.MEASURED)
