"""Build sizes.json: a curated adult body mass per species, for the pop-up's
size line when Wikidata has none (#5). Offline; not in the image.

  uv run --no-project --with openpyxl==3.1.5 python tools/build_sizes.py \\
      --avonet .agents/avonet/ELEData/TraitData/AVONET2_eBird.xlsx \\
      --labels BirdNET_GLOBAL_6K_V2.4_Labels.txt

Inputs:
  AVONET (Tobias et al. 2022, Ecology Letters 25:581-597, CC BY 4.0), the
    eBird-taxonomy sheet of ELEData.zip from
    https://figshare.com/articles/dataset/16586228 (file 38429873). `Mass`
    is the species average in grams, males and females together.
  BirdNET's v2.4 label file, as shipped in birdnetlib 0.18.0
    (src/birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Labels.txt).
    Only its birds (taxa.is_bird) go in the table.

Only measured masses are kept (MEASURED below). AVONET also fills gaps with
genus averages, mass-length models, a relative's value and a parent
species' value after a split; those are estimates and a card should not show
them as facts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import taxa  # noqa: E402

OUT = ROOT / "sizes.json"
SHEET = "AVONET2_eBird"
MEASURED = {
    "Dunning": "Dunning (2008), CRC Handbook of Avian Body Masses",
    "EltonTraits_Other": "EltonTraits 1.0 (Wilman et al. 2014), from the literature",
    "Updated_literature": "published literature, per AVONET",
    "Updated_live.sample": "museum labels or live birds, per AVONET",
}
SOURCE = {
    "name": "AVONET",
    "citation": ("Tobias, J. A. et al. (2022). AVONET: morphological, ecological and geographical "
                 "data for all birds. Ecology Letters 25: 581\u2013597."),
    "url": "https://doi.org/10.1111/ele.13898",
    "data_url": "https://figshare.com/articles/dataset/16586228",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "sheet": SHEET,
    "refs": MEASURED,
}
MASS_LO, MASS_HI = 1.0, 200000.0     # g; a hummingbird to an ostrich


def read_avonet(path: Path) -> dict[str, tuple[float, str]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    rows = wb[SHEET].iter_rows(values_only=True)
    hdr = next(rows)
    i_sp, i_mass, i_src = hdr.index("Species2"), hdr.index("Mass"), hdr.index("Mass.Source")
    out = {}
    for r in rows:
        if r[i_sp]:
            out[str(r[i_sp]).strip()] = (r[i_mass], str(r[i_src]).strip())
    return out


def read_labels(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sci = line.split("_", 1)[0].strip()
        if sci and taxa.is_bird(sci):
            out.append(sci)
    return out


def build(avonet: dict[str, tuple[float, str]], birds: list[str]) -> tuple[dict, dict[str, int]]:
    species, skipped = {}, {"no row": 0, "estimate": 0, "bad value": 0}
    for sci in sorted(set(birds)):
        row = avonet.get(sci)
        if row is None:
            skipped["no row"] += 1
            continue
        mass, ref = row
        if ref not in MEASURED:
            skipped["estimate"] += 1
            continue
        if not isinstance(mass, (int, float)) or not MASS_LO <= mass <= MASS_HI:
            skipped["bad value"] += 1
            continue
        species[sci] = {"mass_g": round(float(mass), 1), "ref": ref}
    return species, skipped


def dump(species: dict, generated: str) -> str:
    """One species per line, so a rebuild diffs readably."""
    head = json.dumps({"generated": generated, "source": SOURCE}, ensure_ascii=False, indent=1)[:-2]
    lines = [f" {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
             for k, v in species.items()]
    return head + ',\n "species": {\n' + ",\n".join(lines) + "\n }\n}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--avonet", type=Path, required=True, help="AVONET2_eBird.xlsx")
    ap.add_argument("--labels", type=Path, required=True, help="BirdNET v2.4 label file")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    birds = read_labels(args.labels)
    species, skipped = build(read_avonet(args.avonet), birds)
    args.out.write_text(dump(species, dt.datetime.now(dt.timezone.utc).date().isoformat()), encoding="utf-8")
    print(f"{args.out.name}: {len(species)} of {len(set(birds))} BirdNET birds; skipped "
          + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
