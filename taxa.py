"""Which BirdNET labels are birds.

BirdNET v2.4's 6,522 labels are mostly birds, but they also include sound
events (Dog, Engine, Human vocal, ...) and frogs, insects and mammals. The
detections table keeps every row; the page and ntfy show birds only.

This is a denylist. NON_BIRD_GENERA came from diffing birdnetlib 0.18.0's
BirdNET_GLOBAL_6K_V2.4_Labels.txt against the eBird/Clements taxonomy
(api.ebird.org/v2/ref/taxonomy/ebird) on 2026-10-06: every label whose genus
is unknown to eBird, minus the birds eBird files under a newer genus
(Ixobrychus, Ciccaba, Milvago, ...), which were checked by hand. A denylist
fails open: a label added by a future model shows up rather than vanishing.
"""

from __future__ import annotations

# BirdNET's non-species classes; each label is "<name>_<name>".
NON_BIRD_LABELS = frozenset({
    "Dog", "Engine", "Environmental", "Fireworks", "Gun", "Human non-vocal",
    "Human vocal", "Human whistle", "Noise", "Power tools", "Siren",
})

NON_BIRD_GENERA = frozenset({
    # frogs and toads
    "Acris", "Anaxyrus", "Dryophytes", "Eleutherodactylus", "Gastrophryne", "Hyliola",
    "Incilius", "Lithobates", "Pseudacris", "Scaphiopus", "Spea",
    # crickets, katydids, bees
    "Allonemobius", "Amblycorypha", "Anaxipha", "Apis", "Atlanticus", "Conocephalus",
    "Cyrtoxipha", "Eunemobius", "Gryllus", "Microcentrum", "Miogryllus",
    "Neoconocephalus", "Neonemobius", "Oecanthus", "Orchelimum", "Orocharis",
    "Phyllopalpus", "Pterophylla", "Scudderia",
    # mammals
    "Alouatta", "Canis", "Odocoileus", "Sciurus", "Tamias", "Tamiasciurus",
})


def is_bird(scientific_name: str) -> bool:
    """False for BirdNET's known non-bird labels; True for everything else."""
    name = scientific_name.strip()
    if name in NON_BIRD_LABELS:
        return False
    return name.split(" ", 1)[0] not in NON_BIRD_GENERA
