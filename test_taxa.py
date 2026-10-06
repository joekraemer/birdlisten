import pytest

import taxa


@pytest.mark.parametrize("name,bird", [
    ("Turdus migratorius", True),
    ("Psaltriparus minimus", True),
    ("Charadrius vociferus", True),       # Killdeer: "deer" in the common name only
    ("Merops apiaster", True),            # European Bee-eater
    ("Spiloptila clamans", True),         # Cricket Longtail
    ("Dog", False),
    ("Human vocal", False),
    ("  Engine ", False),
    ("Lithobates catesbeianus", False),   # American Bullfrog
    ("Pseudacris crucifer", False),       # Spring Peeper
    ("Gryllus pennsylvanicus", False),    # Fall Field Cricket
    ("Sciurus carolinensis", False),      # Eastern Gray Squirrel
    ("Canis latrans", False),             # Coyote
    ("Some newlabel", True),              # unknown: fail open, show it
])
def test_is_bird(name, bird):
    assert taxa.is_bird(name) is bird


def test_lists_do_not_overlap_and_are_clean():
    assert not taxa.NON_BIRD_LABELS & taxa.NON_BIRD_GENERA
    for g in taxa.NON_BIRD_GENERA:
        assert g.isalpha() and g[0].isupper() and g[1:].islower()
