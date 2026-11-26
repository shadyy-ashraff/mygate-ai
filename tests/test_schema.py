"""Check that the CIC-IDS2017 and live column maps agree — if this breaks, training and live disagree."""

from mygate_ai.schema import CANONICAL_FEATURES, CIC2017, LIVE


def test_every_cic2017_feature_exists_live():
    missing = [f for f in CANONICAL_FEATURES if f not in LIVE]
    assert missing == []


def test_no_duplicate_canonical_names():
    vals = [v for v in CIC2017.values()]
    assert len(vals) == len(set(vals))
