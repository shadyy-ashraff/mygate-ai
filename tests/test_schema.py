"""يتأكد إن خريطة CIC-IDS2017 والـ live متطابقين — لو اتكسر، التدريب والـ live مش هيتفقوا."""

from mygate_ai.schema import CANONICAL_FEATURES, CIC2017, LIVE


def test_every_cic2017_feature_exists_live():
    missing = [f for f in CANONICAL_FEATURES if f not in LIVE]
    assert missing == []


def test_no_duplicate_canonical_names():
    vals = [v for v in CIC2017.values()]
    assert len(vals) == len(set(vals))
