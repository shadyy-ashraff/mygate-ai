"""
features.py — the feature list + any engineered features.

Golden rule: training and inference.py both use this file, so a feature can
never be computed one way in training and another way live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CANONICAL_FEATURES

# Engineered features (empty in v1). Example for v2:
#   def bytes_per_packet(df): return df.totlen_fwd_pkts / (df.tot_fwd_pkts + 1)
#   ENGINEERED = {"bytes_per_packet": bytes_per_packet}
ENGINEERED: dict = {}


def feature_list(cfg: dict) -> list[str]:
    feats = CANONICAL_FEATURES if cfg["features"] == "all" else cfg["features"]
    feats = [f for f in feats if f not in cfg.get("drop_features", [])]
    return feats + list(ENGINEERED)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return X with the features in order — the same function for training and live."""
    df = df.copy()
    for name, fn in ENGINEERED.items():
        df[name] = fn(df)
    feats = feature_list(cfg)
    for f in feats:
        if f not in df.columns:
            # NaN, not 0.0 — the imputer inside the Pipeline fills it with the median
            # learned during training. A zero here would be a silent lie: zero is a real
            # value for most features (flow_duration=0 is one thing, "unknown" is
            # another). It doesn't affect training — the data has all 69 features.
            df[f] = np.nan
    return df[feats]
