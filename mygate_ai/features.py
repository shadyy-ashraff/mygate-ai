"""
features.py — قايمة الـ features + أي feature محسوبة.

القاعدة الذهبية: التدريب و inference.py الاتنين بيستخدموا الملف ده،
فمستحيل feature تتحسب بطريقة في التدريب وبطريقة تانية لايف.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CANONICAL_FEATURES

# الـ features المحسوبة (فاضية في v1). مثال لـ v2:
#   def bytes_per_packet(df): return df.totlen_fwd_pkts / (df.tot_fwd_pkts + 1)
#   ENGINEERED = {"bytes_per_packet": bytes_per_packet}
ENGINEERED: dict = {}


def feature_list(cfg: dict) -> list[str]:
    feats = CANONICAL_FEATURES if cfg["features"] == "all" else cfg["features"]
    feats = [f for f in feats if f not in cfg.get("drop_features", [])]
    return feats + list(ENGINEERED)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """يرجّع X بالـ features بالترتيب — نفس الدالة في التدريب والـ live."""
    df = df.copy()
    for name, fn in ENGINEERED.items():
        df[name] = fn(df)
    feats = feature_list(cfg)
    for f in feats:
        if f not in df.columns:
            # NaN مش 0.0 — الـ imputer جوا الـ Pipeline بيملّيها بالـ median
            # اللي اتعلمه من التدريب. صفر هنا كان هيبقى كذبة صامتة: صفر قيمة
            # حقيقية لمعظم الـ features (flow_duration=0 حاجة، و"مش عارف" حاجة
            # تانية خالص). مابيأثرش على التدريب — الداتا فيها الـ 69 كاملة.
            df[f] = np.nan
    return df[feats]
