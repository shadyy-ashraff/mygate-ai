"""
inference.py — what the team calls from core/controllers/ai_engine.py

⚠️ Import from `mygate_ai` directly, not from `mygate_ai.inference`:
       from mygate_ai import load, predict, predict_many
   this module is an implementation detail. (It used to be predict.py; renamed
   because `mygate_ai.predict` was ambiguous — a function and a module with the same name.)

The contract (final — agreed with the team, §3):
    predict(flow_dict) -> {
        "is_attack": bool,
        "probability": float,          # probability that the flow is an attack (0..1)
        "attack_type": "benign" | "portscan" | "brute_force" | "dos",
        "model_version": str,
    }
    predict_many([flow, ...]) -> [ the same contract per flow, in the same order ]

flow_dict is a raw flow from cicflowmeter, as-is (live snake_case names).
Extra keys (src_ip, dst_ip, timestamp, ...) are ignored.
Missing keys → NaN → the imputer fills them with the training median.

Usage:
    from mygate_ai import load, predict, predict_many
    load()                       # once, at startup
    predict(flow)                # one flow
    predict_many(flows)          # a batch — much faster during a DoS

⚠️ All returned values are **plain Python types**, not numpy — because psycopg2
   cannot adapt numpy.float64 and raises "can't adapt type numpy.float64".
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .features import ENGINEERED, build_features
from .schema import to_canonical

BENIGN = "benign"

# The version shipped inside the package. To change it: export the new version to
# mygate_ai/models/<v>/ and change this line.
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", DEFAULT_MODEL_VERSION
)

# State loaded once at startup — not per flow
_model = None
_schema: dict | None = None
_features: list[str] = []
_base_features: list[str] = []
_classes: list[str] = []
_benign_index: int = 0
_model_version: str = ""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityError(RuntimeError):
    """model.pkl does not match the sha256 in schema.json."""


class SklearnVersionError(RuntimeError):
    """The scikit-learn version differs from the one the model was exported with."""


def load(model_dir: str | None = None, strict: bool = True) -> None:
    """Load model.pkl + schema.json **once**, at startup.

    model_dir: if None, uses the version shipped inside the package
               (mygate_ai/models/v1 — relative to __file__).
    strict   : True (default) → any failed check raises and the model is
               not loaded at all. False → warning only, **for development only**.

    Checks:
      1. sha256 of model.pkl — computed on the file bytes **before** joblib.load.
         This is not a formality: unpickling runs code, so a tampered file =
         arbitrary code execution with the appliance's privileges. The first check
         means we never unpickle a file we have no reason to trust.
      2. scikit-learn version — pickles are not guaranteed across versions and may
         load fine yet silently give different results (§3).

    The appliance does not stop if this raises: the backend catches the exception,
    AIEngine.status() returns loaded=False with the reason, and Suricata keeps running.
    See INTEGRATION.md.
    """
    global _model, _schema, _features, _base_features
    global _classes, _benign_index, _model_version

    import joblib
    import sklearn

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"No schema.json in {model_dir}")
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    model_path = os.path.join(model_dir, schema.get("model_file", "model.pkl"))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} not found")

    def _fail(exc: Exception, msg: str) -> None:
        if strict:
            raise exc
        print(f"⚠️  [mygate_ai] strict=False — {msg}")

    # --- Check 1: file integrity, before any joblib.load ---
    expected = schema.get("model_file_sha256")
    if expected:
        actual = _sha256(model_path)
        if actual != expected:
            detail = (
                f"sha256 of {model_path} does not match schema.json.\n"
                f"  expected: {expected}\n"
                f"  actual  : {actual}\n"
                f"The file changed after export. Unpickling runs code, "
                f"so we will not load it.\n"
                f"Fix: restore the original model.pkl, or use load(..., strict=False) "
                f"for development only."
            )
            _fail(ModelIntegrityError(detail), detail)
    elif strict:
        raise ModelIntegrityError(
            f"{schema_path} has no model_file_sha256 — cannot verify the integrity "
            f"of model.pkl. Re-export, or use load(..., strict=False)."
        )

    # --- Check 2: sklearn version ---
    want = schema.get("sklearn_version")
    if want and want != sklearn.__version__:
        detail = (
            f"scikit-learn version mismatch: you have {sklearn.__version__}, "
            f"the model was exported with {want}.\n"
            f"Pickles are not guaranteed across versions — it may load and give different "
            f"results silently.\n"
            f"Fix: pip install scikit-learn=={want}"
        )
        _fail(SklearnVersionError(detail), detail)

    model = joblib.load(model_path)

    _model = model
    _schema = schema
    _features = list(schema["features"])
    # Engineered features are already in schema["features"], so remove them from
    # the base list so feature_list() doesn't add them twice
    _base_features = [f for f in _features if f not in ENGINEERED]
    _classes = [str(c) for c in model.classes_]
    _benign_index = _classes.index(BENIGN)
    _model_version = str(schema["model_version"])

    print(
        f"[mygate_ai] loaded {model_dir} | version={_model_version} | "
        f"{len(_features)} features | classes={_classes}"
    )


def _require_loaded() -> None:
    if _model is None:
        raise RuntimeError(
            "Model not loaded. Call load() once at startup "
            "before the first predict()/predict_many().\n"
            "In the backend: inside core/controllers/ai_engine.py when the app starts, "
            "not on every request."
        )


def _build_X(flows: list[dict]) -> pd.DataFrame:
    """Raw flows → X ready for the model, via exactly the training path (§7).

    to_canonical(live) drops extra keys (src_ip, timestamp, ...),
    and build_features orders the columns and fills missing ones with NaN.
    """
    raw = pd.DataFrame(flows)
    df = to_canonical(raw, "live")

    cfg = {"features": _base_features, "drop_features": []}
    X = build_features(df, cfg)
    X = X.reindex(columns=_features)  # exactly the training order

    # Values may arrive as strings if the flow went through JSON/HTTP
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Inf → NaN — same as clean() does during training
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.astype("float32")


def predict_many(flows: list[dict]) -> list[dict]:
    """The same contract for every flow, **in input order**.

    Use it during a DoS: thousands of flows per second. The difference is huge —
    sklearn's fixed overhead (~3 ms) is paid once for the whole batch instead of
    once per flow (see throughput in metrics.json).
    """
    _require_loaded()
    if not flows:
        return []

    X = _build_X(flows)
    proba = _model.predict_proba(X)

    p_attack = 1.0 - proba[:, _benign_index]  # §3
    best = proba.argmax(axis=1)

    results = []
    for i in range(len(flows)):
        attack_type = _classes[best[i]]
        results.append(
            {
                "is_attack": bool(attack_type != BENIGN),
                "probability": round(float(p_attack[i]), 6),
                "attack_type": str(attack_type),
                "model_version": _model_version,
            }
        )
    return results


def predict(flow: dict) -> dict:
    """One flow. Exactly the same logic as predict_many — no separate code path."""
    return predict_many([flow])[0]
