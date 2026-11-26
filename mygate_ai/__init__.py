"""
mygate_ai — MyGate's network-flow classification model.

This package is **everything** the backend needs. It contains no training code.

    from mygate_ai import load, predict, predict_many

    load()                     # once, at startup
    predict(flow)              # one flow    → dict
    predict_many(flows)        # a batch     → list[dict]

Import everything from `mygate_ai` directly — the internal modules
(`inference`, `features`, `schema`) are implementation details and may change.

The full contract is in INTEGRATION.md.
"""

from __future__ import annotations

from .inference import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_VERSION,
    ModelIntegrityError,
    SklearnVersionError,
    load,
    predict,
    predict_many,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_VERSION",
    "ModelIntegrityError",
    "SklearnVersionError",
    "load",
    "predict",
    "predict_many",
]

# Package version (PEP 440) — not the same as the model version.
# The model version is DEFAULT_MODEL_VERSION and is also in schema.json.
__version__ = "1.0.0"
