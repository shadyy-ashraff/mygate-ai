"""
export.py — writes the delivery files to models/<model_version>/

⬅️ Replaces Cell 74. Differences from the notebook:
  - The notebook saved 4 models + scaler.pkl + label_encoder.pkl + .h5.
    We save **one Pipeline** with the imputer inside → no scaler (§8 item 3),
    no label encoder (§8 item 6) and no TensorFlow (§8 item 8).
  - We save schema.json with it — the contract the backend and mygate_ai read from.

Outputs:
    model.pkl     — the whole Pipeline (imputer + model), saved with n_jobs=1
    schema.json   — ordered features + classes + versions + the model hash
    metrics.json  — evaluate() results
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from mygate_ai.features import feature_list

CICFLOWMETER_VERSION = "0.2.0"  # §3 — source of live flows (should be 0.5.0, see KNOWN_ISSUES.md)


def _sha256(path: str) -> str:
    """File hash — lets the team verify their model.pkl is the one that was exported."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordered_features(model, cfg: dict) -> list[str]:
    """The feature list in the order the model was **actually trained on**.

    Taken from the fitted imputer, not from feature_list(cfg), so if someone changes
    config.yaml after training and before export, the file stays truthful. The order
    is literal: sklearn doesn't check names when given numpy, so a column in the
    wrong place = a wrong result, **silently**.
    """
    from_cfg = feature_list(cfg)
    step = model.named_steps.get("impute") if hasattr(model, "named_steps") else None
    names = getattr(step, "feature_names_in_", None)
    if names is None:
        return from_cfg

    fitted = [str(c) for c in names]
    if fitted != from_cfg:
        print("⚠️  feature_list(config) differs from what the model was trained on!")
        print(f"    model: {len(fitted)} features  |  config: {len(from_cfg)}")
        print("    Writing what the model was trained on to schema.json.")
    return fitted


def _prepare_dir(out_dir: str, force: bool) -> None:
    """Write policy: never wipe a model already delivered to the team by accident."""
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not force:
            raise SystemExit(
                f"\n❌ {out_dir} already exists and has files.\n"
                f"   Change model_version in config.yaml (v2, v1.1, ...) so you "
                f"don't overwrite a delivered model,\n"
                f"   or pass --force if you really want to overwrite it."
            )
        print(f"⚠️  --force: overwriting {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def export(
    model,
    cfg: dict,
    metrics: dict,
    info: dict,
    force: bool = False,
) -> str:
    """Save model.pkl + schema.json + metrics.json and return the folder path."""
    out_dir = os.path.join(cfg["paths"]["models_dir"], str(cfg["model_version"]))
    _prepare_dir(out_dir, force)

    # ---- 1) The model ----
    # n_jobs=1: training uses every core, but the backend predicts one flow
    # at a time — spreading 100 trees over threads for one row costs more than the
    # work itself (14.6 ms vs 2.7 ms). The shipped model is tuned for single flows.
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if getattr(est, "n_jobs", None) is not None:
        est.n_jobs = 1

    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)

    # ---- 2) The contract ----
    feats = _ordered_features(model, cfg)
    schema = {
        "model_version": str(cfg["model_version"]),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # --- Inputs ---
        "cicflowmeter_version": CICFLOWMETER_VERSION,
        "input": (
            "raw flow dict from cicflowmeter (live snake_case keys); "
            "extra keys ignored, missing → NaN → imputer"
        ),
        "features": feats,  # order is literal
        "n_features": len(feats),
        "drop_features": list(cfg.get("drop_features") or []),
        # --- Outputs ---
        "classes": [str(c) for c in model.classes_],
        "positive_probability": "1 - P(benign)",
        "label_map": dict(cfg["label_map"]),
        # --- Model ---
        "model": {
            "type": cfg["model"]["type"],
            "params": dict(cfg["model"].get("params") or {}),
            "pipeline": (
                [name for name, _ in model.steps] if hasattr(model, "steps") else []
            ),
        },
        "model_file": "model.pkl",
        "model_file_sha256": _sha256(model_path),
        "n_jobs_at_export": getattr(est, "n_jobs", None),
        # --- Environment: the team must install the same versions (§3) ---
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        # --- Data ---
        "sources": list(info.get("sources") or []),
        "n_train": int(info["n_train"]),
        "n_test": int(info["n_test"]),
        "random_state": cfg["random_state"],
        "test_size": cfg["test_size"],
        "smoke_run_nrows": info.get("nrows"),
    }
    _write_json(os.path.join(out_dir, "schema.json"), schema)

    # ---- 3) Results ----
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)

    _print_summary(out_dir, schema)
    return out_dir


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _print_summary(out_dir: str, schema: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"EXPORT → {out_dir}")
    print(f"{'=' * 72}")
    for name in ("model.pkl", "schema.json", "metrics.json"):
        size_kb = os.path.getsize(os.path.join(out_dir, name)) / 1024
        unit = "KB" if size_kb < 1024 else "MB"
        val = size_kb if size_kb < 1024 else size_kb / 1024
        print(f"  {name:14s} {val:8.1f} {unit}")
    print(f"\n  model_version  : {schema['model_version']}")
    print(f"  features       : {schema['n_features']} (ordered)")
    print(f"  classes        : {schema['classes']}")
    print(f"  sklearn        : {schema['sklearn_version']}   ← the team must install this exact version")
    print(f"  n_jobs         : {schema['n_jobs_at_export']}")
    print(f"  sha256         : {schema['model_file_sha256'][:16]}...")
    if schema.get("smoke_run_nrows"):
        print("\n  ⚠️  This is SMOKE run output — not for delivery.")


if __name__ == "__main__":
    sys.exit(
        "export.py is a library — it is called from src/train.py.\n"
        "To train and export:  python -m src.train"
    )
