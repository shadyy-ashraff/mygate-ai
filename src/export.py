"""
export.py — يطلّع ملفات التسليم في models/<model_version>/

⬅️ بدل Cell 74. الفرق عن النوت‑بوك:
  - النوت‑بوك كان بيحفظ 4 موديلات + scaler.pkl + label_encoder.pkl + .h5.
    إحنا بنحفظ **Pipeline واحد** فيه الـ imputer جوّه → مفيش scaler (§8 بند ٣)
    ولا label encoder (§8 بند ٦) ولا TensorFlow (§8 بند ٨).
  - بنحفظ معاه schema.json — العقد اللي الباك‑إند و mygate_ai بيقرأوا منه.

المخرجات:
    model.pkl     — الـ Pipeline كله (imputer + model)، محفوظ بـ n_jobs=1
    schema.json   — الـ features بالترتيب + الـ classes + النسخ + hash الموديل
    metrics.json  — نتايج evaluate()
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

CICFLOWMETER_VERSION = "0.2.0"  # §3 — مصدر الـ flows اللايف


def _sha256(path: str) -> str:
    """هاش الملف — التيم بيتأكد بيه إن model.pkl اللي عندهم هو اللي اتصدّر."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordered_features(model, cfg: dict) -> list[str]:
    """قايمة الـ features بالترتيب اللي الموديل **اتدرّب عليه فعلاً**.

    بناخدها من الـ imputer المدرَّب مش من feature_list(cfg)، عشان لو حد غيّر
    config.yaml بعد التدريب وقبل الـ export، الملف يفضل صادق. والترتيب ده
    حرفي: sklearn مش بيتحقق من الأسماء لما بياخد numpy، فعمود في مكان
    عمود تاني = نتيجة غلط **بصمت**.
    """
    from_cfg = feature_list(cfg)
    step = model.named_steps.get("impute") if hasattr(model, "named_steps") else None
    names = getattr(step, "feature_names_in_", None)
    if names is None:
        return from_cfg

    fitted = [str(c) for c in names]
    if fitted != from_cfg:
        print("⚠️  feature_list(config) مختلفة عن اللي الموديل اتدرّب عليه!")
        print(f"    الموديل: {len(fitted)} feature  |  config: {len(from_cfg)}")
        print("    بنكتب اللي الموديل اتدرّب عليه في schema.json.")
    return fitted


def _prepare_dir(out_dir: str, force: bool) -> None:
    """سياسة الكتابة: مانمسحش موديل متسلّم للتيم بالغلط."""
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not force:
            raise SystemExit(
                f"\n❌ {out_dir} موجود خلاص وفيه ملفات.\n"
                f"   غيّر model_version في config.yaml (v2, v1.1, ...) عشان "
                f"متمسحش موديل متسلّم،\n"
                f"   أو ضيف --force لو انت متأكد إنك عايز تكتب فوقه."
            )
        print(f"⚠️  --force: بنكتب فوق {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def export(
    model,
    cfg: dict,
    metrics: dict,
    info: dict,
    force: bool = False,
) -> str:
    """يحفظ model.pkl + schema.json + metrics.json ويرجّع مسار الفولدر."""
    out_dir = os.path.join(cfg["paths"]["models_dir"], str(cfg["model_version"]))
    _prepare_dir(out_dir, force)

    # ---- 1) الموديل ----
    # n_jobs=1: التدريب بيستخدم كل الأنوية، لكن الباك‑إند بيتنبّأ لـ flow واحد
    # في المرة — وتوزيع 100 شجرة على threads لصف واحد بيكلّف أكتر من الشغل
    # نفسه (14.6ms مقابل 2.7ms). الموديل المشحون مضبوط للـ flow الواحد.
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if getattr(est, "n_jobs", None) is not None:
        est.n_jobs = 1

    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)

    # ---- 2) العقد ----
    feats = _ordered_features(model, cfg)
    schema = {
        "model_version": str(cfg["model_version"]),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # --- المدخلات ---
        "cicflowmeter_version": CICFLOWMETER_VERSION,
        "input": (
            "raw flow dict from cicflowmeter (live snake_case keys); "
            "extra keys ignored, missing → NaN → imputer"
        ),
        "features": feats,  # الترتيب حرفي
        "n_features": len(feats),
        "drop_features": list(cfg.get("drop_features") or []),
        # --- المخرجات ---
        "classes": [str(c) for c in model.classes_],
        "positive_probability": "1 - P(benign)",
        "label_map": dict(cfg["label_map"]),
        # --- الموديل ---
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
        # --- البيئة: التيم لازم يثبّتوا نفس النسخ (§3) ---
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        # --- الداتا ---
        "sources": list(info.get("sources") or []),
        "n_train": int(info["n_train"]),
        "n_test": int(info["n_test"]),
        "random_state": cfg["random_state"],
        "test_size": cfg["test_size"],
        "smoke_run_nrows": info.get("nrows"),
    }
    _write_json(os.path.join(out_dir, "schema.json"), schema)

    # ---- 3) النتايج ----
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
    print(f"  features       : {schema['n_features']} (بالترتيب)")
    print(f"  classes        : {schema['classes']}")
    print(f"  sklearn        : {schema['sklearn_version']}   ← التيم يثبّتوا نفسها")
    print(f"  n_jobs         : {schema['n_jobs_at_export']}")
    print(f"  sha256         : {schema['model_file_sha256'][:16]}...")
    if schema.get("smoke_run_nrows"):
        print("\n  ⚠️  دي مخرجات SMOKE run — مش للتسليم.")


if __name__ == "__main__":
    sys.exit(
        "export.py مكتبة — بتتنادى من src/train.py.\n"
        "للتدريب والتصدير:  python -m src.train"
    )
