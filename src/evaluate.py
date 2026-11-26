"""
evaluate.py — evaluation with IDS metrics (not generic ML metrics).

⬅️ Ported from: Cell 46 (comparison) + Cell 58 (confusion matrix) + Cell 60
   (per-class) + Cell 62 (ROC binary) + Cell 68 (feature importance).

What was **not** ported from the notebook (§8):
  item 7 — the notebook picked the best model by weighted accuracy/F1. That is badly wrong
          here: the data is ~98% benign, so a model that says "all clean" scores 98% accuracy
          while missing every attack. Our criteria = per-class recall + FPR + macro-F1.
  item 10 — Cell 62 used `predict_proba(...).values[:, 0]` — that was a
          cuDF DataFrame from cuML. sklearn returns numpy, so there is no .values,
          and the benign index is looked up by name, not by a fixed number.

The dict returned from here is exactly what gets written to models/<v>/metrics.json.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

BENIGN = "benign"  # must match config.classes[0] and the contract in §3

# Auto-block thresholds the team will choose from in core/decision.py
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def threshold_table(
    y_true: np.ndarray, p_attack: np.ndarray, thresholds=THRESHOLDS
) -> list[dict]:
    """For each probability threshold: how many attacks are caught and how many clean flows are wrongly blocked.

    This is the table the team picks the auto-block threshold from. The trade-off:
    a higher threshold = fewer false alarms (fewer users wrongly blocked) but more
    missed attacks. The decision is theirs (§3) — we give them the numbers.

    Note: these columns differ from binary.fpr above — that one is on argmax
    (what the model actually says), these use an explicit threshold on P(attack).
    """
    true_attack = y_true != BENIGN
    n_attack = int(true_attack.sum())
    n_benign = int((~true_attack).sum())

    rows = []
    for t in thresholds:
        flagged = p_attack >= t
        tp = int((flagged & true_attack).sum())
        fp = int((flagged & ~true_attack).sum())
        rows.append(
            {
                "threshold": t,
                "detection_rate": round(tp / n_attack, 5) if n_attack else 0.0,
                "missed_attacks": n_attack - tp,
                "fpr": round(fp / n_benign, 6) if n_benign else 0.0,
                "false_positives": fp,
            }
        )
    return rows


def _latency_ms(model, X: pd.DataFrame, n: int = 200) -> dict:
    """Prediction latency for **one** flow — what the backend actually does.

    We don't measure a batch, because ai_engine.py receives one flow at a time.

    Measured with n_jobs=1 on purpose: training uses n_jobs=-1 (every core), which is
    right for training, but for a single row the thread overhead exceeds the work itself
    (measured: 14.6 ms with n_jobs=-1 vs 2.6 ms with n_jobs=1). export.py saves
    the model with n_jobs=1, so this number is what production actually sees.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    original_n_jobs = getattr(est, "n_jobs", None)
    if original_n_jobs is not None:
        est.n_jobs = 1

    try:
        rows = [X.iloc[i : i + 1] for i in range(min(n, len(X)))]
        model.predict_proba(rows[0])  # warm-up

        times = []
        for row in rows:
            t0 = time.perf_counter()
            model.predict_proba(row)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        if original_n_jobs is not None:
            est.n_jobs = original_n_jobs

    arr = np.asarray(times)
    return {
        "n_samples": len(arr),
        "measured_with_n_jobs": 1,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def _throughput(model, X: pd.DataFrame, single_p50_ms: float) -> dict:
    """Flows per second — single vs batch.

    Matters during a DoS: the attack generates thousands of flows per second. If the backend
    calls predict() once per flow it only gets ~1/single_p50, while
    predict_many() passes the whole batch to predict_proba at once —
    orders of magnitude faster, because sklearn's fixed overhead is shared across the batch.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    original_n_jobs = getattr(est, "n_jobs", None)
    if original_n_jobs is not None:
        est.n_jobs = 1  # same setting as the saved model

    try:
        out = {"single_flows_per_sec": round(1000.0 / single_p50_ms, 1), "batch": []}
        for size in (100, 1000):
            if len(X) < size:
                continue
            batch = X.head(size)
            model.predict_proba(batch)  # warm-up
            t0 = time.perf_counter()
            model.predict_proba(batch)
            elapsed = time.perf_counter() - t0
            out["batch"].append(
                {
                    "size": size,
                    "total_ms": round(elapsed * 1000, 2),
                    "flows_per_sec": round(size / elapsed, 0),
                }
            )
    finally:
        if original_n_jobs is not None:
            est.n_jobs = original_n_jobs

    return out


def _feature_importance(model, feature_names: list[str], top: int = 20) -> list[dict]:
    """Cell 68 — top features according to the model. For the report, not for decisions."""
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(est, "feature_importances_"):
        return []
    pairs = sorted(zip(feature_names, est.feature_importances_), key=lambda p: -p[1])[
        :top
    ]
    return [{"feature": f, "importance": round(float(v), 5)} for f, v in pairs]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """All evaluation metrics → a dict ready to be saved as metrics.json."""
    classes = [str(c) for c in model.classes_]
    y_true = np.asarray(y_test, dtype=object)
    y_pred = np.asarray(model.predict(X_test), dtype=object)
    proba = model.predict_proba(X_test)  # numpy array (item 10)

    # ---------------- 1) per-class (Cell 60) ----------------
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class = {
        c: {
            "precision": round(float(report[c]["precision"]), 4),
            "recall": round(float(report[c]["recall"]), 4),
            "f1": round(float(report[c]["f1-score"]), 4),
            "support": int(report[c]["support"]),
        }
        for c in classes
    }

    # ---------------- 2) binary: benign vs attack (Cell 62) ----------------
    # P(attack) = 1 - P(benign) — the same formula as in the contract (§3)
    benign_i = classes.index(BENIGN)
    p_attack = 1.0 - proba[:, benign_i]

    true_attack = y_true != BENIGN
    pred_attack = y_pred != BENIGN
    tn, fp, fn, tp = confusion_matrix(
        true_attack, pred_attack, labels=[False, True]
    ).ravel()

    try:
        auc = round(float(roc_auc_score(true_attack, p_attack)), 4)
    except ValueError:
        auc = None  # only one class in the test set

    binary = {
        # The most important number for the team: clean flows that would wrongly be blocked
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 5),
        "detection_rate": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 4),
        "precision": round(float(tp / (tp + fp)) if (tp + fp) else 0.0, 4),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # ---------------- 3) confusion matrix (Cell 58) ----------------
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # ---------------- 4) Performance ----------------
    latency = _latency_ms(model, X_test)

    metrics = {
        "classes": classes,
        "n_test": len(y_true),
        "per_class": per_class,
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        # For reference only — not a selection criterion (§8 item 7)
        "accuracy_not_for_selection": round(float(report["accuracy"]), 4),
        "binary": binary,
        "thresholds": threshold_table(y_true, p_attack),
        "confusion_matrix": {
            "labels": classes,
            "rows": "true",
            "cols": "predicted",
            "matrix": cm.tolist(),
        },
        "latency": latency,
        "throughput": _throughput(model, X_test, latency["p50_ms"]),
        "top_features": _feature_importance(model, list(X_test.columns)),
    }

    _print_summary(metrics)
    return metrics


def _print_summary(m: dict) -> None:
    """Readable summary — sorted by recall so the worst shows first."""
    print(f"\n{'=' * 72}")
    print(f"EVALUATION  ({m['n_test']:,} test flows)")
    print(f"{'=' * 72}")

    rows = pd.DataFrame(m["per_class"]).T.sort_values("recall")
    rows["missed"] = (1 - rows["recall"]) * rows["support"]
    rows["missed"] = rows["missed"].round(0).astype(int)
    print("\nPer-class (sorted by recall — lowest = most dangerous):")
    print(rows.to_string())

    b = m["binary"]
    print("\nBinary (benign vs attack):")
    print(f"  detection rate : {b['detection_rate']:.4f}   ← share of attacks caught")
    print(f"  FPR            : {b['fpr']:.5f}   ← clean flows classified as attacks")
    print(f"  ROC-AUC        : {b['roc_auc']}")
    print(f"  TN={b['tn']:,}  FP={b['fp']:,}  FN={b['fn']:,}  TP={b['tp']:,}")

    print("\nThresholds on P(attack) — for auto-block in core/decision.py:")
    th = pd.DataFrame(m["thresholds"]).set_index("threshold")
    print(th.to_string())

    print(f"\nmacro-F1: {m['macro_f1']:.4f}  |  macro-recall: {m['macro_recall']:.4f}")
    print(
        f"accuracy: {m['accuracy_not_for_selection']:.4f}  "
        f"← for reference only, not a selection criterion (§8 item 7)"
    )

    lat, thr = m["latency"], m["throughput"]
    print(f"\nLatency/flow (n_jobs=1): p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms")
    print(
        f"  predict()      one flow at a time  : {thr['single_flows_per_sec']:>10,.0f} flow/sec"
    )
    for b in thr["batch"]:
        print(
            f"  predict_many() batch of {b['size']:>5,}     : "
            f"{b['flows_per_sec']:>10,.0f} flow/sec  ({b['total_ms']}ms per batch)"
        )

    cm = m["confusion_matrix"]
    print("\nConfusion matrix (rows = truth, columns = prediction):")
    print(
        pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]).to_string()
    )

    if m["top_features"]:
        print("\nTop 10 features:")
        for i, f in enumerate(m["top_features"][:10], 1):
            print(f"  {i:2d}. {f['feature']:22s} {f['importance']:.5f}")


def evaluate_saved(
    model_dir: str,
    test_set: str = "data/test_v1.parquet",
    config: str = "config.yaml",
) -> dict:
    """Evaluate a **saved** model on the golden test set — without retraining.

    This is how we compare v1.1 with v1 on exactly the same rows (§7).
    Features are built in schema.json order, not feature_list(config), so the
    model sees the columns in the order it was trained on.
    """
    import json

    import joblib
    import yaml

    from mygate_ai.features import build_features

    with open(os.path.join(model_dir, "schema.json"), encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    df = pd.read_parquet(test_set)
    print(f"Model : {model_dir}  (version {schema['model_version']})")
    print(f"Test  : {test_set}  ({len(df):,} rows)")

    X = build_features(df, cfg)
    missing = [f for f in schema["features"] if f not in X.columns]
    if missing:
        raise SystemExit(
            f"The golden set is missing these features: {missing[:5]}\n"
            f"So config.yaml changed after the model was trained."
        )
    X = X[schema["features"]].astype("float32")  # exactly the same order
    return evaluate(model, X, df["label"])


def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Evaluate a saved model on the golden test set (no training)"
    )
    ap.add_argument("--model-dir", default="models/v1")
    ap.add_argument("--test-set", default="data/test_v1.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write the result to <model-dir>/metrics.json",
    )
    args = ap.parse_args()

    metrics = evaluate_saved(args.model_dir, args.test_set, args.config)
    if args.write:
        path = os.path.join(args.model_dir, "metrics.json")
        # Merge, not overwrite: evaluate() doesn't produce the keys that train.py
        # adds afterwards (like train_seconds) — overwriting used to lose them.
        merged = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                merged = json.load(fh)
            kept = [k for k in merged if k not in metrics]
            if kept:
                print(f"\nKept keys from the old file: {kept}")
        merged.update(metrics)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False)
        print(f"✅ Written → {path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _main()
