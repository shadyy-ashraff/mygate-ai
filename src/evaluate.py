"""
evaluate.py — التقييم بمقاييس IDS (مش بمقاييس ML عامة).

⬅️ متنقول من: Cell 46 (المقارنة) + Cell 58 (confusion matrix) + Cell 60
   (per-class) + Cell 62 (ROC binary) + Cell 68 (feature importance).

اللي **ما**اتنقلش من النوت‑بوك (§8):
  بند ٧ — النوت‑بوك كان بيختار أحسن موديل بـ weighted accuracy/F1. ده غلط خطير
          هنا: الداتا ~98% benign، فموديل بيقول "كله سليم" بياخد 98% accuracy
          وهو بيفوّت كل الهجمات. القرار عندنا = recall لكل class + FPR + macro-F1.
  بند ١٠ — Cell 62 كان بيكتب `predict_proba(...).values[:, 0]` — ده كان
          cuDF DataFrame من cuML. sklearn بيرجّع numpy، فمفيش .values،
          والـ index بتاع benign بنجيبه بالاسم مش برقم ثابت.

الـ dict اللي بيرجع من هنا هو نفسه اللي بيتكتب في models/<v>/metrics.json.
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

BENIGN = "benign"  # لازم يطابق config.classes[0] والعقد في §3

# عتبات الـ auto-block اللي التيم هيختار منها في core/decision.py
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def threshold_table(
    y_true: np.ndarray, p_attack: np.ndarray, thresholds=THRESHOLDS
) -> list[dict]:
    """لكل عتبة على probability: كام هجمة بتتمسك وكام flow سليم بيتقفل غلط.

    ده الجدول اللي التيم بيختار منه عتبة الـ auto-block. المقايضة:
    عتبة أعلى = false alarms أقل (مستخدمين مش هيتقفلوا غلط) بس هجمات
    فايتة أكتر. القرار بتاعهم (§3) — إحنا بنديهم الأرقام.

    ملاحظة: الأعمدة دي غير الـ binary.fpr اللي فوق — دي على argmax
    (اللي الموديل بيقوله فعلاً)، ودي على عتبة صريحة على P(attack).
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
    """زمن التنبؤ لـ flow **واحد** — ده اللي الباك‑إند هيعمله فعلاً.

    مش بنقيس batch، لأن ai_engine.py بيستقبل flow واحد في المرة.

    بنقيس بـ n_jobs=1 عن قصد: التدريب بيستخدم n_jobs=-1 (كل الأنوية) وده
    صح للتدريب، لكن على صف واحد تكلفة توزيع الـ threads أكبر من الشغل نفسه
    (قِسناها: 14.6ms بـ n_jobs=-1 مقابل 2.6ms بـ n_jobs=1). export.py بيحفظ
    الموديل بـ n_jobs=1، فالرقم ده هو اللي الإنتاج هيشوفه فعلاً.
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
    """كام flow في الثانية — single مقابل batch.

    مهم وقت الـ DoS: الهجوم بيولّد آلاف الـ flows في الثانية. لو الباك‑إند
    بينادي predict() لكل flow لوحده هياخد ~1/single_p50 فقط، بينما
    predict_many() بتعدّي الدفعة كلها لـ predict_proba مرة واحدة —
    الفرق أوامر مقدار، لأن تكلفة sklearn الثابتة بتتوزّع على الدفعة.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    original_n_jobs = getattr(est, "n_jobs", None)
    if original_n_jobs is not None:
        est.n_jobs = 1  # نفس إعداد الموديل المحفوظ

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
    """Cell 68 — أهم الـ features حسب الموديل. للتقرير، مش للقرار."""
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(est, "feature_importances_"):
        return []
    pairs = sorted(zip(feature_names, est.feature_importances_), key=lambda p: -p[1])[
        :top
    ]
    return [{"feature": f, "importance": round(float(v), 5)} for f, v in pairs]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """كل مقاييس التقييم → dict جاهز يتحفظ في metrics.json."""
    classes = [str(c) for c in model.classes_]
    y_true = np.asarray(y_test, dtype=object)
    y_pred = np.asarray(model.predict(X_test), dtype=object)
    proba = model.predict_proba(X_test)  # numpy array (بند ١٠)

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

    # ---------------- 2) binary: benign ضد هجوم (Cell 62) ----------------
    # P(attack) = 1 - P(benign) — نفس المعادلة اللي في العقد (§3)
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
        auc = None  # class واحدة بس في الـ test set

    binary = {
        # أخطر رقم للتيم: flows سليمة هتتقفل على المستخدم غلط
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

    # ---------------- 4) الأداء ----------------
    latency = _latency_ms(model, X_test)

    metrics = {
        "classes": classes,
        "n_test": len(y_true),
        "per_class": per_class,
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        # للعلم بس — مش أساس الاختيار (§8 بند ٧)
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
    """ملخّص مقروء — مرتّب بالـ recall عشان الأخطر يبان الأول."""
    print(f"\n{'=' * 72}")
    print(f"EVALUATION  ({m['n_test']:,} test flows)")
    print(f"{'=' * 72}")

    rows = pd.DataFrame(m["per_class"]).T.sort_values("recall")
    rows["missed"] = (1 - rows["recall"]) * rows["support"]
    rows["missed"] = rows["missed"].round(0).astype(int)
    print("\nPer-class (مرتّب بالـ recall — الأقل = الأخطر):")
    print(rows.to_string())

    b = m["binary"]
    print("\nBinary (benign ضد هجوم):")
    print(f"  detection rate : {b['detection_rate']:.4f}   ← نسبة الهجمات اللي اتمسكت")
    print(f"  FPR            : {b['fpr']:.5f}   ← flows سليمة اتصنّفت هجوم")
    print(f"  ROC-AUC        : {b['roc_auc']}")
    print(f"  TN={b['tn']:,}  FP={b['fp']:,}  FN={b['fn']:,}  TP={b['tp']:,}")

    print("\nThresholds على P(attack) — للـ auto-block في core/decision.py:")
    th = pd.DataFrame(m["thresholds"]).set_index("threshold")
    print(th.to_string())

    print(f"\nmacro-F1: {m['macro_f1']:.4f}  |  macro-recall: {m['macro_recall']:.4f}")
    print(
        f"accuracy: {m['accuracy_not_for_selection']:.4f}  "
        f"← للعلم بس، مش أساس الاختيار (§8 بند ٧)"
    )

    lat, thr = m["latency"], m["throughput"]
    print(f"\nLatency/flow (n_jobs=1): p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms")
    print(
        f"  predict()      flow واحد في المرة : {thr['single_flows_per_sec']:>10,.0f} flow/sec"
    )
    for b in thr["batch"]:
        print(
            f"  predict_many() دفعة {b['size']:>5,}         : "
            f"{b['flows_per_sec']:>10,.0f} flow/sec  ({b['total_ms']}ms للدفعة)"
        )

    cm = m["confusion_matrix"]
    print("\nConfusion matrix (صفوف = الحقيقة، أعمدة = التنبؤ):")
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
    """يقيّم موديل **محفوظ** على الـ golden test set — من غير إعادة تدريب.

    ده اللي بنستخدمه لمقارنة v1.1 بـ v1 على نفس الصفوف بالظبط (§7).
    الـ features بنبنيها بترتيب schema.json مش feature_list(config)، عشان
    الموديل يشوف الأعمدة بنفس الترتيب اللي اتدرّب عليه.
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
            f"الـ golden set مفيهوش الـ features دي: {missing[:5]}\n"
            f"يعني config.yaml اتغيّر بعد ما الموديل اتدرّب."
        )
    X = X[schema["features"]].astype("float32")  # نفس الترتيب بالظبط
    return evaluate(model, X, df["label"])


def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="يقيّم موديل محفوظ على الـ golden test set (من غير تدريب)"
    )
    ap.add_argument("--model-dir", default="models/v1")
    ap.add_argument("--test-set", default="data/test_v1.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--write",
        action="store_true",
        help="يكتب النتيجة في <model-dir>/metrics.json",
    )
    args = ap.parse_args()

    metrics = evaluate_saved(args.model_dir, args.test_set, args.config)
    if args.write:
        path = os.path.join(args.model_dir, "metrics.json")
        # merge مش overwrite: evaluate() مابينتجش المفاتيح اللي train.py
        # بيضيفها بعده (زي train_seconds) — الكتابة فوق كانت بتضيّعها.
        merged = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                merged = json.load(fh)
            kept = [k for k in merged if k not in metrics]
            if kept:
                print(f"\nمحافظين على مفاتيح من الملف القديم: {kept}")
        merged.update(metrics)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False)
        print(f"✅ Written → {path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _main()
