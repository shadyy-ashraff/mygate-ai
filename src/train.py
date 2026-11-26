"""
train.py — الـ split + بناء الموديل + التدريب، والربط بين كل خطوات الـ pipeline.

⬅️ متنقول من: Cell 21 (stratified split) + Cell 39 (Random Forest).

اللي **ما**اتنقلش من النوت‑بوك (§8):
  بند ١ — cuML (Cell 35/39) → sklearn بس، الـ VM مفيهاش GPU.
  بند ٣ — Cell 23 (StandardScaler) اتشال خالص، الـ tree models مش بتتأثر بالـ scaling.
  بند ٤ — الـ median imputation اللي كان على الداتا كلها قبل الـ split (Cell 17)
          بقى SimpleImputer **جوا** الـ Pipeline → الـ test مش بيسرّب معلومات للـ train.
  بند ٦ — مفيش LabelEncoder بـ 15 class، الـ labels ٤ strings من config.label_map.
  بند ٩ — مفيش SMOTE، بدله class_weight="balanced_subsample".

CLI:
    python -m src.train                  # الداتا كلها  → models/v1/  + data/clean.parquet
    python -m src.train --nrows 20000    # عيّنة عشوائية → models/smoke/ + data/smoke_*
    python -m src.train --no-cache       # يتجاهل الـ parquet cache ويعيد التحميل

الـ smoke run معزول تماماً: مساراته كلها مختلفة، ومابيعملش golden test set.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mygate_ai.features import build_features, feature_list
from src.clean import clean, map_labels
from src.evaluate import evaluate
from src.export import export
from src.loaders import load_all


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------
# 1) تجهيز الداتا: load → map_labels → clean  (+ cache)
# ---------------------------------------------------------------------
def smoke_cfg(cfg: dict, nrows: int) -> dict:
    """نسخة من الـ config كل مساراتها معزولة عن التشغيلة الحقيقية.

    أي حاجة بـ --nrows لازم ماتقربش من data/clean.parquet ولا data/test_v1.parquet
    ولا models/<version>/ — لأن عيّنة صغيرة لو دَعَست على دول، التدريب الكامل
    هيلاقي cache قديم أو golden test set مبني على 160 ألف صف بدل 2.8 مليون.
    """
    return {
        **cfg,
        "model_version": "smoke",  # → models/smoke/
        "paths": {
            **cfg["paths"],
            "cache": f"data/smoke_clean_{nrows}.parquet",
        },
    }


def prepare(
    cfg: dict,
    nrows: int | None = None,
    use_cache: bool = True,
    random_sample: bool = False,
) -> pd.DataFrame:
    """الداتا بعد التنظيف. بيتحفظ في parquet وبيتعاد استخدامه لو موجود (§7).

    random_sample: ياخد nrows صف **عشوائي** من كل ملف بدل أول nrows صف.
    لازم للـ smoke run — الهجمات في CIC-IDS2017 بتحصل آخر اليوم، فأول
    الصفوف كلها benign تقريباً وأول 20 ألف صف بيدّوا 2 portscan بس.
    """
    path = cfg["paths"]["cache"]
    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"Cache hit: {path}  →  {len(df):,} rows")
        return df

    if random_sample:
        df = load_all(cfg, sample=nrows, random_state=cfg["random_state"])
    else:
        df = load_all(cfg, nrows=nrows)
    df = map_labels(df, cfg["label_map"])
    df = clean(df, feature_list(cfg))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Cached → {path}")
    return df


# ---------------------------------------------------------------------
# 2) الـ split  (Cell 21) + الـ golden test set
# ---------------------------------------------------------------------
def _row_keys(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """بصمة لكل صف (features + label).

    بنستخدمها بدل الـ index عشان نعرف صفوف الـ golden test set حتى لو ترتيب
    الصفوف أو حجم الداتا اتغيّر بين v1 و v2.
    """
    cols = [c for c in feats if c in df.columns] + ["label"]
    return pd.util.hash_pandas_object(df[cols], index=False)


def _check_classes(y: pd.Series, cfg: dict) -> None:
    """رسالة مفهومة بدل ما train_test_split الـ stratified يقع برسالة غامضة."""
    counts = y.value_counts()
    missing = [c for c in cfg["classes"] if c not in counts.index]
    tiny = counts[counts < 2]
    if missing or len(tiny):
        print("\nClass counts:\n" + counts.to_string())
        raise SystemExit(
            f"\nCannot do a stratified split.\n"
            f"  missing classes : {missing or 'none'}\n"
            f"  classes with <2 rows: {tiny.to_dict() or 'none'}\n"
            f"Hint: لو دي تشغيلة --nrows، كبّر الرقم — العيّنة العشوائية بتحافظ\n"
            f"      على نسب الـ classes، وأقل class (brute_force ≈ 0.5%) محتاجة\n"
            f"      عيّنة كبيرة كفاية عشان توصل صفّين على الأقل."
        )


def split(df: pd.DataFrame, cfg: dict, golden_path: str | None = None):
    """Stratified split (Cell 21). لو golden_path موجود بيتعاد استخدامه كـ test set.

    بيرجّع X_train, X_test, y_train, y_test.
    """
    feats = feature_list(cfg)
    _check_classes(df["label"], cfg)

    if golden_path and os.path.exists(golden_path):
        golden = pd.read_parquet(golden_path)
        in_test = _row_keys(df, feats).isin(set(_row_keys(golden, feats)))
        matched = int(in_test.sum())
        print(f"\nGolden test set reused: {golden_path}  ({len(golden):,} rows)")
        print(f"  matched in current data: {matched:,}")
        if matched < 0.5 * len(golden):
            print("  ⚠️  أقل من نص الـ golden set اتلاقى في الداتا الحالية —")
            print("      يعني الداتا اتغيّرت. المقارنة مع النسخ القديمة مش عادلة.")
        df_train, df_test = df[~in_test], golden
    else:
        df_train, df_test = train_test_split(
            df,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=df["label"],
        )
        if golden_path:
            os.makedirs(os.path.dirname(golden_path) or ".", exist_ok=True)
            keep = [c for c in feats if c in df_test.columns] + ["label"]
            df_test[keep].to_parquet(golden_path, index=False)
            print(f"\nGolden test set saved → {golden_path}  ({len(df_test):,} rows)")

    # float32: الـ RandomForest بيحوّل الـ input لـ float32 جوّاه على أي حال،
    # فالتحويل بدري بيوفّر نسخة float64 بحجم ~1.4 GB على الداتا الكاملة
    # من غير ما يغيّر ولا رقم في نتيجة الموديل.
    X_train = build_features(df_train, cfg).astype("float32")
    X_test = build_features(df_test, cfg).astype("float32")
    y_train, y_test = df_train["label"], df_test["label"]

    print(
        f"\nSplit: train {len(X_train):,}  |  test {len(X_test):,}  |  {len(feats)} features"
    )
    dist = (
        pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()})
        .fillna(0)
        .astype(int)
    )
    dist["train %"] = (dist["train"] / len(y_train) * 100).round(2)
    print(dist.to_string())

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# 3) الموديل  (Cell 39 / 41)
# ---------------------------------------------------------------------
def build_model(cfg: dict) -> Pipeline:
    """Pipeline([impute, model]) — الـ imputer جوّه عشان يتطبّق بنفس الطريقة لايف."""
    mcfg = cfg["model"]
    params = dict(mcfg.get("params") or {})
    params.setdefault("random_state", cfg["random_state"])

    if mcfg["type"] == "random_forest":
        # §8 بند ٩ — بدل SMOTE. مش في config.yaml عشان ميتغيّرش بالغلط.
        params.setdefault("class_weight", "balanced_subsample")
        est = RandomForestClassifier(**params)
    elif mcfg["type"] == "xgboost":
        # ملاحظة: XGBClassifier عايز labels أرقام 0..n-1 مش strings، فمحتاج
        # LabelEncoder جوا الـ Pipeline — مش متعمول في v1.
        raise SystemExit(
            "xgboost مش موصّل في v1. لو عايزه: شيل التعليق من xgboost في "
            "requirements.txt وضيف encoding للّـ labels هنا."
        )
    else:
        raise ValueError(f"Unknown model type: {mcfg['type']!r}")

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


# ---------------------------------------------------------------------
# 4) التشغيل الكامل
# ---------------------------------------------------------------------
def train(
    cfg: dict,
    nrows: int | None = None,
    use_cache: bool = True,
    force: bool = False,
):
    """load_all → map_labels → clean → build_features → split → fit → evaluate → export."""
    smoke = nrows is not None
    if smoke:
        cfg = smoke_cfg(cfg, nrows)
        print(f"=== SMOKE RUN — عيّنة عشوائية {nrows:,} صف/ملف ===")
        print(f"    cache  → {cfg['paths']['cache']}")
        out = os.path.join(cfg["paths"]["models_dir"], cfg["model_version"])
        print(f"    output → {out}/")
        print(
            "    مش بيلمس data/clean.parquet ولا data/test_v1.parquet "
            "ولا النسخة المتسلّمة\n"
        )

    df = prepare(cfg, nrows=nrows, use_cache=use_cache, random_sample=smoke)

    # الـ golden test set للتشغيلة الكاملة بس — عشان النسخ الجاية تتقارن عليه بعدل
    golden = (
        None
        if smoke
        else os.path.join(
            os.path.dirname(cfg["paths"]["cache"]) or ".", "test_v1.parquet"
        )
    )
    X_train, X_test, y_train, y_test = split(df, cfg, golden_path=golden)

    model = build_model(cfg)
    print(f"\nFitting {cfg['model']['type']} on {len(X_train):,} rows ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    print(f"Done in {fit_seconds:.1f}s  |  classes: {list(model.classes_)}")

    metrics = evaluate(model, X_test, y_test)
    metrics["train_seconds"] = round(fit_seconds, 1)

    info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sources": [n for n, s in cfg["sources"].items() if s.get("enabled")],
        "nrows": nrows,
    }
    # الـ smoke بيكتب فوق نفسه دايماً — هو مخرجات تطوير، الحماية على v1
    out_dir = export(model, cfg, metrics, info, force=force or smoke)
    print(f"\n✅ Exported → {out_dir}")
    return model, metrics, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="MyGate AI — train the flow classifier")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="عيّنة عشوائية N صف من كل ملف — تجربة سريعة معزولة في "
        "data/smoke_* و models/smoke/",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="يتجاهل الـ parquet cache ويعيد التحميل والتنظيف",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="يكتب فوق models/<model_version>/ الموجود — للتطوير بس",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    train(
        cfg,
        nrows=args.nrows,
        use_cache=not args.no_cache,
        force=args.force,
    )


if __name__ == "__main__":
    main()
