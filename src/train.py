"""
train.py — the split + building the model + training, and wiring all pipeline steps together.

⬅️ Ported from: Cell 21 (stratified split) + Cell 39 (Random Forest).

What was **not** ported from the notebook (§8):
  item 1 — cuML (Cell 35/39) → sklearn only, the VM has no GPU.
  item 3 — Cell 23 (StandardScaler) removed entirely; tree models are not affected by scaling.
  item 4 — the median imputation that ran on all data before the split (Cell 17)
          is now a SimpleImputer **inside** the Pipeline → the test set leaks nothing into training.
  item 6 — no LabelEncoder over 15 classes; labels are 4 strings from config.label_map.
  item 9 — no SMOTE; class_weight="balanced_subsample" instead.

CLI:
    python -m src.train                  # all data      → models/v1/  + data/clean.parquet
    python -m src.train --nrows 20000    # random sample → models/smoke/ + data/smoke_*
    python -m src.train --no-cache       # ignore the parquet cache and reload

The smoke run is fully isolated: all its paths are different, and it never builds a golden test set.
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
# 1) Prepare the data: load → map_labels → clean  (+ cache)
# ---------------------------------------------------------------------
def smoke_cfg(cfg: dict, nrows: int) -> dict:
    """A copy of the config with every path isolated from the real run.

    Anything run with --nrows must never touch data/clean.parquet, data/test_v1.parquet
    or models/<version>/ — if a small sample overwrote those, full training would
    find a stale cache or a golden test set built from 160K rows instead of 2.8M.
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
    """Data after cleaning. Saved as parquet and reused if it exists (§7).

    random_sample: take nrows **random** rows from each file instead of the first nrows.
    Required for the smoke run — attacks in CIC-IDS2017 happen late in the day, so the first
    rows are almost all benign and the first 20K rows contain only 2 portscan flows.
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
# 2) The split  (Cell 21) + the golden test set
# ---------------------------------------------------------------------
def _row_keys(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """A fingerprint per row (features + label).

    Used instead of the index so we can find the golden test set rows even if the row
    order or data size changes between v1 and v2.
    """
    cols = [c for c in feats if c in df.columns] + ["label"]
    return pd.util.hash_pandas_object(df[cols], index=False)


def _check_classes(y: pd.Series, cfg: dict) -> None:
    """A clear message instead of the stratified train_test_split failing with a cryptic one."""
    counts = y.value_counts()
    missing = [c for c in cfg["classes"] if c not in counts.index]
    tiny = counts[counts < 2]
    if missing or len(tiny):
        print("\nClass counts:\n" + counts.to_string())
        raise SystemExit(
            f"\nCannot do a stratified split.\n"
            f"  missing classes : {missing or 'none'}\n"
            f"  classes with <2 rows: {tiny.to_dict() or 'none'}\n"
            f"Hint: if this is a --nrows run, use a bigger number — the random sample keeps\n"
            f"      the class ratios, and the smallest class (brute_force ≈ 0.5%) needs\n"
            f"      a sample big enough to get at least two rows."
        )


def split(df: pd.DataFrame, cfg: dict, golden_path: str | None = None):
    """Stratified split (Cell 21). If golden_path exists it is reused as the test set.

    Returns X_train, X_test, y_train, y_test.
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
            print("  ⚠️  Less than half of the golden set was found in the current data —")
            print("      so the data has changed. Comparison with older versions is not fair.")
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

    # float32: RandomForest converts its input to float32 internally anyway,
    # so converting early saves a ~1.4 GB float64 copy on the full data
    # without changing a single number in the model's output.
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
# 3) The model  (Cell 39 / 41)
# ---------------------------------------------------------------------
def build_model(cfg: dict) -> Pipeline:
    """Pipeline([impute, model]) — the imputer lives inside so it is applied the same way live."""
    mcfg = cfg["model"]
    params = dict(mcfg.get("params") or {})
    params.setdefault("random_state", cfg["random_state"])

    if mcfg["type"] == "random_forest":
        # §8 item 9 — replaces SMOTE. Not in config.yaml so it can't be changed by accident.
        params.setdefault("class_weight", "balanced_subsample")
        est = RandomForestClassifier(**params)
    elif mcfg["type"] == "xgboost":
        # Note: XGBClassifier needs numeric labels 0..n-1, not strings, so it needs
        # a LabelEncoder inside the Pipeline — not implemented in v1.
        raise SystemExit(
            "xgboost is not wired up in v1. To use it: uncomment xgboost in "
            "requirements.txt and add label encoding here."
        )
    else:
        raise ValueError(f"Unknown model type: {mcfg['type']!r}")

    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", est)])


# ---------------------------------------------------------------------
# 4) Full run
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
        print(f"=== SMOKE RUN — random sample of {nrows:,} rows/file ===")
        print(f"    cache  → {cfg['paths']['cache']}")
        out = os.path.join(cfg["paths"]["models_dir"], cfg["model_version"])
        print(f"    output → {out}/")
        print(
            "    does not touch data/clean.parquet, data/test_v1.parquet "
            "or the delivered model\n"
        )

    df = prepare(cfg, nrows=nrows, use_cache=use_cache, random_sample=smoke)

    # The golden test set is for the full run only — so future versions are compared fairly
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
    # Smoke always overwrites itself — it is development output; the protection is for v1
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
        help="random sample of N rows from each file — a quick isolated run in "
        "data/smoke_* and models/smoke/",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the parquet cache and reload + re-clean",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing models/<model_version>/ — for development only",
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
