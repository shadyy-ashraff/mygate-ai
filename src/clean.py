"""
clean.py — data cleaning + mapping labels to the 4 classes.

Ported from notebook Cells 17 + 19, with 4 changes:
  1) NaN filling is not done here — it moved into the Pipeline (SimpleImputer) in train.py,
     so it happens exactly the same way live, and the test set cannot leak into training.
  2) Constant (zero-variance) columns are not dropped automatically — they are listed
     explicitly in config.yaml → drop_features, so the feature list never changes
     on its own when we add a new dataset.
  3) dtype downcasting (uint8/int16) was removed — loaders.py already does it (float32).
  4) Labels: instead of a LabelEncoder over 15 classes → label_map from config.yaml,
     and any out-of-scope label (Web Attacks, Bot, ...) is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """Original labels → benign/portscan/brute_force/dos; everything else is dropped."""
    df = df.copy()
    raw = df["label"].astype(str).str.strip()
    df["label"] = raw.map(label_map)

    out_of_scope = df["label"].isna()
    if out_of_scope.any():
        print("Dropped (out of scope):")
        print(raw[out_of_scope].value_counts().to_string())
    df = df[~out_of_scope]

    print("\nClasses after mapping:")
    print(df["label"].value_counts().to_string())
    return df


def preview(df: pd.DataFrame, feature_cols: list[str], n: int = 10) -> None:
    """Table of the final data shape — the first n rows as the model will see them.

    Shown **transposed**: each line = a feature, each column = a data row.
    Because 69 columns side by side don't fit in a terminal.
    """
    feats = [c for c in feature_cols if c in df.columns]
    cols = feats + ["label"]

    sample = df[cols].head(n).copy()
    num = sample[feats].select_dtypes("number").columns
    sample[num] = sample[num].round(2)  # keep the numbers readable

    table = sample.T
    table.columns = [f"row{i}" for i in range(len(sample))]
    # NaN count per feature over the whole data — not just these 10 rows
    table.insert(0, "n_nan", [int(df[c].isna().sum()) for c in cols])

    print(f"\n{'=' * 72}")
    print(
        f"FINAL DATA — first {n} rows   ({len(df):,} rows × {len(feats)} features + label)"
    )
    print(f"{'=' * 72}")
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(table.to_string())

    print(
        f"\nlabel: plain strings, no LabelEncoder (§8 item 6) → {sorted(df['label'].unique())}"
    )
    print(
        "n_nan > 0: these NaNs are filled with the median inside the Pipeline at fit time (§8 item 4),"
    )
    print(
        "           not here — so the median comes from the train split only, with no data leakage."
    )


def clean(
    df: pd.DataFrame, feature_cols: list[str], show_preview: bool = True
) -> pd.DataFrame:
    """Inf → NaN + drop duplicate rows. (NaNs themselves are filled in the Pipeline.)"""
    df = df.copy()
    feats = [c for c in feature_cols if c in df.columns]

    # 1) Inf → NaN — happens when flow_duration = 0 (division by zero in Bytes/s and Packets/s)
    n_inf = int(np.isinf(df[feats].to_numpy(dtype="float64")).sum())
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    print(f"Inf → NaN: {n_inf:,} values")
    print(
        f"NaN (will be imputed in the Pipeline): {int(df[feats].isna().sum().sum()):,}"
    )

    # 2) Duplicate rows — before the split, so the same row can't land in both train and test
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["label"])
    print(f"Duplicates removed: {before - len(df):,}  →  {len(df):,} rows")

    # 3) For the report: rows with identical features but different labels
    conflicts = int(df.duplicated(subset=feats, keep=False).sum())
    print(f"Rows with identical features but different labels: {conflicts:,}")

    # 4) Final data table
    if show_preview:
        preview(df, feats)

    return df


def _main() -> None:
    """Run this file on its own (▶ button or `python -m src.clean`):
    loads a sample → map_labels → clean → prints the final data table.
    For experimenting only — real training is in src/train.py.
    """
    import argparse

    import yaml

    from mygate_ai.features import feature_list
    from src.loaders import load_all

    ap = argparse.ArgumentParser(description="Clean a sample and show the final data table")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nrows", type=int, default=20000, help="first N rows of each file")
    ap.add_argument("--rows", type=int, default=10, help="how many rows to show in the table")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    df = load_all(cfg, nrows=args.nrows)
    df = map_labels(df, cfg["label_map"])
    df = clean(df, feature_list(cfg), show_preview=False)
    preview(df, feature_list(cfg), n=args.rows)


if __name__ == "__main__":
    # When VS Code runs the file directly (▶ button) it puts src/ on the path,
    # not the project root, so `from src...` fails. This line adds the root.
    # Not needed with `python -m src.clean`, but harmless.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    _main()
