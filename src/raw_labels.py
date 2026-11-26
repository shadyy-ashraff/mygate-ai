"""
raw_labels.py — recovers the original label (DoS Hulk, slowloris, ...) for every row
of the golden test set, so we can tell exactly which kind of attack was missed
and where the false positives come from.

⚠️ This file **does not touch** anything that already exists:
   - it does not read data/clean.parquet (the cache has no raw_label anyway)
   - it does not write or modify data/test_v1.parquet or data/clean.parquet
   - it does not change the model
   Its only output: data/test_v1_raw_labels.parquet

The idea: replay the exact same pipeline (same file read order, same
map_labels, same clean with the same dedup subset and keep="first", same split
and seed) with one extra column, raw_label. If the steps match, the resulting test set
must be **the same rows in the same order**. We assert that on a fingerprint
of every row — not just the label, because there are hundreds of thousands of benign
rows in a row, so any shift could slip through.

    python -m src.raw_labels
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from mygate_ai.features import build_features, feature_list
from src.clean import clean, map_labels
from src.loaders import _find_csv_dir, load_all

GOLDEN = "data/test_v1.parquet"
OUT = "data/test_v1_raw_labels.parquet"
MODEL_DIR = "mygate_ai/models/v1"
BENIGN = "benign"


def _short(name: str) -> str:
    """Short file name so the tables fit."""
    return name.replace(".pcap_ISCX.csv", "").replace("-WorkingHours", "")


def _source_file_column(cfg: dict, n_rows: int) -> np.ndarray:
    """A column with the name of the CSV each row came from.

    We don't modify loaders.py — we read **only the Label column** of each file in the
    same sorted(os.listdir) order load_cic2017 uses, so we know each file's row count,
    and build the column with np.repeat. The assert below guarantees alignment.
    """
    src = cfg["sources"]["cic2017"]
    base = _find_csv_dir(src["path"])
    files = sorted(f for f in os.listdir(base) if f.endswith(".csv"))

    names, counts = [], []
    for f in files:
        n = len(
            pd.read_csv(os.path.join(base, f), usecols=[" Label"], low_memory=False)
        )
        names.append(_short(f))
        counts.append(n)
        print(f"  {_short(f):42s} {n:>9,}")

    total = sum(counts)
    if total != n_rows:
        raise SystemExit(
            f"\n❌ Sum of file rows ({total:,}) != DataFrame rows "
            f"({n_rows:,}).\n   So the read order/method changed. Stopping."
        )
    return np.repeat(names, counts)


# ---------------------------------------------------------------------
# 1) Rebuild the data with raw_label
# ---------------------------------------------------------------------
def rebuild_with_raw(cfg: dict) -> pd.DataFrame:
    """The exact training pipeline + a raw_label column."""
    feats = feature_list(cfg)

    print("=" * 72)
    print("1) Reading the CSVs from scratch (no cache)")
    print("=" * 72)
    df = load_all(cfg)  # same read order: sorted(os.listdir)

    # The original label, before map_labels overwrites it.
    # .str.strip() to match exactly what map_labels does.
    df["raw_label"] = df["label"].astype(str).str.strip().astype("category")

    print("\nRows per file (for the source_file column):")
    df["source_file"] = pd.Categorical(_source_file_column(cfg, len(df)))

    print("\n" + "=" * 72)
    print("2) map_labels")
    print("=" * 72)
    df = map_labels(df, cfg["label_map"])

    # Check before dedup: rows identical in features + label but with
    # different raw_labels (like DDoS and DoS Hulk — both map to dos)
    _report_raw_label_collisions(df, feats)

    print("\n" + "=" * 72)
    print("3) clean — same dedup subset (features + label, without raw_label)")
    print("=" * 72)
    # raw_label is just an extra column, not in the subset — so dedup behaves
    # exactly like training, and keep="first" (the default) keeps the first row in
    # read order.
    df = clean(df, feats, show_preview=False)

    # Rows with exactly the same features but different final labels
    # (the 1,438 that clean() prints). Computed on **all** the data, not on
    # the test set alone, because the twin may be in the train set.
    conflict = df.duplicated(subset=feats, keep=False)
    df["feature_conflict"] = conflict.to_numpy()
    print(f"\nRows with identical features and different labels: {int(conflict.sum()):,}")
    return df


def _report_raw_label_collisions(df: pd.DataFrame, feats: list[str]) -> None:
    """Rows that dedup will drop even though their raw_label differs."""
    print("\n--- Identical rows with different raw_labels ---")
    cols = [c for c in feats if c in df.columns] + ["label"]
    key = pd.util.hash_pandas_object(df[cols], index=False)

    tmp = pd.DataFrame({"key": key.to_numpy(), "raw": df["raw_label"].to_numpy()})
    nun = tmp.groupby("key", observed=True)["raw"].nunique()
    conflict_keys = nun[nun > 1]

    if conflict_keys.empty:
        print("None — every group of identical rows has a single raw_label.")
        return

    mask = tmp["key"].isin(set(conflict_keys.index))
    print(f"Affected groups : {len(conflict_keys):,}")
    print(f"Affected rows   : {int(mask.sum()):,}")
    print("\nTop 10 raw_label combinations that get mixed up:")
    combos = (
        tmp[mask]
        .groupby("key", observed=True)["raw"]
        .agg(lambda s: " + ".join(sorted(set(map(str, s)))))
        .value_counts()
        .head(10)
    )
    print(combos.to_string())
    print(
        "\nNote: keep='first' keeps the first row in read order, so the raw_label\n"
        "      that survives is the one from the alphabetically first file. This only\n"
        "      affects the breakdown below, not the model — the model saw the unified\n"
        "      label (dos) in both cases."
    )


# ---------------------------------------------------------------------
# 2) Same split + verify the match
# ---------------------------------------------------------------------
def same_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Same train_test_split with the same seed → returns the test set."""
    print("\n" + "=" * 72)
    print("4) Same split (same test_size and random_state)")
    print("=" * 72)
    _, df_test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df["label"],
    )
    print(f"test: {len(df_test):,} rows")
    return df_test


def verify_alignment(df_test: pd.DataFrame, golden: pd.DataFrame, feats: list[str]):
    """Assert the rows match test_v1.parquet row by row.

    We compare a fingerprint of each row (all features + label), not the label alone —
    the label is a weak key because there are 622K benign rows.
    """
    print("\n" + "=" * 72)
    print("5) Verifying the match with " + GOLDEN)
    print("=" * 72)

    if len(df_test) != len(golden):
        raise SystemExit(
            f"\n❌ Count differs: the rebuild produced {len(df_test):,} rows, "
            f"and {GOLDEN} has {len(golden):,}.\n"
            f"   So a pipeline step changed. Stopping."
        )
    print(f"Count         : {len(df_test):,} ✅")

    lbl_new = df_test["label"].astype(str).to_numpy()
    lbl_gold = golden["label"].astype(str).to_numpy()
    n_bad_label = int((lbl_new != lbl_gold).sum())

    cols = [c for c in feats if c in golden.columns] + ["label"]
    k_new = pd.util.hash_pandas_object(df_test[cols], index=False).to_numpy()
    k_gold = pd.util.hash_pandas_object(golden[cols], index=False).to_numpy()
    n_bad_row = int((k_new != k_gold).sum())

    print(f"Labels differ : {n_bad_label:,}")
    print(f"Rows differ   : {n_bad_row:,}   (fingerprint of all features + label)")

    if n_bad_label or n_bad_row:
        first = int(np.argmax((k_new != k_gold) | (lbl_new != lbl_gold)))
        raise SystemExit(
            f"\n❌ Rows don't match. First difference at row {first:,}:\n"
            f"   rebuilt : label={lbl_new[first]}\n"
            f"   {GOLDEN}     : label={lbl_gold[first]}\n"
            f"\nWe won't try to patch this — it means a pipeline step changed\n"
            f"(read order, dedup, map or seed). We need to understand\n"
            f"the cause first."
        )

    print("\n✅ Matches row by row — raw_labels are aligned with the golden set.")

    # Third check: map_labels(raw_label) must give the same label
    lm = dict(_cfg["label_map"])
    mapped = df_test["raw_label"].astype(str).map(lm).to_numpy()
    n_bad_map = int((mapped != lbl_gold).sum())
    print(f"map_labels(raw_label) == label : {n_bad_map:,} mismatches")
    if n_bad_map:
        raise SystemExit("❌ raw_label does not map to the stored label.")
    return df_test


# ---------------------------------------------------------------------
# 3) The breakdown
# ---------------------------------------------------------------------
def breakdown(golden: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> None:
    print("\n" + "=" * 72)
    print("6) BREAKDOWN — missed attacks and false positives")
    print("=" * 72)

    with open(os.path.join(MODEL_DIR, "schema.json"), encoding="utf-8") as fh:
        schema = json.load(fh)
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

    X = build_features(golden, cfg)[schema["features"]].astype("float32")
    y = golden["label"].astype(str).to_numpy()
    classes = [str(c) for c in model.classes_]
    bi = classes.index(BENIGN)

    proba = model.predict_proba(X)
    pred = np.array([classes[i] for i in proba.argmax(axis=1)])
    p_attack = 1.0 - proba[:, bi]
    raw = meta["raw_label"].astype(str).to_numpy()
    src = meta["source_file"].astype(str).to_numpy()

    # ---- A) Missed attacks ----
    missed = (y != BENIGN) & (pred == BENIGN)
    print(f"\n--- A) Missed attacks: {int(missed.sum())} ---")
    print("\nBy raw_label:")
    print(pd.Series(raw[missed]).value_counts().to_string())

    print("\nRecall per raw_label (all types):")
    tab = (
        pd.DataFrame({"raw": raw, "caught": pred != BENIGN, "is_attack": y != BENIGN})[
            lambda d: d["is_attack"]
        ]
        .groupby("raw", observed=True)["caught"]
        .agg(support="count", caught="sum")
    )
    tab["missed"] = tab["support"] - tab["caught"]
    tab["recall"] = (tab["caught"] / tab["support"]).round(5)
    print(tab.sort_values("recall").to_string())

    # ---- B) False positives ----
    fp = (y == BENIGN) & (pred != BENIGN)
    print(f"\n--- B) False positives: {int(fp.sum())} ---")
    print("\nClassified as:")
    print(pd.Series(pred[fp]).value_counts().to_string())

    if "dst_port" in golden.columns:
        ports = golden.loc[fp, "dst_port"]
        print("\nTop 10 dst_port:")
        top = ports.value_counts().head(10)
        total_benign_per_port = golden.loc[y == BENIGN, "dst_port"].value_counts()
        out = pd.DataFrame(
            {
                "false_positives": top,
                "benign_flows_on_port": total_benign_per_port.reindex(top.index),
            }
        )
        out["fp_rate"] = (out["false_positives"] / out["benign_flows_on_port"]).round(6)
        print(out.to_string())

    # ---- B2/B3/B4) FPs by file ----
    print("\n--- B2) FPs by file × predicted class ---")
    print(
        pd.crosstab(
            pd.Series(src[fp], name="source_file"),
            pd.Series(pred[fp], name="predicted"),
            margins=True,
        ).to_string()
    )

    sure = fp & (p_attack >= 0.99)
    print(
        f"\n--- B3) Same table for FPs with probability >= 0.99 only "
        f"({int(sure.sum())}) ---"
    )
    print(
        pd.crosstab(
            pd.Series(src[sure], name="source_file"),
            pd.Series(pred[sure], name="predicted"),
            margins=True,
        ).to_string()
    )

    if "dst_port" in golden.columns:
        p444 = fp & (golden["dst_port"].to_numpy() == 444)
        print(f"\n--- B4) FPs on port 444 ({int(p444.sum())}) ---")
        if p444.any():
            print(
                pd.DataFrame({"source_file": src[p444], "predicted": pred[p444]})
                .value_counts()
                .to_string()
            )

    # ---- A2) Missed attacks by file ----
    print(f"\n--- A2) The {int(missed.sum())} missed attacks by file ---")
    print(
        pd.crosstab(
            pd.Series(src[missed], name="source_file"),
            pd.Series(raw[missed], name="raw_label"),
            margins=True,
        ).to_string()
    )

    # ---- D) Conflicting rows ----
    if "feature_conflict" in meta.columns:
        conf = meta["feature_conflict"].to_numpy().astype(bool)
        print("\n--- D) Rows with identical features and different labels ---")
        print(f"In the golden set             : {int(conf.sum()):,}")
        print(
            f"Of which among the {int(missed.sum())} missed : {int((conf & missed).sum()):,}"
        )
        print(f"Of which among the {int(fp.sum())} FPs      : {int((conf & fp).sum()):,}")
        if conf.any():
            print("\nBy raw_label:")
            print(pd.Series(raw[conf]).value_counts().head(10).to_string())

    # ---- C) The 12 suspicious ----
    susp = (pred == BENIGN) & (p_attack >= 0.5)
    print(
        f"\n--- C) Suspicious (argmax=benign but probability >= 0.5): "
        f"{int(susp.sum())} ---"
    )
    if susp.any():
        print(
            pd.DataFrame(
                {
                    "raw_label": raw[susp],
                    "truth": y[susp],
                    "probability": p_attack[susp].round(4),
                }
            )
            .sort_values("probability", ascending=False)
            .to_string(index=False)
        )


# ---------------------------------------------------------------------
def main() -> None:
    global _cfg
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--skip-rebuild",
        action="store_true",
        help=f"use the existing {OUT} instead of re-reading the CSVs (breakdown only)",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        _cfg = yaml.safe_load(fh)
    feats = feature_list(_cfg)

    if not os.path.isfile(GOLDEN):
        raise SystemExit(f"No {GOLDEN} — run python -m src.train first.")
    golden = pd.read_parquet(GOLDEN)

    if args.skip_rebuild and os.path.isfile(OUT):
        meta = pd.read_parquet(OUT)
        print(f"Using the existing {OUT} ({len(meta):,} rows)")
        missing = [c for c in ("source_file", "feature_conflict") if c not in meta]
        if missing:
            raise SystemExit(
                f"{OUT} is outdated — missing {missing}. Run without --skip-rebuild."
            )
        if len(meta) != len(golden):
            raise SystemExit(
                f"{OUT} has {len(meta):,} rows and {GOLDEN} has {len(golden):,}."
            )
    else:
        df = rebuild_with_raw(_cfg)
        df_test = same_split(df, _cfg)
        df_test = verify_alignment(df_test, golden, feats)

        meta = pd.DataFrame(
            {
                "row": np.arange(len(df_test)),  # matches test_v1.parquet order
                "raw_label": df_test["raw_label"].astype(str).to_numpy(),
                "source_file": df_test["source_file"].astype(str).to_numpy(),
                "feature_conflict": df_test["feature_conflict"].to_numpy(),
            }
        )
        os.makedirs("data", exist_ok=True)
        meta.to_parquet(OUT, index=False)
        print(f"\n✅ Saved → {OUT}  ({len(meta):,} rows)")
        print("\nGolden set rows by file:")
        print(meta["source_file"].value_counts().to_string())

    breakdown(golden, meta, _cfg)


_cfg: dict = {}

if __name__ == "__main__":
    main()
