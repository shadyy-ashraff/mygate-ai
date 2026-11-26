"""
loaders.py — load data from each source → DataFrame with canonical names.

Ported from notebook Cell 5 (kagglehub + reading the CSVs) + the memory-reduction cell.
Each source has its own function, and all of them return the same shape:
    canonical columns + a label column + a source column
"""

from __future__ import annotations

import os

import pandas as pd

from mygate_ai.schema import to_canonical


def _find_csv_dir(path: str) -> str:
    """Find the folder that holds the CSVs (in case kagglehub put them in a subfolder)."""
    for root, _dirs, files in os.walk(path):
        if any(f.endswith(".csv") for f in files):
            return root
    raise FileNotFoundError(f"No CSV files under {path}")


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Cut memory roughly in half (float64 → float32, int64 → smallest fitting type)."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_cic2017(
    src_cfg: dict,
    nrows: int | None = None,
    sample: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    CIC-IDS2017: if src_cfg has "path", read from disk; otherwise download with kagglehub.

    nrows : first N rows of each file — fast, **but biased**. CIC-IDS2017 files
            are in time order and the attacks happen late in the day, so the first rows
            are almost all benign (PortScan first appears densely after row ~50,000).
    sample: N **random** rows from each file. Reads the whole file first, so it is slower,
            but the sample represents the four classes like the original data.
            This is what the smoke run uses.
    """
    if src_cfg.get("path") and os.path.isdir(src_cfg["path"]):
        base = src_cfg["path"]
    else:
        import kagglehub

        base = kagglehub.dataset_download(src_cfg["kaggle_id"])
    base = _find_csv_dir(base)

    files = sorted(f for f in os.listdir(base) if f.endswith(".csv"))
    dfs = []
    for f in files:
        if sample is not None:
            part = pd.read_csv(os.path.join(base, f), low_memory=False)
            n_read = len(part)
            part = part.sample(n=min(sample, n_read), random_state=random_state)
            print(f"Sampled {f}: {part.shape}  (from {n_read:,} rows)")
        else:
            part = pd.read_csv(os.path.join(base, f), nrows=nrows, low_memory=False)
            print(f"Loaded {f}: {part.shape}")
        dfs.append(_shrink(part))
    df = pd.concat(dfs, ignore_index=True)
    print(
        f"Total: {df.shape[0]:,} rows | "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.0f} MB"
    )
    return df


LOADERS = {
    "cic2017": load_cic2017,
    # "cic2018": load_cic2018,   # v2
    # "lab": load_lab,           # v2
}


def load_source(
    name: str,
    src_cfg: dict,
    nrows: int | None = None,
    sample: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = LOADERS[name](src_cfg, nrows=nrows, sample=sample, random_state=random_state)
    df = to_canonical(df, name)
    df["source"] = name
    return df


def load_all(
    cfg: dict,
    nrows: int | None = None,
    sample: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load every source enabled in config.yaml and concatenate them.

    nrows / sample: see load_cic2017 — the first takes the first N rows, the second N random rows.
    """
    parts = [
        load_source(name, s, nrows=nrows, sample=sample, random_state=random_state)
        for name, s in cfg["sources"].items()
        if s.get("enabled")
    ]
    return pd.concat(parts, ignore_index=True)
