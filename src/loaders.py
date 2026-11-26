"""
loaders.py — تحميل الداتا من كل مصدر → DataFrame بالأسماء الموحّدة.

متنقول من Cell 5 (kagglehub + قراءة الـ CSV) + خلية تقليل الرام.
كل مصدر ليه function، وكلهم بيرجّعوا نفس الشكل:
    أعمدة canonical + عمود label + عمود source
"""

from __future__ import annotations

import os

import pandas as pd

from mygate_ai.schema import to_canonical


def _find_csv_dir(path: str) -> str:
    """يدوّر على الفولدر اللي فيه الـ CSV (لو kagglehub نزّلها جوا فولدر فرعي)."""
    for root, _dirs, files in os.walk(path):
        if any(f.endswith(".csv") for f in files):
            return root
    raise FileNotFoundError(f"No CSV files under {path}")


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """تقليل الرام للنص تقريباً (float64 → float32، int64 → أصغر نوع مناسب)."""
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
    CIC-IDS2017: لو src_cfg فيه "path" بيقرا من الجهاز، غير كده بينزّل بـ kagglehub.

    nrows : أول N صف من كل ملف — سريع، **بس متحيّز**. ملفات CIC-IDS2017
            مترتّبة زمنياً والهجمات بتحصل آخر اليوم، فأول الصفوف كلها benign
            تقريباً (PortScan أول ظهور كثيف ليها بعد الصف ~50,000).
    sample: N صف **عشوائي** من كل ملف. بيقرا الملف كله الأول فبياخد وقت أطول،
            لكن العيّنة بتمثّل الـ classes الأربعة زي الداتا الأصلية.
            ده اللي الـ smoke run بيستخدمه.
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
            print(f"Sampled {f}: {part.shape}  (من {n_read:,} صف)")
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
    """يحمّل كل المصادر اللي enabled في config.yaml ويدمجهم.

    nrows / sample: شوف load_cic2017 — الأولى أول N صف، التانية N عشوائي.
    """
    parts = [
        load_source(name, s, nrows=nrows, sample=sample, random_state=random_state)
        for name, s in cfg["sources"].items()
        if s.get("enabled")
    ]
    return pd.concat(parts, ignore_index=True)
