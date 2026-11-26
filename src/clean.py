"""
clean.py — تنظيف الداتا + تحويل الـ labels للـ 4 classes.

متنقول من Cell 17 + Cell 19 في النوت‑بوك، مع ٤ تعديلات:
  1) ملء الـ NaN مش هنا — بقى جوا الـ Pipeline (SimpleImputer) في train.py،
     عشان يتعمل بنفس الطريقة بالظبط لايف، ومن غير ما الـ test يأثر على الـ train.
  2) الأعمدة الثابتة (zero-variance) مش بتتشال أوتوماتيك — بقت مكتوبة
     صراحةً في config.yaml → drop_features، عشان قايمة الـ features متتغيّرش
     لوحدها لما نضيف داتاست جديدة.
  3) تصغير الأنواع (uint8/int16) اتشال — اتعمل خلاص في loaders.py (float32).
  4) الـ labels: بدل LabelEncoder على 15 class → label_map من config.yaml،
     وأي label برا النطاق (Web Attacks, Bot, ...) بيتشال.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """الـ labels الأصلية → benign/portscan/brute_force/dos، والباقي يتشال."""
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
    """جدول الشكل النهائي للداتا — أول n صفوف زي ما الموديل هيشوفها.

    معروض **مقلوب**: كل سطر = feature، كل عمود = صف من الداتا.
    السبب إن 69 عمود جنب بعض مش بيبانوا في الـ terminal.
    """
    feats = [c for c in feature_cols if c in df.columns]
    cols = feats + ["label"]

    sample = df[cols].head(n).copy()
    num = sample[feats].select_dtypes("number").columns
    sample[num] = sample[num].round(2)  # عشان الأرقام تبان مقروءة

    table = sample.T
    table.columns = [f"row{i}" for i in range(len(sample))]
    # عدد الـ NaN في الداتا كلها لكل feature — مش في الـ 10 صفوف بس
    table.insert(0, "n_nan", [int(df[c].isna().sum()) for c in cols])

    print(f"\n{'=' * 72}")
    print(
        f"FINAL DATA — أول {n} صفوف   ({len(df):,} صف × {len(feats)} features + label)"
    )
    print(f"{'=' * 72}")
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(table.to_string())

    print(
        f"\nlabel: strings زي ما هي، مفيش LabelEncoder (§8 بند ٦) → {sorted(df['label'].unique())}"
    )
    print(
        "n_nan > 0: الـ NaN دي بتتملي بالـ median جوّه الـ Pipeline وقت الـ fit (§8 بند ٤)،"
    )
    print(
        "           مش هنا — عشان الـ median يتحسب من الـ train بس ومايحصلش data leakage."
    )


def clean(
    df: pd.DataFrame, feature_cols: list[str], show_preview: bool = True
) -> pd.DataFrame:
    """Inf → NaN + شيل الصفوف المكررة. (الـ NaN نفسها بتتملي في الـ Pipeline.)"""
    df = df.copy()
    feats = [c for c in feature_cols if c in df.columns]

    # 1) Inf → NaN — بيحصل لما flow_duration = 0 (قسمة على صفر في Bytes/s و Packets/s)
    n_inf = int(np.isinf(df[feats].to_numpy(dtype="float64")).sum())
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    print(f"Inf → NaN: {n_inf:,} values")
    print(
        f"NaN (will be imputed in the Pipeline): {int(df[feats].isna().sum().sum()):,}"
    )

    # 2) الصفوف المكررة — قبل الـ split، عشان نفس الصف ميقعش في الـ train والـ test
    before = len(df)
    df = df.drop_duplicates(subset=feats + ["label"])
    print(f"Duplicates removed: {before - len(df):,}  →  {len(df):,} rows")

    # 3) معلومة للتقرير: صفوف ليها نفس الـ features بالظبط بس labels مختلفة
    conflicts = int(df.duplicated(subset=feats, keep=False).sum())
    print(f"Rows with identical features but different labels: {conflicts:,}")

    # 4) جدول الشكل النهائي
    if show_preview:
        preview(df, feats)

    return df


def _main() -> None:
    """تشغيل الملف ده لوحده (زرار ▶ أو `python -m src.clean`):
    بيحمّل عيّنة → map_labels → clean → ويطبع جدول الداتا النهائية.
    ده للتجربة بس — التدريب الحقيقي في src/train.py.
    """
    import argparse

    import yaml

    from mygate_ai.features import feature_list
    from src.loaders import load_all

    ap = argparse.ArgumentParser(description="تنظيف عيّنة + عرض جدول الداتا النهائية")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nrows", type=int, default=20000, help="أول N صف من كل ملف")
    ap.add_argument("--rows", type=int, default=10, help="كام صف يتعرضوا في الجدول")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    df = load_all(cfg, nrows=args.nrows)
    df = map_labels(df, cfg["label_map"])
    df = clean(df, feature_list(cfg), show_preview=False)
    preview(df, feature_list(cfg), n=args.rows)


if __name__ == "__main__":
    # لما VS Code يشغّل الملف مباشرة (زرار ▶) بيحط فولدر src/ في الـ path
    # مش جذر المشروع، فـ `from src...` بتفشل. السطر ده بيضيف الجذر.
    # مش محتاج مع `python -m src.clean`، بس مابيضرّش.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    _main()
