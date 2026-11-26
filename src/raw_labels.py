"""
raw_labels.py — يرجّع الـ label الأصلية (DoS Hulk, slowloris, ...) لكل صف
في الـ golden test set، عشان نعرف الهجمات الفايتة والـ false positives من
أنهي نوع بالظبط.

⚠️ الملف ده **مابيلمسش** أي حاجة موجودة:
   - مابيقراش data/clean.parquet (الكاش مفيهوش raw_label أصلاً)
   - مابيكتبش ولا بيعدّل data/test_v1.parquet ولا data/clean.parquet
   - مابيغيّرش الموديل
   المخرج الوحيد: data/test_v1_raw_labels.parquet

الفكرة: نعيد نفس الـ pipeline بالحرف (نفس ترتيب قراءة الملفات، نفس
map_labels، نفس clean بنفس subset الـ dedup و keep="first"، نفس الـ split
والـ seed) بس مع عمود زيادة raw_label. لو الخطوات مطابقة، الـ test set
اللي هيطلع لازم يبقى **نفس الصفوف بنفس الترتيب**. وبنتأكد من ده بـ assert
على بصمة كل صف — مش على الـ label بس، لأن فيه مئات الآلاف من benign ورا
بعض فأي إزاحة ممكن تعدّي.

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
    """اسم الملف مختصر عشان الجداول تبان."""
    return name.replace(".pcap_ISCX.csv", "").replace("-WorkingHours", "")


def _source_file_column(cfg: dict, n_rows: int) -> np.ndarray:
    """عمود باسم الـ CSV اللي كل صف جاي منه.

    مش بنعدّل loaders.py — بنقرا **عمود الـ Label بس** من كل ملف بنفس
    ترتيب sorted(os.listdir) اللي load_cic2017 بيستخدمه، فنعرف عدد صفوف
    كل ملف، وبنبني العمود بـ np.repeat. الـ assert تحت بيضمن المحاذاة.
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
            f"\n❌ مجموع صفوف الملفات ({total:,}) مش = صفوف الـ DataFrame "
            f"({n_rows:,}).\n   يعني ترتيب/طريقة القراءة اتغيّرت. مش هنكمّل."
        )
    return np.repeat(names, counts)


# ---------------------------------------------------------------------
# 1) إعادة بناء الداتا مع raw_label
# ---------------------------------------------------------------------
def rebuild_with_raw(cfg: dict) -> pd.DataFrame:
    """نفس pipeline التدريب بالحرف + عمود raw_label."""
    feats = feature_list(cfg)

    print("=" * 72)
    print("1) قراءة الـ CSVs من الأول (من غير كاش)")
    print("=" * 72)
    df = load_all(cfg)  # نفس ترتيب القراءة: sorted(os.listdir)

    # الـ label الأصلية قبل ما map_labels تكتب فوقها.
    # .str.strip() عشان تطابق اللي map_labels بتعمله بالظبط.
    df["raw_label"] = df["label"].astype(str).str.strip().astype("category")

    print("\nعدد صفوف كل ملف (لعمود source_file):")
    df["source_file"] = pd.Categorical(_source_file_column(cfg, len(df)))

    print("\n" + "=" * 72)
    print("2) map_labels")
    print("=" * 72)
    df = map_labels(df, cfg["label_map"])

    # فحص قبل الـ dedup: صفوف متطابقة في الـ features + label بس
    # raw_labels مختلفة (زي DDoS و DoS Hulk — الاتنين بيتحولوا لـ dos)
    _report_raw_label_collisions(df, feats)

    print("\n" + "=" * 72)
    print("3) clean — نفس subset الـ dedup (features + label، من غير raw_label)")
    print("=" * 72)
    # raw_label عمود زيادة بس، مش في الـ subset — فالـ dedup بيتصرّف
    # بالظبط زي التدريب، و keep="first" (الافتراضي) بيخلّي أول صف في
    # ترتيب القراءة.
    df = clean(df, feats, show_preview=False)

    # الصفوف اللي ليها نفس الـ features بالظبط بس labels نهائية مختلفة
    # (الـ 1,438 اللي clean() بتطبعها). بنحسبها على الداتا **كلها** مش على
    # الـ test لوحده، لأن التوأم ممكن يكون في الـ train.
    conflict = df.duplicated(subset=feats, keep=False)
    df["feature_conflict"] = conflict.to_numpy()
    print(f"\nصفوف بـ features متطابقة و labels مختلفة: {int(conflict.sum()):,}")
    return df


def _report_raw_label_collisions(df: pd.DataFrame, feats: list[str]) -> None:
    """صفوف هتتشال في الـ dedup مع إن raw_label بتاعتها مختلفة."""
    print("\n--- صفوف متطابقة بـ raw_labels مختلفة ---")
    cols = [c for c in feats if c in df.columns] + ["label"]
    key = pd.util.hash_pandas_object(df[cols], index=False)

    tmp = pd.DataFrame({"key": key.to_numpy(), "raw": df["raw_label"].to_numpy()})
    nun = tmp.groupby("key", observed=True)["raw"].nunique()
    conflict_keys = nun[nun > 1]

    if conflict_keys.empty:
        print("مفيش — كل مجموعة صفوف متطابقة ليها raw_label واحدة.")
        return

    mask = tmp["key"].isin(set(conflict_keys.index))
    print(f"مجموعات متأثرة : {len(conflict_keys):,}")
    print(f"صفوف متأثرة    : {int(mask.sum()):,}")
    print("\nأكتر 10 تركيبات raw_label بتتلخبط مع بعض:")
    combos = (
        tmp[mask]
        .groupby("key", observed=True)["raw"]
        .agg(lambda s: " + ".join(sorted(set(map(str, s)))))
        .value_counts()
        .head(10)
    )
    print(combos.to_string())
    print(
        "\nملاحظة: keep='first' بيخلّي أول صف في ترتيب القراءة، فالـ raw_label\n"
        "        اللي بتفضل هي بتاعة أول ملف أبجدياً. ده بيأثر على الـ\n"
        "        breakdown تحت بس، مش على الموديل — الموديل شاف الـ label\n"
        "        الموحّدة (dos) في الحالتين."
    )


# ---------------------------------------------------------------------
# 2) نفس الـ split + التحقّق من المطابقة
# ---------------------------------------------------------------------
def same_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """نفس train_test_split بنفس الـ seed → بيرجّع الـ test set."""
    print("\n" + "=" * 72)
    print("4) نفس الـ split (نفس test_size و random_state)")
    print("=" * 72)
    _, df_test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df["label"],
    )
    print(f"test: {len(df_test):,} صف")
    return df_test


def verify_alignment(df_test: pd.DataFrame, golden: pd.DataFrame, feats: list[str]):
    """assert إن الصفوف مطابقة لـ test_v1.parquet صف بصف.

    بنقارن بصمة كل صف (كل الـ features + label) مش الـ label لوحدها —
    الـ label ضعيفة كمفتاح لأن فيه 622 ألف benign.
    """
    print("\n" + "=" * 72)
    print("5) التحقّق من المطابقة مع " + GOLDEN)
    print("=" * 72)

    if len(df_test) != len(golden):
        raise SystemExit(
            f"\n❌ العدد مختلف: أعدنا البناء فطلع {len(df_test):,} صف، "
            f"و{GOLDEN} فيه {len(golden):,}.\n"
            f"   يعني خطوة في الـ pipeline اتغيّرت. مش هنكمّل."
        )
    print(f"العدد        : {len(df_test):,} ✅")

    lbl_new = df_test["label"].astype(str).to_numpy()
    lbl_gold = golden["label"].astype(str).to_numpy()
    n_bad_label = int((lbl_new != lbl_gold).sum())

    cols = [c for c in feats if c in golden.columns] + ["label"]
    k_new = pd.util.hash_pandas_object(df_test[cols], index=False).to_numpy()
    k_gold = pd.util.hash_pandas_object(golden[cols], index=False).to_numpy()
    n_bad_row = int((k_new != k_gold).sum())

    print(f"labels مختلفة : {n_bad_label:,}")
    print(f"صفوف مختلفة  : {n_bad_row:,}   (بصمة كل الـ features + label)")

    if n_bad_label or n_bad_row:
        first = int(np.argmax((k_new != k_gold) | (lbl_new != lbl_gold)))
        raise SystemExit(
            f"\n❌ الصفوف مش متطابقة. أول اختلاف عند الصف {first:,}:\n"
            f"   أعدنا البناء : label={lbl_new[first]}\n"
            f"   {GOLDEN}     : label={lbl_gold[first]}\n"
            f"\nمش هنحاول نصلّح ده — معناه إن خطوة في الـ pipeline اتغيّرت\n"
            f"(ترتيب القراءة، الـ dedup، الـ map، أو الـ seed). لازم نفهم\n"
            f"السبب الأول."
        )

    print("\n✅ مطابق صف بصف — الـ raw_labels متحاذية مع الـ golden set.")

    # فحص تالت: map_labels(raw_label) لازم تدّي نفس الـ label
    lm = dict(_cfg["label_map"])
    mapped = df_test["raw_label"].astype(str).map(lm).to_numpy()
    n_bad_map = int((mapped != lbl_gold).sum())
    print(f"map_labels(raw_label) == label : {n_bad_map:,} اختلاف")
    if n_bad_map:
        raise SystemExit("❌ الـ raw_label مابتتحولش للـ label الموجودة.")
    return df_test


# ---------------------------------------------------------------------
# 3) الـ breakdown
# ---------------------------------------------------------------------
def breakdown(golden: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> None:
    print("\n" + "=" * 72)
    print("6) BREAKDOWN — الهجمات الفايتة والـ false positives")
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

    # ---- أ) الهجمات الفايتة ----
    missed = (y != BENIGN) & (pred == BENIGN)
    print(f"\n--- أ) الهجمات الفايتة: {int(missed.sum())} ---")
    print("\nحسب raw_label:")
    print(pd.Series(raw[missed]).value_counts().to_string())

    print("\nrecall لكل raw_label (كل الأنواع):")
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

    # ---- ب) الـ false positives ----
    fp = (y == BENIGN) & (pred != BENIGN)
    print(f"\n--- ب) الـ false positives: {int(fp.sum())} ---")
    print("\nاتصنّفوا كإيه:")
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

    # ---- ب2/ب3/ب4) الـ FP حسب الملف ----
    print("\n--- ب2) الـ FP حسب الملف × التصنيف المتوقّع ---")
    print(
        pd.crosstab(
            pd.Series(src[fp], name="source_file"),
            pd.Series(pred[fp], name="اتصنّف"),
            margins=True,
        ).to_string()
    )

    sure = fp & (p_attack >= 0.99)
    print(
        f"\n--- ب3) نفس الجدول للـ FP بـ probability >= 0.99 فقط "
        f"({int(sure.sum())}) ---"
    )
    print(
        pd.crosstab(
            pd.Series(src[sure], name="source_file"),
            pd.Series(pred[sure], name="اتصنّف"),
            margins=True,
        ).to_string()
    )

    if "dst_port" in golden.columns:
        p444 = fp & (golden["dst_port"].to_numpy() == 444)
        print(f"\n--- ب4) الـ FP على بورت 444 ({int(p444.sum())}) ---")
        if p444.any():
            print(
                pd.DataFrame({"source_file": src[p444], "اتصنّف": pred[p444]})
                .value_counts()
                .to_string()
            )

    # ---- أ2) الفايتة حسب الملف ----
    print(f"\n--- أ2) الـ {int(missed.sum())} الفايتة حسب الملف ---")
    print(
        pd.crosstab(
            pd.Series(src[missed], name="source_file"),
            pd.Series(raw[missed], name="raw_label"),
            margins=True,
        ).to_string()
    )

    # ---- د) الصفوف المتعارضة ----
    if "feature_conflict" in meta.columns:
        conf = meta["feature_conflict"].to_numpy().astype(bool)
        print("\n--- د) صفوف features متطابقة بـ labels مختلفة ---")
        print(f"في الـ golden set          : {int(conf.sum()):,}")
        print(
            f"منهم ضمن الـ {int(missed.sum())} الفايتة : {int((conf & missed).sum()):,}"
        )
        print(f"منهم ضمن الـ {int(fp.sum())} FP       : {int((conf & fp).sum()):,}")
        if conf.any():
            print("\nتوزيعهم حسب raw_label:")
            print(pd.Series(raw[conf]).value_counts().head(10).to_string())

    # ---- ج) الـ 12 suspicious ----
    susp = (pred == BENIGN) & (p_attack >= 0.5)
    print(
        f"\n--- ج) suspicious (argmax=benign بس probability >= 0.5): "
        f"{int(susp.sum())} ---"
    )
    if susp.any():
        print(
            pd.DataFrame(
                {
                    "raw_label": raw[susp],
                    "الحقيقة": y[susp],
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
        help=f"يستخدم {OUT} الموجود بدل ما يعيد قراءة الـ CSVs (للـ breakdown بس)",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        _cfg = yaml.safe_load(fh)
    feats = feature_list(_cfg)

    if not os.path.isfile(GOLDEN):
        raise SystemExit(f"مفيش {GOLDEN} — شغّل python -m src.train الأول.")
    golden = pd.read_parquet(GOLDEN)

    if args.skip_rebuild and os.path.isfile(OUT):
        meta = pd.read_parquet(OUT)
        print(f"استخدمنا {OUT} الموجود ({len(meta):,} صف)")
        missing = [c for c in ("source_file", "feature_conflict") if c not in meta]
        if missing:
            raise SystemExit(
                f"{OUT} قديم — ناقصه {missing}. شغّل من غير --skip-rebuild."
            )
        if len(meta) != len(golden):
            raise SystemExit(
                f"{OUT} فيه {len(meta):,} صف و{GOLDEN} فيه {len(golden):,}."
            )
    else:
        df = rebuild_with_raw(_cfg)
        df_test = same_split(df, _cfg)
        df_test = verify_alignment(df_test, golden, feats)

        meta = pd.DataFrame(
            {
                "row": np.arange(len(df_test)),  # يطابق ترتيب test_v1.parquet
                "raw_label": df_test["raw_label"].astype(str).to_numpy(),
                "source_file": df_test["source_file"].astype(str).to_numpy(),
                "feature_conflict": df_test["feature_conflict"].to_numpy(),
            }
        )
        os.makedirs("data", exist_ok=True)
        meta.to_parquet(OUT, index=False)
        print(f"\n✅ اتحفظ → {OUT}  ({len(meta):,} صف)")
        print("\nصفوف الـ golden set حسب الملف:")
        print(meta["source_file"].value_counts().to_string())

    breakdown(golden, meta, _cfg)


_cfg: dict = {}

if __name__ == "__main__":
    main()
