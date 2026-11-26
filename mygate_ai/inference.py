"""
inference.py — اللي التيم بيستدعيه جوا core/controllers/ai_engine.py

⚠️ استوردوا من `mygate_ai` مباشرةً، مش من `mygate_ai.inference`:
       from mygate_ai import load, predict, predict_many
   الموديول ده تفصيلة تنفيذ. (كان اسمه predict.py، واتغيّر لأن
   `mygate_ai.predict` كان بيبقى غامض — الدالة والموديول بنفس الاسم.)

العقد (نهائي — متفق عليه مع التيم، §3):
    predict(flow_dict) -> {
        "is_attack": bool,
        "probability": float,          # احتمال إن الـ flow هجوم (0..1)
        "attack_type": "benign" | "portscan" | "brute_force" | "dos",
        "model_version": str,
    }
    predict_many([flow, ...]) -> [ نفس العقد لكل flow، بنفس الترتيب ]

الـ flow_dict جاي من cicflowmeter==0.2.0 زي ما هو (أسماء live snake_case).
مفاتيح زيادة (src_ip, dst_ip, timestamp, ...) بتتجاهل.
مفاتيح ناقصة → NaN → الـ imputer بيملّيها بالـ median بتاع التدريب.

الاستخدام:
    from mygate_ai import load, predict, predict_many
    load()                       # مرة واحدة وقت الـ startup
    predict(flow)                # flow واحد
    predict_many(flows)          # دفعة — أسرع بكتير وقت الـ DoS

⚠️ كل القيم الراجعة **أنواع Python عادية** مش numpy — لأن psycopg2
   مابيعرفش يحوّل numpy.float64 وبيرمي "can't adapt type numpy.float64".
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .features import ENGINEERED, build_features
from .schema import to_canonical

BENIGN = "benign"

# النسخة اللي بتتشحن جوا الـ package. لتغييرها: صدّر النسخة الجديدة في
# mygate_ai/models/<v>/ وغيّر السطر ده.
DEFAULT_MODEL_VERSION = "v1"
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", DEFAULT_MODEL_VERSION
)

# حالة محمّلة مرة واحدة وقت الـ startup — مش مع كل flow
_model = None
_schema: dict | None = None
_features: list[str] = []
_base_features: list[str] = []
_classes: list[str] = []
_benign_index: int = 0
_model_version: str = ""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityError(RuntimeError):
    """model.pkl مش مطابق للـ sha256 اللي في schema.json."""


class SklearnVersionError(RuntimeError):
    """نسخة scikit-learn مختلفة عن اللي الموديل اتصدّر بيها."""


def load(model_dir: str | None = None, strict: bool = True) -> None:
    """يحمّل model.pkl + schema.json **مرة واحدة** وقت الـ startup.

    model_dir: لو None بياخد النسخة المشحونة جوا الـ package
               (mygate_ai/models/v1 — نسبي لـ __file__).
    strict   : True (الافتراضي) → أي فحص بيفشل بيرمي exception والموديل
               مابيتحمّلش خالص. False → تحذير بس، **للتطوير بس**.

    الفحوص:
      1. sha256 بتاع model.pkl — بيتحسب على bytes الملف **قبل** joblib.load.
         ده مش تحقّق شكلي: الـ pickle بيشغّل كود وقت الفك، فملف متبدّل =
         تنفيذ كود عشوائي بصلاحيات الـ appliance. الفحص الأول يعني
         إننا مابنفكّش ملف محصلناش عليه ثقة.
      2. نسخة scikit-learn — الـ pickle مش مضمون بين النسخ، وممكن يتفك
         ويدّي نتايج مختلفة بصمت (§3).

    الـ appliance مش هيقف لو ده رمى: الباك‑إند بيمسك الـ exception،
    AIEngine.status() بترجّع loaded=False والسبب، و Suricata بيفضل شغال.
    شوف INTEGRATION.md.
    """
    global _model, _schema, _features, _base_features
    global _classes, _benign_index, _model_version

    import joblib
    import sklearn

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"مفيش schema.json في {model_dir}")
    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)

    model_path = os.path.join(model_dir, schema.get("model_file", "model.pkl"))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"مفيش {model_path}")

    def _fail(exc: Exception, msg: str) -> None:
        if strict:
            raise exc
        print(f"⚠️  [mygate_ai] strict=False — {msg}")

    # --- فحص 1: سلامة الملف، قبل أي joblib.load ---
    expected = schema.get("model_file_sha256")
    if expected:
        actual = _sha256(model_path)
        if actual != expected:
            detail = (
                f"sha256 بتاع {model_path} مش مطابق لـ schema.json.\n"
                f"  المتوقّع: {expected}\n"
                f"  الفعلي  : {actual}\n"
                f"الملف اتغيّر بعد الـ export. الـ pickle بيشغّل كود وقت الفك، "
                f"فمش هنفكّه.\n"
                f"الحل: هات model.pkl الأصلي، أو load(..., strict=False) "
                f"للتطوير بس."
            )
            _fail(ModelIntegrityError(detail), detail)
    elif strict:
        raise ModelIntegrityError(
            f"{schema_path} مفيهوش model_file_sha256 — مش قادرين نتحقّق من "
            f"سلامة model.pkl. أعد الـ export، أو load(..., strict=False)."
        )

    # --- فحص 2: نسخة sklearn ---
    want = schema.get("sklearn_version")
    if want and want != sklearn.__version__:
        detail = (
            f"نسخة scikit-learn مختلفة: عندك {sklearn.__version__}، "
            f"الموديل اتصدّر بـ {want}.\n"
            f"الـ pickle مش مضمون بين النسخ — ممكن يتفك ويدّي نتايج مختلفة "
            f"بصمت.\n"
            f"الحل: pip install scikit-learn=={want}"
        )
        _fail(SklearnVersionError(detail), detail)

    model = joblib.load(model_path)

    _model = model
    _schema = schema
    _features = list(schema["features"])
    # الـ features المحسوبة موجودة خلاص في schema["features"]، فبنشيلها من
    # القايمة الأساسية عشان feature_list() ماتضيفهاش تاني
    _base_features = [f for f in _features if f not in ENGINEERED]
    _classes = [str(c) for c in model.classes_]
    _benign_index = _classes.index(BENIGN)
    _model_version = str(schema["model_version"])

    print(
        f"[mygate_ai] loaded {model_dir} | version={_model_version} | "
        f"{len(_features)} features | classes={_classes}"
    )


def _require_loaded() -> None:
    if _model is None:
        raise RuntimeError(
            "الموديل مش محمّل. نادِ load() مرة واحدة وقت الـ startup "
            "قبل أول predict()/predict_many().\n"
            "في الباك‑إند: جوا core/controllers/ai_engine.py عند تشغيل التطبيق، "
            "مش جوا كل request."
        )


def _build_X(flows: list[dict]) -> pd.DataFrame:
    """flows خام → X جاهزة للموديل، بنفس مسار التدريب بالظبط (§7).

    to_canonical(live) بيشيل المفاتيح الزيادة (src_ip, timestamp, ...)،
    و build_features بيرتّب الأعمدة ويملّي الناقصة NaN.
    """
    raw = pd.DataFrame(flows)
    df = to_canonical(raw, "live")

    cfg = {"features": _base_features, "drop_features": []}
    X = build_features(df, cfg)
    X = X.reindex(columns=_features)  # الترتيب بالظبط زي التدريب

    # القيم ممكن توصل strings لو الـ flow عدّى بـ JSON/HTTP
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Inf → NaN — نفس اللي clean() بيعمله في التدريب
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.astype("float32")


def predict_many(flows: list[dict]) -> list[dict]:
    """نفس العقد لكل flow، **بنفس ترتيب المدخلات**.

    استخدمها وقت الـ DoS: الآلاف من الـ flows في الثانية. الفرق ضخم —
    تكلفة sklearn الثابتة (~3ms) بتتدفع مرة واحدة للدفعة كلها بدل مرة
    لكل flow (شوف throughput في metrics.json).
    """
    _require_loaded()
    if not flows:
        return []

    X = _build_X(flows)
    proba = _model.predict_proba(X)

    p_attack = 1.0 - proba[:, _benign_index]  # §3
    best = proba.argmax(axis=1)

    results = []
    for i in range(len(flows)):
        attack_type = _classes[best[i]]
        results.append(
            {
                "is_attack": bool(attack_type != BENIGN),
                "probability": round(float(p_attack[i]), 6),
                "attack_type": str(attack_type),
                "model_version": _model_version,
            }
        )
    return results


def predict(flow: dict) -> dict:
    """flow واحد. نفس منطق predict_many بالظبط — مفيش مسار تاني يختلف عنه."""
    return predict_many([flow])[0]
