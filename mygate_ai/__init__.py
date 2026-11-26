"""
mygate_ai — موديل تصنيف الـ network flows بتاع MyGate.

الـ package دي هي **كل** اللي الباك‑إند محتاجه. مفيهاش كود تدريب.

    from mygate_ai import load, predict, predict_many

    load()                     # مرة واحدة وقت الـ startup
    predict(flow)              # flow واحد   → dict
    predict_many(flows)        # دفعة        → list[dict]

استوردوا كل حاجة من `mygate_ai` مباشرةً — الموديولات الداخلية
(`inference`, `features`, `schema`) تفاصيل تنفيذ وممكن تتغيّر.

العقد الكامل في INTEGRATION.md.
"""

from __future__ import annotations

from .inference import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_VERSION,
    ModelIntegrityError,
    SklearnVersionError,
    load,
    predict,
    predict_many,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_VERSION",
    "ModelIntegrityError",
    "SklearnVersionError",
    "load",
    "predict",
    "predict_many",
]

# نسخة الـ package (PEP 440) — غير نسخة الموديل.
# نسخة الموديل في DEFAULT_MODEL_VERSION وفي schema.json.
__version__ = "1.0.0"
