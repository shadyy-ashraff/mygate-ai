# MyGate AI

موديل كشف الهجمات لـ MyGate (portscan / brute_force / dos) على flows بتاعة CICFlowMeter.

للربط مع الباك‑إند: **[INTEGRATION.md](INTEGRATION.md)**.

## الهيكل

الـ imports في اتجاه واحد: `src/` → `mygate_ai/`. الـ package مابتعرفش
حاجة عن كود التدريب، عشان تشتغل لوحدها جوا الـ wheel.

```
mygate_ai/          📦 اللي بيتسلّم — inference بس
├── __init__.py     load · predict · predict_many · الاستثناءات · DEFAULT_MODEL_*
├── inference.py    العقد + فحوص load()
├── schema.py       to_canonical() + COLUMN_MAPS
├── features.py     feature_list() + build_features()
└── models/v1/      model.pkl · schema.json · metrics.json

src/                🔧 التدريب — مش بيتشحن
├── loaders.py · clean.py · train.py · evaluate.py · export.py
```

## الأوامر

```powershell
.\.venv\Scripts\python -m src.train                 # تدريب كامل  → mygate_ai/models/v1/
.\.venv\Scripts\python -m src.train --nrows 20000   # عيّنة عشوائية → mygate_ai/models/smoke/
.\.venv\Scripts\python -m src.evaluate --write      # إعادة تقييم موديل موجود، من غير تدريب
.\.venv\Scripts\python -m src.clean --nrows 3000    # تنظيف عيّنة + جدول الداتا
.\.venv\Scripts\python -m pytest -v
.\.venv\Scripts\python -m build --wheel             # → dist/mygate_ai-*.whl
```

## خريطة النوت‑بوك → الملفات

| خلية النوت‑بوك | بتروح فين |
|---|---|
| 0, 1, 3 (imports / pip) | `requirements.txt` |
| 5 (تحميل الداتا) + تقليل الرام | `src/loaders.py` |
| 7, 10, 12, 14 (EDA ورسومات) | **تفضل في النوت‑بوك** (للتقرير) |
| 17 (التنظيف) | `src/clean.py` → `clean()` |
| 19 (الـ labels) | `src/clean.py` → `map_labels()` + `config.yaml` |
| 21 (split) | `src/train.py` → `split()` |
| 23 (scaling) | ❌ مش محتاجينها (tree models) |
| 29 (SMOTE) | ❌ مش في v1 — `class_weight="balanced_subsample"` بدالها |
| 39 / 41 (RF / XGBoost) | `src/train.py` → `build_model()` |
| 37, 44, 52, 54 (LogReg / LightGBM / NN) | تفضل في النوت‑بوك للمقارنة بس |
| 46, 58, 60, 62, 68 (التقييم) | `src/evaluate.py` |
| 70, 72 (SHAP) | النوت‑بوك (للتقرير) |
| 74 (الحفظ) | `src/export.py` |
| — (جديد) | `mygate_ai/inference.py` |

## إضافة داتاست جديدة
1. خريطة أعمدة في `mygate_ai/schema.py` → `COLUMN_MAPS`
2. loader في `src/loaders.py`
3. أسماء الـ labels بتاعتها في `config.yaml` → `label_map`

## إضافة feature جديدة
function واحدة في `mygate_ai/features.py` → `ENGINEERED`

## المخرجات
`mygate_ai/models/<version>/` → `model.pkl` · `schema.json` · `metrics.json`،
وكلهم بيتحزموا مع الكود في `dist/mygate_ai-*.whl`.

**`data/test_v1.parquet` هو الـ golden test set — مايتمسحش أبداً.**
كل نسخة جديدة بتتقاس عليه بالظبط عشان المقارنة تبقى عادلة.
