# MyGate AI — دليل الربط للباك‑إند

موديل تصنيف الـ **network flows** (CICFlowMeter features, L3/L4).
مش بيشوف HTTP payloads — الـ SQLi / XSS / Web brute force دي شغل Suricata.

---

## 1. التثبيت

```bash
pip install mygate_ai-1.0.0-py3-none-any.whl
```

الـ wheel جواه الموديل نفسه، فمفيش ملفات تانية تتنقل ولا مسارات تتظبط.

⚠️ **الـ wheel محتاج Python 3.12** (`Requires-Python: >=3.12,<3.13`).
ده لأن `model.pkl` معمول بـ pickle والموديل اتدرّب على 3.12.10، فنفس
منطق تثبيت نسخ المكتبات ينطبق على نسخة Python نفسها. لو الـ VM عندكم
على نسخة تانية كلّمونا قبل ما تغيّروا حاجة — شادي هيأكّد النسخة معاكم.

**النسخ مثبّتة بالظبط** ومتحطّة كـ dependencies تلقائياً:

```
scikit-learn==1.6.1
pandas==3.0.6
numpy==2.5.3
joblib==1.6.0
```

⚠️ `model.pkl` متسلسل بـ **pickle**. نسخة `scikit-learn` مختلفة ممكن تفك الملف
وتدّي نتايج مختلفة **بصمت**. عشان كده `load()` بترفض التشغيل لو النسخة مختلفة.
متعملوش `pip install -U scikit-learn` في نفس الـ environment.

---

## 2. العقد

```python
predict(flow: dict) -> {
    "is_attack": bool,     # True لو attack_type != "benign"
    "probability": float,  # P(attack) = 1 - P(benign)، من 0 لـ 1
    "attack_type": str,    # "benign" | "portscan" | "brute_force" | "dos"
    "model_version": str,  # "v1"
}

predict_many(flows: list[dict]) -> list[dict]   # نفس العقد لكل flow، بنفس الترتيب
```

في الـ DB: `ai_confidence = probability` · `ai_label = attack_type`.

**المدخل:** flow خام من `cicflowmeter==0.2.0` زي ما هو (أسماء snake_case).

| الحالة | السلوك |
|---|---|
| مفاتيح زيادة (`src_ip`, `dst_ip`, `timestamp`, `protocol`) | بتتجاهل |
| مفاتيح ناقصة | → `NaN` → الـ imputer بيملّيها بالـ median بتاع التدريب |
| `Inf` (لما `flow_duration = 0`) | → `NaN` → نفس الحاجة |
| قيم strings (لو الـ flow عدّى بـ JSON/HTTP) | بتتحوّل لأرقام؛ اللي مش رقم → `NaN` |
| `predict_many([])` | `[]` |

القيم الراجعة **أنواع Python عادية** (`bool`, `float`, `str`) — مش numpy.
ده مقصود: `psycopg2` مابيعرفش يحوّل `numpy.float64` وبيرمي
`can't adapt type 'numpy.float64'`.

---

## 3. `load()` — مرة واحدة وقت الـ startup

`load()` بتقرا 19 MB من الديسك وبتفك pickle. **متناديهاش جوا كل request.**

```python
# core/controllers/ai_engine.py
import logging
from mygate_ai import load, predict, predict_many

log = logging.getLogger(__name__)


class AIEngine:
    def __init__(self) -> None:
        self._loaded = False
        self._error: str | None = None

    def startup(self) -> None:
        """بتتنادى مرة واحدة وقت تشغيل التطبيق."""
        try:
            load()                    # النسخة المشحونة جوا الـ package
            self._loaded = True
        except Exception as exc:      # الـ appliance بيفضل شغال من غير AI
            self._error = f"{type(exc).__name__}: {exc}"
            log.error("AI model failed to load — running on Suricata only:\n%s", exc)

    def status(self) -> dict:
        return {"loaded": self._loaded, "error": self._error}

    def classify(self, flow: dict) -> dict | None:
        if not self._loaded:
            return None               # مفيش رأي من الـ AI؛ Suricata لوحدها
        return predict(flow)

    def classify_many(self, flows: list[dict]) -> list[dict]:
        if not self._loaded:
            return []
        return predict_many(flows)
```

**مهم: فشل الـ load مش بيوقّف الـ appliance.** امسكوا الـ exception،
خلّوا `status()` ترجّع `loaded=False` والسبب، وسيبوا Suricata شغالة.
`classify()` بترجّع `None` وقتها، و`core/decision.py` بتاخد قرارها من غير AI.

### الفحوص اللي `load()` بتعملها

| الفحص | لو فشل |
|---|---|
| `sha256` بتاع `model.pkl` ضد `schema.json` | `ModelIntegrityError` |
| نسخة `scikit-learn` ضد `schema.json` | `SklearnVersionError` |

الـ sha256 بيتحسب على bytes الملف **قبل** `joblib.load` — لأن فك الـ pickle
بيشغّل كود. ملف متبدّل = تنفيذ كود عشوائي بصلاحيات الـ appliance، فالفحص
بيمنع إننا نفك ملف محصلناش عليه ثقة.

```python
load()                  # strict=True — الافتراضي، وده اللي يتستخدم في الإنتاج
load(strict=False)      # تحذير بدل exception — للتطوير بس
load("/path/to/model")  # نسخة من مكان تاني
```

`predict()` قبل `load()` بترمي `RuntimeError` برسالة واضحة.

---

## 4. اختيار الـ threshold

على الـ golden test set: **748,682 flow** (622,161 سليم · 126,521 هجمة).

| threshold | detection_rate | هجمات فايتة | FPR | false positives |
|---|---|---|---|---|
| **0.50** | 0.99954 | 58 | 0.000937 | 583 |
| **0.70** | 0.99805 | 247 | 0.000571 | 355 |
| **0.80** | 0.99693 | 389 | 0.000530 | 330 |
| **0.90** | 0.99497 | 637 | 0.000503 | 313 |
| **0.95** | 0.99220 | 987 | 0.000500 | 311 |
| **0.99** | 0.98294 | 2,158 | 0.000479 | 298 |

### التوصية

```
alert       عند probability >= 0.5
auto-block  عند probability >= 0.9
            بشرط:  الـ IP مش في trusted_ips
            و:     الـ block مؤقت (time-based) مش دائم
```

### ⚠️ القرار من `probability` — مش من `is_attack`

`attack_type` و `is_attack` **للعرض والتسجيل بس**. كل الشروط فوق على
`probability`، وده مقصود: الاتنين ممكن يختلفوا.

`attack_type` هو الـ **argmax** — أعلى class لوحدها. و`probability` هو
`1 - P(benign)` — مجموع كل الهجوم. فممكن الاحتمال يبقى موزّع على أكتر من
نوع هجوم لدرجة إن مجموعهم يعدّي `benign` من غير ما أي واحد فيهم لوحده
يعدّيها. حصل **12 مرة** على الـ golden set، **4 منهم هجمات حقيقية**
(التفاصيل في §8).

**لو `probability >= 0.5` و `attack_type == "benign"`:** سجّلوه alert بنوع
**`"suspicious"`** — الموديل شايف هجوم بس مش متأكد من نوعه.

```python
# core/decision.py
r = ai.classify(flow)
if r is None:
    ...                                   # الـ AI مش محمّل — Suricata بس
elif r["probability"] >= 0.5:
    label = r["attack_type"]
    if label == "benign":
        label = "suspicious"              # الـ 12 حالة بتاعة §8
    alert(ip, label, confidence=r["probability"])
    if r["probability"] >= 0.9 and ip not in trusted_ips:
        block_temporarily(ip)
```

لاحظوا إن `ai_label` في الـ DB يفضل `attack_type` زي ما هو (`"benign"`)
عشان الأرقام تفضل قابلة للمقارنة — `"suspicious"` نوع الـ alert مش
تصنيف الموديل.

### ليه

من **0.5** لـ **0.99** الـ false positives بتنزل من **583** لـ **298** بس — يعني
النص تقريباً مابيتحلّش بالعتبة مهما رفعناها. في المقابل الهجمات اللي بتفوت
بتزيد من **58** لـ **2,158**، أي **37 ضعف**. المقايضة سيئة جداً.

السبب إن الموديل **واثق في غلطه**: فيه ~300 flow سليم الموديل مقتنع إنهم
هجوم بـ `probability >= 0.99`. العتبة لوحدها مش هتنقذ الـ 300 دول.

**وعشان كده الشرطين مش اختياريين:**

- **`trusted_ips`** — الـ whitelist هو خط الدفاع الوحيد ضد الـ 300 دول. من
  غيره مستخدم شرعي ممكن يتقفل وهو بيعمل حاجة عادية.
- **الـ block المؤقت** — الغلطة بتنتهي لوحدها بدل ما تفضل لحد ما حد ياخد باله.

`0.5` للـ alert (تسجيل بس، بلا ضرر) و`0.9` للـ block (إجراء فعلي) بيفصلوا
بين "شوف ده" و"اقفل ده".

الأرقام دي كلها في `mygate_ai/models/v1/metrics.json` → `thresholds`.

---

## 5. الأداء

مقيسة على الـ golden test set، والموديل محفوظ بـ `n_jobs=1`:

| | |
|---|---|
| `predict()` — flow واحد | p50 **3.1 ms** · p95 3.7 ms → ~**320 flow/sec** |
| `predict_many()` — دفعة 100 | ~**20,000 flow/sec** |
| `predict_many()` — دفعة 1,000 | ~**94,000 flow/sec** |

**وقت الـ DoS استخدموا `predict_many()`.** الهجوم بيولّد آلاف الـ flows في
الثانية، و`predict()` في لوب هتقف عند ~320/sec. تكلفة sklearn الثابتة (~3ms)
بتتدفع مرة واحدة للدفعة كلها بدل مرة لكل flow — الفرق **300 ضعف**.

```python
# بدل:  results = [predict(f) for f in flows]
results = predict_many(flows)
```

`n_jobs=1` مقصود: توزيع الـ 100 شجرة على كل الأنوية لصف واحد بيكلّف أكتر
من الشغل نفسه (14.6 ms مقابل 2.7 ms).

---

## 6. جودة الموديل

`macro-F1 = 0.9980` · `macro-recall = 0.9995` · `binary ROC-AUC = 1.0`

| class | precision | recall | support |
|---|---|---|---|
| benign | 0.9999 | 0.9991 | 622,161 |
| brute_force | 1.0000 | 1.0000 | 2,746 |
| dos | 0.9972 | 0.9994 | 96,529 |
| portscan | 0.9887 | 0.9995 | 27,246 |

اتدرّب على **1,746,923** flow من CIC-IDS2017 (بعد شيل المكرر).

### حدود لازم تعرفوها

**١) الأداة اللي بتطلّع الـ features مش هي نفسها — لسه متحققناش من ده.**

الموديل اتدرّب على CIC-IDS2017، والـ CSVs بتاعته طلّعها
**CICFlowMeter (Java)**. اللايف عندكم بيجي من
**`cicflowmeter==0.2.0` (Python)** — إعادة تنفيذ مستقلة.

الأداتين ممكن يحسبوا نفس الـ feature بشكل مختلف:

- `flow_duration` — وحدات (microseconds مقابل milliseconds)
- الـ **flow timeout** — امتى الـ flow يتقفل ويتعدّ واحد جديد
- `init_fwd_win_byts` / `init_bwd_win_byts` — تعريف "أول window"
- الـ Active/Idle thresholds

لو فيه فرق هنا، الموديل هيشوف أرقام **مش زي اللي اتدرّب عليها**، والنتايج
في §6 فوق مش هتنطبق — حتى لو الكود شغّال صح.

الـ tests عندنا بتثبت إن **مسار الكود** مطابق (`dict → to_canonical →
build_features → predict_proba` بيدّي نفس أرقام التدريب على نفس الصفوف)،
بس **مابتثبتش تطابق الأداتين** — ده مستحيل نختبره من غير داتا حقيقية.

**اللي هيحسم ده:** CSV من معمل تيم النيتورك، متطلّع بـ
`cicflowmeter==0.2.0` من ترافيك معروف (benign + هجوم مولّد). لما
يوصلنا هنقارن توزيع كل feature بالتدريب ونرجّعلكم النتيجة. **لحد ما
ده يحصل، اعتبروا أرقام §6 حد أعلى نظري مش أداء مضمون.**

**٢) مفيش تغطية لـ**: SQLi · XSS · Web brute force · Heartbleed (→ Suricata)،
و Bot · Infiltration (حد موثّق، مش في v1).

**٣)** أهم feature هي `dst_port` (12.6%)، وده ممكن يكون shortcut من
الداتاست. في تجربة جاية (v1.1) هنشيلها ونقارن على نفس الـ golden set.

**٤)** الموديل بيشوف **flow واحد معزول** — مش بيشوف تاريخ ولا علاقة بين
الـ flows. هجوم بطيء موزّع على flows كتير ممكن يفلت.

---

## 7. `schema.json` — تفسير الحقول

`mygate_ai/models/v1/schema.json`

| الحقل | معناه |
|---|---|
| `model_version` | النسخة اللي بترجع في العقد (`"v1"`) |
| `trained_at` | وقت التدريب (UTC, ISO-8601) |
| `cicflowmeter_version` | نسخة أداة استخراج الـ flows المتوقّعة (`"0.2.0"`) |
| `input` | وصف المدخل وسلوك المفاتيح الزيادة/الناقصة |
| `features` | **69 اسم بالترتيب** — الترتيب حرفي، `mygate_ai` بيبني الأعمدة بيه |
| `n_features` | `len(features)` |
| `drop_features` | أعمدة اتشالت عن قصد (مكرر + zero-variance) |
| `classes` | مخرجات الموديل مرتّبة — نفس ترتيب أعمدة `predict_proba` |
| `positive_probability` | `"1 - P(benign)"` — تعريف `probability` في العقد |
| `label_map` | labels CIC-IDS2017 الأصلية → الـ 4 classes |
| `model.type` / `model.params` | `random_forest` + الـ hyperparameters |
| `model.pipeline` | خطوات الـ Pipeline: `["impute", "model"]` |
| `model_file` / `model_file_sha256` | اسم الملف وهاشه — `load()` بتتحقّق منه |
| `n_jobs_at_export` | `1` — مضبوط للتنبؤ بـ flow واحد |
| `sklearn_version` | **لازم تطابق اللي عندكم** — `load()` بترفض غير كده |
| `pandas_version` / `numpy_version` / `python_version` | بيئة التدريب |
| `sources` | مصادر الداتا (`["cic2017"]`) |
| `n_train` / `n_test` | 1,746,923 / 748,682 |
| `random_state` / `test_size` | `42` / `0.3` |
| `smoke_run_nrows` | `null` للنسخ الحقيقية؛ رقم = دي مخرجات تجربة **مش للتسليم** |

---

## 8. أسئلة متكررة

**الموديل بيتحدّث إزاي؟** wheel جديد فيه `DEFAULT_MODEL_VERSION` جديدة.
`model_version` في الرد بيتغيّر لوحده، فتقدروا تقارنوا النسخ في الـ DB.

**ينفع نشغّل نسختين مع بعض؟** `load()` بتحمّل واحدة في الـ process.
لمقارنة نسختين، شغّلوا processes منفصلة.

**thread-safe؟** `predict()` و `predict_many()` قراءة بس بعد `load()`،
و`RandomForest.predict_proba` آمن للقراءة المتوازية. نادوا `load()` مرة
واحدة قبل ما أي worker يبدأ.

**`is_attack` و`probability` ممكن يختلفوا؟** آه — **في اتجاه واحد بس**،
ولازم تاخدوا بالكوا منه.

**مستحيل:** `is_attack=True` مع `probability < 0.5`.
لو الـ argmax مش `benign` يبقى `P(benign) <= 0.5` بالضرورة — لأن لو كانت
أكبر من 0.5، الباقي كله مجموعه أقل من 0.5 فكل واحد فيهم أقل منها وتبقى هي
الـ argmax. (اتأكدنا: **صفر** حالة على الـ golden set كله.)

**ممكن:** `is_attack=False` مع `probability >= 0.5`. مثال حقيقي من الـ golden set:

```
benign=0.497   dos=0.461   portscan=0.041   brute_force=0.000
→ attack_type = "benign"   is_attack = False   probability = 0.503
```

الاحتمال موزّع على أكتر من class هجوم، فمجموع الهجوم أكبر من `benign`
من غير ما أي class هجوم لوحدها تعدّيها. على الـ golden set (748,682 flow)
ده حصل **12 مرة** — probability بينهم من 0.5023 لـ 0.5676، و**4 منهم كانوا
هجمات حقيقية** و8 سليمة.

عشان كده **القرار بيتاخد من `probability` مش من `is_attack`** — شوف §4.
ده بالظبط الفرق بين الطريقتين في الأرقام:

| | هجمات فايتة | false positives |
|---|---|---|
| `is_attack` (argmax) | 62 | 575 |
| `probability >= 0.5` | **58** | 583 |

الـ 4 هجمات الزيادة اللي الـ threshold بيمسكها، والـ 8 FP الزيادة، هم الـ 12 دول.
