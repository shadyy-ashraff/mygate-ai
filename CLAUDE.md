# MyGate AI — project context for Claude Code

> Reply to the user in **Egyptian Arabic**, keeping technical terms in English.
> The user (Shady, AI engineer on the team) is learning — explain *what* and *why* before each change.

## 1. What this project is
MyGate is a graduation project: a network **Security Appliance** (nftables + Suricata IDS +
FastAPI + PostgreSQL). The backend lives in a separate repo (`../MyGate-main(2)/MyGate-main`).
**This folder is only the AI detection model.** It classifies network **flows** (CICFlowMeter
features, L3/L4) — not HTTP payloads.

## 2. Goal — the deliverable
A single installable package, `mygate_ai/`, containing everything the backend needs:
```
mygate_ai/__init__.py            the public API — everything is imported from here
mygate_ai/inference.py           the contract (§3) + load() integrity checks
mygate_ai/schema.py              canonical names, COLUMN_MAPS, to_canonical()
mygate_ai/features.py            feature_list(), build_features(), ENGINEERED
mygate_ai/models/v1/model.pkl    one sklearn Pipeline (imputer + model), n_jobs=1
mygate_ai/models/v1/schema.json  ordered features, classes, label_map, versions, sha256, ...
mygate_ai/models/v1/metrics.json evaluation results (see §7)
```
Shipped as a wheel (`pyproject.toml`): `python -m build --wheel` → `dist/mygate_ai-*.whl`.
The model for `DEFAULT_MODEL_VERSION` only travels inside the wheel as package data.
The backend then writes `from mygate_ai import load, predict_many` — no paths, no `config.yaml`.
Integration notes for the team: `INTEGRATION.md`.

## 3. The contract with the backend team — FINAL, do not change
```python
predict(flow_dict) -> {
    "is_attack": bool,
    "probability": float,        # P(attack) = 1 - P(benign), 0..1
    "attack_type": "benign" | "portscan" | "brute_force" | "dos",
    "model_version": str,
}
```
- `flow_dict` = one raw flow from `cicflowmeter==0.2.0` (snake_case keys, e.g. `flow_duration`).
- Backend loads the model **once at startup** inside `core/controllers/ai_engine.py` and pins the
  **same scikit-learn version** as `requirements.txt`. In their DB: `ai_confidence = probability`,
  `ai_label = attack_type`. Block/allow decisions are theirs (`core/decision.py`), not ours.
- Out of scope for this model: SQLi / XSS / Web brute force / Heartbleed (→ Suricata),
  Bot / Infiltration (→ documented limitation).

## 4. Data
- `../archive/*.csv` — 8 CIC-IDS2017 MachineLearningCVE files, 2,830,743 rows × 79 cols
  (also `../archive.zip`). `config.yaml → sources.cic2017.path = "../archive"`; if that folder
  is missing (e.g. Colab) the loader falls back to `kagglehub` (`chethuhn/network-intrusion-dataset`).
- Known quirks: leading spaces in column names; `Inf` in `Flow Bytes/s` & `Flow Packets/s`
  (flow_duration = 0); broken characters in the Web Attack labels (dropped anyway); duplicate
  column `Fwd Header Length.1`; 8 zero-variance columns (listed in `config.yaml`).
- Class counts before cleaning: BENIGN 2,273,097 · dos 380,688 · portscan 158,930 ·
  brute_force 13,835 (≈9k after dedupe — the smallest class, watch it).

## 5. Layout & status
Two packages, imports flow **one way**: `src/ → mygate_ai/`. `mygate_ai/` knows nothing
about `src/` and uses relative imports internally, so it works standalone in the wheel.

**`mygate_ai/` — runtime, ships to the backend**
| File | Status |
|---|---|
| `__init__.py` | ✅ the only import surface: `load`, `predict`, `predict_many`, `ModelIntegrityError`, `SklearnVersionError`, `DEFAULT_MODEL_VERSION`, `DEFAULT_MODEL_DIR` |
| `schema.py` | ✅ canonical names = cicflowmeter live names; `COLUMN_MAPS`; `to_canonical()` |
| `features.py` | ✅ `feature_list()`, `build_features()`, `ENGINEERED`; missing feature → `NaN` (never `0.0`) |
| `inference.py` | ✅ contract §3; `load(model_dir=None, strict=True)` verifies sha256 **before** unpickling. Named `inference`, not `predict`, so `mygate_ai.predict` unambiguously means the function |

**`src/` — training only, never shipped**
| File | Status |
|---|---|
| `config.yaml` | ✅ sources, `label_map` (15→4), `drop_features`, model params, paths |
| `loaders.py` | ✅ `load_all(cfg, nrows=None, sample=None)` — `sample` = random rows/file |
| `clean.py` | ✅ `map_labels()`, `clean()`, `preview()`; runnable: `python -m src.clean` |
| `train.py` | ✅ `prepare()`, `split()`, `build_model()`, `train()`; CLI `--nrows/--no-cache/--force` |
| `evaluate.py` | ✅ `evaluate()`, `threshold_table()`, `evaluate_saved()`; CLI `python -m src.evaluate` |
| `export.py` | ✅ `export(model, cfg, metrics, info, force=False)` |
| `pyproject.toml` | ✅ builds the wheel; deps pinned to `schema.json` |
| `tests/test_schema.py` | ✅ passing |
| `tests/test_predict.py` | ✅ 25 tests — contract, edge cases, load() security, version pinning |

Keep the design of the ✅ files. Improve them only if needed, and say why.

## 6. Notebook → file mapping
Source: `intrusion-detection-system.ipynb` (original Kaggle notebook). **Do not edit it** — it is
the reference. Port logic from it, not code verbatim.

| Notebook cell | Goes to |
|---|---|
| 0, 1, 3 (imports / pip) | `requirements.txt` ✅ |
| 5 (load) + float32 downcast | `src/loaders.py` ✅ |
| 7, 10, 12, 14 (EDA plots) | stay in a notebook (`notebooks/01_eda.ipynb`, importing from `src/` + `mygate_ai/`) — for the report |
| 17 (cleaning) | `src/clean.py → clean()` ✅ |
| 19 (labels) | `src/clean.py → map_labels()` ✅ |
| 21 (split) | `src/train.py → split()` ✅ |
| 23 (scaling) | ❌ dropped — tree models don't need it |
| 29 (SMOTE) | ❌ not in v1 (see §8) |
| 39 / 41 (RF / XGBoost) | `src/train.py → build_model()` ✅ (RF only; xgboost not wired) |
| 37, 44, 52, 54 (LogReg / LightGBM / NN) | notebook-only comparison, optional |
| 46, 58, 60, 62, 68 (evaluation) | `src/evaluate.py` ✅ |
| 70, 72 (SHAP) | notebook, for the report |
| 74 (save) | `src/export.py` ✅ |
| — | `mygate_ai/inference.py` (new) ✅ |

## 7. Design rules
- **One source of truth:** `schema.to_canonical()` + `features.build_features()` are used by BOTH
  training and `mygate_ai.inference` → no skew inside our own code path.
- `model.pkl` = `Pipeline([("impute", SimpleImputer(strategy="median")), ("model", <from config.model>)])`.
  4-class model; `probability = 1 - P(benign)`; `attack_type = argmax class`.
- Pipeline order in `src/train.py`: `load_all → map_labels → clean → build_features → split → fit → evaluate → export`.
- Cache cleaned data to `config.paths.cache` (`data/clean.parquet`); reuse it if present.
- Split: stratified, `random_state` from config. Save the **golden test set** to
  `data/test_v1.parquet` and reuse it for every future version (fair comparison).
- `schema.json`: model_version, ordered features, classes, label_map, drop_features,
  sklearn version, trained_at, data sources, n_train / n_test.
- `metrics.json`: per-class precision/recall/f1, macro-F1, **binary FPR** (benign flagged as
  attack), binary detection rate, binary ROC-AUC, confusion matrix, inference latency (ms/flow).
  Model selection = recall per class + FPR — **not** weighted accuracy.
- `mygate_ai/inference.py`: `load()` once at startup (verifies sha256 before unpickling); `predict(flow)` / `predict_many(flows)` handle missing keys, extra keys, strings and `Inf` safely.
- CLI: `python -m src.train` (reads `config.yaml`, writes `models/<model_version>/`);
  `--nrows N` for a quick smoke test.
- Style: black + ruff (same as the backend), type hints, short docstrings.
- Don't delete/rename existing files without asking. Don't commit `data/` or `models/`.

## 8. Bugs in the notebook — must NOT be carried over
1. cuML (`cuLR`, `cuRF`, cells 35/37/39) → use scikit-learn (the VM has no GPU).
2. `'BENGIN'` typo (cell 19) → irrelevant now, labels come from `config.label_map`.
3. `scaler.fit_transform(X_test)` (cell 23) → no scaler at all.
4. Median imputation on the whole dataset before the split (cell 17) → `SimpleImputer` inside the Pipeline.
5. Automatic zero-variance drop → explicit `config.drop_features`.
6. 15-class `LabelEncoder` → 4 classes via `config.label_map`.
7. Choosing the best model by weighted accuracy/F1 → per-class recall + FPR + macro-F1.
8. Neural Network / `.h5` / TensorFlow → not in `src/` (heavy dependency on the VM).
9. SMOTE → not in v1. Use `class_weight="balanced_subsample"` instead (SMOTE was generating
   ~146k synthetic rows from 8 Heartbleed samples). Can be tried later as an experiment.
10. Cell 62 `.values[:, 0]` was cuML-specific.

## 9. Environment (Windows + VS Code)
- Python 3.11 or 3.12, virtual env `.venv` in this folder.
- `requirements.txt` = runtime (what the VM needs). Dev tools (ipykernel, matplotlib, seaborn,
  pytest, black, ruff) go in `requirements-dev.txt`.
- After installing, **pin the exact scikit-learn version** in `requirements.txt`.
- RAM: full data ≈ 830 MB after float32; RF on ~2.5M rows needs several GB → use `--nrows` first.

## 10. Working style
Work **one file at a time**: explain the plan → write → run → show the result → wait for OK.
