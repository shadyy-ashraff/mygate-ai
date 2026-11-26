"""
tests/test_predict.py — العقد (§3) + أمان load() + تطابق مسار الكود.

الـ tests اللي محتاجة model.pkl أو الـ golden set بتعمل skip لو مش موجودين،
عشان pytest يعدّي على أي جهاز من غير ما حد يدرّب الأول.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

import mygate_ai
from mygate_ai import (
    DEFAULT_MODEL_DIR,
    ModelIntegrityError,
    SklearnVersionError,
    load,
    predict,
    predict_many,
)

# الموديول نفسه — عشان نوصل للحالة الداخلية (_model, _features) في الـ tests.
# التيم بيستورد من mygate_ai مباشرةً؛ ده وصول داخلي للاختبار بس.
from mygate_ai import inference as pmod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "data", "test_v1.parquet")
BENIGN = "benign"
N_ROWS = 1000
SEED = 12345

needs_model = pytest.mark.skipif(
    not os.path.isfile(os.path.join(DEFAULT_MODEL_DIR, "model.pkl")),
    reason=f"مفيش model.pkl في {DEFAULT_MODEL_DIR} — شغّل python -m src.train",
)
needs_golden = pytest.mark.skipif(
    not os.path.isfile(GOLDEN),
    reason=f"مفيش {GOLDEN} — شغّل python -m src.train",
)


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded():
    """الموديل محمّل مرة واحدة لكل الـ module."""
    load()
    return pmod._model


@pytest.fixture(scope="module")
def sample(loaded):
    """1000 صف عشوائي (seed ثابت) من الـ golden set + الـ X المقابلة.

    بيرجّع (flows, X) — flows بأسماء cicflowmeter زي ما بتوصل من الـ VM،
    و X هي نفس الصفوف بالشكل اللي الـ Pipeline بياخده.
    """
    import pandas as pd
    import yaml

    from mygate_ai.features import build_features

    df = pd.read_parquet(GOLDEN).sample(n=N_ROWS, random_state=SEED)
    with open(os.path.join(REPO, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def _py(v):
        """numpy scalar → Python scalar؛ NaN → None (زي مفتاح ناقص)."""
        if pd.isna(v):
            return None
        return v.item() if hasattr(v, "item") else v

    feats = pmod._features
    X = build_features(df, cfg)[feats].astype("float32")
    flows = [{c: _py(v) for c, v in row.items()} for _, row in df[feats].iterrows()]
    return flows, X


@pytest.fixture
def model_copy(tmp_path):
    """نسخة معزولة من models/v1 — كل test أمان بيعبث في نسخته."""
    dst = tmp_path / "model_dir"
    dst.mkdir()
    for f in ("model.pkl", "schema.json"):
        shutil.copy(os.path.join(DEFAULT_MODEL_DIR, f), dst)
    return dst


def _read_schema(d):
    with open(os.path.join(d, "schema.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _write_schema(d, s):
    with open(os.path.join(d, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh)


# ---------------------------------------------------------------------
# 1) تطابق مسار الكود بين التدريب والـ inference
# ---------------------------------------------------------------------
@needs_model
@needs_golden
def test_consistency_with_pipeline(loaded, sample):
    """predict_many على dicts == الـ Pipeline على نفس الصفوف.

    **بيثبت إيه بالظبط:** إن مسار الـ inference
    (dict → to_canonical → build_features → predict_proba) بيدّي نفس
    الأرقام زي ما لو عدّينا الـ DataFrame للـ Pipeline مباشرةً. يعني
    مفيش عمود اتقلب، ولا feature اتحسبت بطريقة تانية، ولا ترتيب اتغيّر
    في الكود بتاعنا.

    **مش بيثبت إيه:** إن مفيش training/serving skew بالمعنى الكامل.
    الصفوف دي جاية من CSV طلّعه **CICFlowMeter (Java)**، بينما اللايف
    جاي من **cicflowmeter==0.2.0 (Python)**. الأداتين ممكن يحسبوا نفس
    الـ feature بشكل مختلف (وحدات flow_duration، الـ flow timeout،
    init_win_byts، ...). التحقّق من ده محتاج lab CSV من تيم النيتورك —
    شوف §6 في INTEGRATION.md.
    """
    flows, X = sample
    model = loaded
    classes = [str(c) for c in model.classes_]
    bi = classes.index(BENIGN)

    proba = model.predict_proba(X)
    expected_p = 1.0 - proba[:, bi]
    expected_type = [classes[i] for i in proba.argmax(axis=1)]

    results = predict_many(flows)
    assert len(results) == N_ROWS

    got_p = np.array([r["probability"] for r in results])
    np.testing.assert_allclose(got_p, expected_p, atol=1e-6)

    got_type = [r["attack_type"] for r in results]
    assert got_type == expected_type


@needs_model
@needs_golden
def test_is_attack_matches_attack_type(loaded, sample):
    for r in predict_many(sample[0]):
        assert r["is_attack"] is (r["attack_type"] != BENIGN)


@needs_model
@needs_golden
def test_is_attack_true_implies_probability_at_least_half(loaded, sample):
    """الاتجاه المستحيل (§8 في INTEGRATION.md): is_attack=True → p >= 0.5."""
    for r in predict_many(sample[0]):
        if r["is_attack"]:
            assert r["probability"] >= 0.5


@needs_model
@needs_golden
def test_predict_matches_predict_many(loaded, sample):
    flows = sample[0][:20]
    batch = predict_many(flows)
    for flow, expected in zip(flows, batch):
        assert predict(flow) == expected


# ---------------------------------------------------------------------
# 2) العقد: الأنواع native مش numpy (psycopg2)
# ---------------------------------------------------------------------
@needs_model
@needs_golden
def test_return_types_are_native(loaded, sample):
    r = predict(sample[0][0])
    assert set(r) == {"is_attack", "probability", "attack_type", "model_version"}
    assert type(r["is_attack"]) is bool
    assert type(r["probability"]) is float
    assert type(r["attack_type"]) is str
    assert type(r["model_version"]) is str
    for v in r.values():
        assert not isinstance(v, np.generic)


@needs_model
@needs_golden
def test_contract_value_ranges(loaded, sample):
    for r in predict_many(sample[0][:100]):
        assert 0.0 <= r["probability"] <= 1.0
        assert r["attack_type"] in {"benign", "portscan", "brute_force", "dos"}
        assert r["model_version"] == mygate_ai.DEFAULT_MODEL_VERSION


# ---------------------------------------------------------------------
# 3) الحالات الحدّية
# ---------------------------------------------------------------------
@needs_model
def test_empty_batch_returns_empty_list(loaded):
    assert predict_many([]) == []


@needs_model
@needs_golden
def test_order_is_preserved(loaded, sample):
    flows = sample[0][:50]
    one_by_one = [predict(f)["attack_type"] for f in flows]
    assert [r["attack_type"] for r in predict_many(flows)] == one_by_one


@needs_model
@needs_golden
def test_inf_is_handled(loaded, sample):
    flow = {**sample[0][0], "flow_byts_s": float("inf"), "flow_pkts_s": float("-inf")}
    r = predict(flow)
    assert 0.0 <= r["probability"] <= 1.0


@needs_model
@needs_golden
def test_missing_keys_are_handled(loaded, sample):
    flow = {k: sample[0][0][k] for k in list(sample[0][0])[:3]}
    r = predict(flow)
    assert 0.0 <= r["probability"] <= 1.0


@needs_model
@needs_golden
def test_extra_keys_are_ignored(loaded, sample):
    """src_ip / timestamp / ... مابيغيّروش النتيجة."""
    flow = sample[0][0]
    noisy = {
        **flow,
        "src_ip": "10.0.0.7",
        "dst_ip": "192.168.1.1",
        "src_port": 51234,
        "timestamp": "2026-09-24 04:00:00",
        "protocol": 6,
        "something_we_never_saw": "x",
    }
    assert predict(noisy) == predict(flow)


@needs_model
@needs_golden
def test_string_values_give_same_result(loaded, sample):
    """flow عدّى بـ JSON/HTTP فقيمه بقت strings."""
    flow = sample[0][0]
    as_text = {k: ("" if v is None else str(v)) for k, v in flow.items()}
    assert predict(as_text) == predict(flow)


@needs_model
def test_empty_dict_does_not_raise(loaded):
    r = predict({})
    assert 0.0 <= r["probability"] <= 1.0


@needs_model
@needs_golden
def test_unparsable_value_becomes_nan(loaded, sample):
    flow = {**sample[0][0], "flow_duration": "مش رقم"}
    r = predict(flow)
    assert 0.0 <= r["probability"] <= 1.0


# ---------------------------------------------------------------------
# 4) predict قبل load
# ---------------------------------------------------------------------
def test_predict_before_load_raises(monkeypatch):
    monkeypatch.setattr(pmod, "_model", None)
    with pytest.raises(RuntimeError, match="load"):
        predict({"flow_duration": 1})
    with pytest.raises(RuntimeError, match="load"):
        predict_many([{"flow_duration": 1}])


# ---------------------------------------------------------------------
# 5) أمان load() — كل حالة في tmp_path لوحدها
# ---------------------------------------------------------------------
@needs_model
def test_tampered_model_raises_and_never_unpickles(model_copy, monkeypatch):
    """sha256 غلط → ModelIntegrityError، و joblib.load ماتنداش خالص.

    ده جوهر الفحص: الـ pickle بيشغّل كود وقت الفك، فلو الملف متبدّل
    لازم نرفض **قبل** ما نفكّه، مش بعده.
    """
    import joblib

    with open(model_copy / "model.pkl", "ab") as fh:
        fh.write(b"\x00")

    calls = []
    real_load = joblib.load
    monkeypatch.setattr(
        joblib, "load", lambda *a, **k: (calls.append(a), real_load(*a, **k))[1]
    )

    with pytest.raises(ModelIntegrityError):
        load(str(model_copy))
    assert calls == [], "joblib.load اتنادت على ملف مش موثوق!"


@needs_model
def test_tampered_model_with_strict_false_warns_and_loads(model_copy, capsys):
    with open(model_copy / "model.pkl", "ab") as fh:
        fh.write(b"\x00")

    load(str(model_copy), strict=False)
    out = capsys.readouterr().out
    assert "strict=False" in out and "sha256" in out
    assert pmod._model is not None


@needs_model
def test_sklearn_version_mismatch_raises(model_copy):
    """sha256 **صح** + نسخة sklearn مختلفة → SklearnVersionError لوحده.

    الملف مالوش أي تعديل، فالفحص الأول بيعدّي والتاني هو اللي بيمسك.
    """
    s = _read_schema(model_copy)
    s["sklearn_version"] = "0.0.0-not-a-real-version"
    _write_schema(model_copy, s)

    with pytest.raises(SklearnVersionError, match="0.0.0-not-a-real-version"):
        load(str(model_copy))


@needs_model
def test_sklearn_version_mismatch_with_strict_false_loads(model_copy, capsys):
    s = _read_schema(model_copy)
    s["sklearn_version"] = "0.0.0-not-a-real-version"
    _write_schema(model_copy, s)

    load(str(model_copy), strict=False)
    out = capsys.readouterr().out
    assert "strict=False" in out and "scikit-learn" in out


@needs_model
def test_missing_sha256_raises_in_strict_mode(model_copy):
    s = _read_schema(model_copy)
    del s["model_file_sha256"]
    _write_schema(model_copy, s)

    with pytest.raises(ModelIntegrityError, match="model_file_sha256"):
        load(str(model_copy))


def test_missing_schema_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="schema.json"):
        load(str(tmp_path))


@needs_model
def test_missing_model_file_raises(model_copy):
    os.remove(model_copy / "model.pkl")
    with pytest.raises(FileNotFoundError, match="model.pkl"):
        load(str(model_copy))


@needs_model
def test_reload_restores_good_model(model_copy):
    """بعد ما الـ tests عبثت، load() العادية لازم ترجّع النسخة السليمة."""
    load()
    assert pmod._model is not None
    assert pmod._model_version == mygate_ai.DEFAULT_MODEL_VERSION


# ---------------------------------------------------------------------
# 6) النسخ: pyproject.toml == schema.json
# ---------------------------------------------------------------------
@needs_model
def test_pinned_versions_match_schema():
    """أي فرق هنا معناه إن الـ wheel هيثبّت نسخة غير اللي الموديل اتصدّر بيها."""
    tomllib = pytest.importorskip("tomllib")
    path = os.path.join(REPO, "pyproject.toml")
    if not os.path.isfile(path):
        pytest.skip("مفيش pyproject.toml")

    with open(path, "rb") as fh:
        project = tomllib.load(fh)["project"]
    pinned = dict(d.split("==") for d in project["dependencies"] if "==" in d)

    schema = _read_schema(DEFAULT_MODEL_DIR)
    for dep, key in (
        ("scikit-learn", "sklearn_version"),
        ("pandas", "pandas_version"),
        ("numpy", "numpy_version"),
    ):
        assert pinned[dep] == schema[key], (
            f"pyproject يثبّت {dep}=={pinned[dep]} بينما الموديل اتصدّر بـ "
            f"{schema[key]}"
        )

    major_minor = ".".join(schema["python_version"].split(".")[:2])
    assert major_minor in project["requires-python"]


@needs_model
def test_schema_features_match_fitted_model(loaded):
    """الترتيب في schema.json == اللي الـ imputer اتدرّب عليه."""
    schema = _read_schema(DEFAULT_MODEL_DIR)
    fitted = [str(c) for c in loaded.named_steps["impute"].feature_names_in_]
    assert schema["features"] == fitted
    assert schema["n_features"] == len(fitted)
    assert schema["classes"] == [str(c) for c in loaded.classes_]
