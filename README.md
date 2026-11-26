<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:0f3d3e,100:00c896&height=200&section=header&text=MyGate%20AI&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Machine-learning%20intrusion%20detection%20for%20live%20network%20flows&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=800&color=00C896&center=true&vCenter=true&width=640&lines=Port+scan%3F+Brute+force%3F+DoS%3F;Random+Forest+%C2%B7+69+flow+features;Trained+on+1.7M+CIC-IDS2017+flows;Macro-F1+0.998+on+748K+held-out+flows" alt="Typing SVG"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-CIC--IDS2017-0f3d3e?style=for-the-badge)
![Version](https://img.shields.io/badge/release-v1.0.0-00c896?style=for-the-badge)

</div>

---

## 🛡️ What is this?

**MyGate** is a graduation-project network security appliance (IPS) built by a 5-person team:
Suricata + nftables + a FastAPI backend + PostgreSQL. **This repo is its AI layer.**

Every network conversation (a *flow*) is summarised into **69 statistical features** —
durations, packet sizes, inter-arrival times, flag counts — and a Random Forest decides
whether it is normal traffic or one of three attacks:

<div align="center">

| 🟢 `benign` | 🔍 `portscan` | 🔑 `brute_force` | 💥 `dos` |
|:---:|:---:|:---:|:---:|
| normal traffic | nmap-style probing | FTP / SSH password guessing | Hulk, GoldenEye, slowloris, DDoS |

</div>

## ⚙️ How it works

```mermaid
flowchart LR
    A[🌐 Live traffic<br/>on the LAN NIC] --> B[🧩 Flow extractor<br/>cicflowmeter + scapy]
    B -->|69 features| C[🌲 Random Forest<br/>mygate_ai.predict]
    C -->|label + probability| D{⚖️ Decision engine}
    D -->|p ≥ 0.5| E[🔔 Alert<br/>DB + Telegram]
    D -->|p ≥ 0.9| F[⛔ Block<br/>nftables]
    D -->|benign| G[✅ Allow]
```

The backend only ever touches one tiny contract:

```python
from mygate_ai import load, predict

load()                      # once, at startup — verifies sha256 + sklearn version
predict(flow_dict)
# {'is_attack': True, 'probability': 0.97, 'attack_type': 'portscan', 'model_version': 'v1'}
```

## 📊 Results (v1, CIC-IDS2017 held-out test set)

Trained on **1,746,923** flows, tested on **748,682** flows it never saw.

<div align="center">

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| benign | 0.9999 | 0.9991 | 0.9995 |
| brute_force | 1.0000 | 1.0000 | 1.0000 |
| dos | 0.9972 | 0.9994 | 0.9983 |
| portscan | 0.9887 | 0.9995 | 0.9941 |
| **macro avg** | | **0.9995** | **0.998** |

**Detection rate 99.95 %** · **False-positive rate 0.09 %** at the 0.5 alert threshold

</div>

<p align="center">
  <img src="reports/figures/03_confusion_matrix.png" width="48%"/>
  <img src="reports/figures/05_feature_importance.png" width="48%"/>
</p>

<details>
<summary><b>📈 More figures</b></summary>
<br/>
<p align="center">
  <img src="reports/figures/02_class_distribution.png" width="48%"/>
  <img src="reports/figures/04_per_class.png" width="48%"/>
  <img src="reports/figures/06_thresholds.png" width="48%"/>
  <img src="reports/figures/07_recall_per_raw_label.png" width="48%"/>
</p>
</details>

## 🧪 From the lab — honest status

> [!WARNING]
> The numbers above are **offline** results on CIC-IDS2017. When v1 ran **live** in our lab, a real
> `nmap -sS` scan was classified `benign`. The cause is a **feature-definition mismatch** between the
> Java CICFlowMeter that produced the dataset and the Python `cicflowmeter` used live (packet length,
> header length and SYN/RST counts are computed differently). Full analysis and fix plan:
> **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**.

## 🗺️ Roadmap

- [x] **v1.0** — Random Forest on CIC-IDS2017, packaged as a wheel with a frozen contract
- [x] Real-time flow extractor (seconds instead of a 240 s timeout) + alert de-duplication
- [ ] Java-compatible feature layer so live flows match the training data
- [ ] Live evaluation from a Kali attacker VM: portscan · brute force · DoS
- [ ] **v1.1** — retrain with lab traffic only if live evaluation still shows gaps

## 📁 Project structure

```
mygate_ai/   📦 shipped package — inference only (load · predict · schema · features)
src/         🔧 training pipeline — loaders · clean · train · evaluate · export
tests/       ✅ contract + schema tests
notebooks/   📓 EDA and model comparison (RF, XGBoost, LightGBM, LogReg, NN)
reports/     📊 evaluation figures
```

<details>
<summary><b>🛠️ Run it yourself</b></summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# put the CIC-IDS2017 CSVs in data/
python -m src.train              # full training  → mygate_ai/models/v1/
python -m src.evaluate --write   # re-evaluate an existing model
python -m pytest -v
python -m build --wheel          # → dist/mygate_ai-*.whl
```

Arabic developer guide: [docs/README_AR.md](docs/README_AR.md) · Backend integration: [INTEGRATION.md](INTEGRATION.md)
</details>

---

<div align="center">

Built by **[Shady Ashraf](https://github.com/shadyy-ashraff)** — AI layer of the MyGate graduation project · Benha University

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c896,50:0f3d3e,100:0d1117&height=100&section=footer" width="100%"/>

</div>
