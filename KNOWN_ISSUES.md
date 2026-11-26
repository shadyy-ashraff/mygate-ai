# Known issues — mygate-ai v1.0.0

> v1.0.0 محفوظة هنا زي ما اتسلّمت للتيم بالظبط (نفس الـ wheel، sha256 `7a9dfff7…3c39`).
> الموديل **نفسه** متقاس كويس على CIC-IDS2017 (macro-F1 = 0.998 على الـ golden test set)،
> لكن لما اتشغّل live في المعمل ظهرت المشاكل دي. التصليحات هتنزل في v1.1 / في ريبو MyGate.

| # | المشكلة | الخطورة | فين الحل |
|---|---|---|---|
| 1 | الموديل بيصنّف nmap SYN scan حقيقي إنه `benign` | **عالية** | MyGate — `fast_flow_session.py` (patch 8) |
| 2 | الدوكيومنتيشن بتقول `cicflowmeter==0.2.0` والصح `0.5.0` | **عالية** | v1.1 — `INTEGRATION.md` + `schema.json` |
| 3 | الوقت كان بالثواني بدل الـ microseconds | اتصلحت | MyGate — patch 7 |
| 4 | أول باكت بيتعد مرتين + RST بيقفل flow غلط | اتصلحت | MyGate — patch 7 |
| 5 | كل flow كان بيتخزن ويبعت alert حتى لو benign (فلود تيليجرام) | اتصلحت | MyGate — `ai_pipeline.py` patch 7 |

---

## 1. Live nmap SYN scan → `benign` (أعلى probability = 0.26)

**الاختبار:** `nmap -sS -p 1-100 192.168.10.1` من Windows host (192.168.10.2).
99 flow اتجمعوا صح (SYN رايح + RST راجع لكل port)، وكلهم اتصنفوا `benign`
بـ probability بين 0.05 و 0.26. Suricata كشف الـ scan، الـ AI لأ.

**السبب: feature mismatch، مش عيب في التدريب.** قارنّا flow من المعمل بـ 157,448
PortScan flow في `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` بنفس الشكل
(1 fwd + 1 bwd):

| Feature | CIC-IDS2017 (Java CICFlowMeter) | Live (Python cicflowmeter 0.5.0) | |
|---|---|---|---|
| `flow_duration` | 42–49 µs | 47.9 µs | ✅ |
| `flow_pkts_s` | ~41k–47k | 41,734 | ✅ |
| `init_fwd_win_byts` | 1024 (بصمة nmap) | 1024 | ✅ |
| `init_bwd_win_byts` | 0 | 0 | ✅ |
| `totlen_fwd_pkts` | 0 أو 2 | **60** | ❌ |
| `pkt_len_mean` | 2 – 3.3 | **57** | ❌ |
| `fwd_header_len` | 24 أو 40 | **20** | ❌ |
| `syn_flag_cnt` / `rst_flag_cnt` | **0 دايمًا** | **1** | ❌ |

يعني الـ traffic نفسه شبه الداتا جدًا، لكن الـ Python tool بيحسب كام feature بتعريف
مختلف عن الـ Java tool اللي عمل الداتا:

- **Packet length:** Java = حجم الـ payload بس، Python = حجم الباكت كله.
- **Header length:** Java = TCP header (`dataofs × 4`)، Python = IP header.
- **SYN/RST counts:** صفر في كل الداتا (bug معروف في CICFlowMeter الجافا)، فالموديل
  اتعلّم إن قيمتهم صفر دايمًا — وأي قيمة 1 بيشوفها شاذة.

**الحل المخطط:** طبقة توافق في `fast_flow_session.py → to_cic_units()` تحسب الـ features
دي بتعريف Java، من غير ما نغير الـ 69 feature ولا نعيد تدريب. بعدها نعيد الاختبار على
portscan + brute_force (hydra) + dos (hping3/slowloris) من Kali. **لو هجمة فضلت مش
بتتكشف بعد كده، ساعتها بس نعيد التدريب بداتا من المعمل (v1.1).**

## 2. `cicflowmeter==0.2.0` في الدوكيومنتيشن

`INTEGRATION.md` و `mygate_ai/inference.py` و `schema.json["cicflowmeter_version"]` بيقولوا
`0.2.0`. النسخة دي بتفرض `numpy<2`، فتسطيبها بيعمل downgrade لـ numpy من 2.5.3 لـ 1.26.4
ويكسر pins الموديل، والـ API بتاعها مختلف عن اللي `FastFlowSession` بيستخدمه.

**الصح:** `cicflowmeter==0.5.0` + `scapy==2.7.0` (اتجرب على Python 3.12 مع numpy 2.5.3 /
scikit-learn 1.6.1 / pandas 3.0.6 / scipy 1.18.1 → `pip check` نضيف).
`load()` مابيفحصش `cicflowmeter_version`، فالقيمة الغلط في `schema.json` مابتكسرش حاجة
لكنها مضللة.

## 3–5. اتصلحت في MyGate (فرع `fix/setup-paths`)

- **Units:** Python بيطلّع الوقت بالثواني، CIC بالـ µs → `to_cic_units()` بيحوّل
  `flow_duration` و الـ IAT و active/idle.
- **Flow pairing:** RST كان بيقفل flows تانية غير flow الباكت نفسه؛ وأول باكت كان بيتعد
  مرتين (`_SingleCountFlow`).
- **Alert flood:** `ai_pipeline.py` بقى بيتجاهل أي flow تحت `AI_ALERT_THRESHOLD` (0.5)،
  ومعاه cooldown 60 ثانية لكل `(src_ip, label)`.
