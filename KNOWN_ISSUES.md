# Known issues — mygate-ai v1.0.0

> v1.0.0 is kept here exactly as it was delivered to the team (same wheel, sha256 `7a9dfff7…3c39`).
> The model **itself** scores well on CIC-IDS2017 (macro-F1 = 0.998 on the golden test set),
> but running it live in the lab surfaced the issues below. Fixes will land in v1.1 / in the MyGate repo.

| # | Issue | Severity | Where it gets fixed |
|---|---|---|---|
| 1 | The model classifies a real nmap SYN scan as `benign` | **High** | MyGate — `fast_flow_session.py` (patch 8) |
| 2 | The docs say `cicflowmeter==0.2.0`; the correct version is `0.5.0` | **High** | v1.1 — `INTEGRATION.md` + `schema.json` |
| 3 | Time features were in seconds instead of microseconds | Fixed | MyGate — patch 7 |
| 4 | The first packet was counted twice + RST closed the wrong flow | Fixed | MyGate — patch 7 |
| 5 | Every flow was stored and alerted even when benign (Telegram flood) | Fixed | MyGate — `ai_pipeline.py`, patch 7 |

---

## 1. Live nmap SYN scan → `benign` (highest probability = 0.26)

**Test:** `nmap -sS -p 1-100 192.168.10.1` from the Windows host (192.168.10.2).
99 flows were assembled correctly (one outgoing SYN + one returning RST per port), and all of them
were classified `benign` with probabilities between 0.05 and 0.26. Suricata detected the scan; the AI did not.

**Cause: a feature mismatch, not a training flaw.** We compared one lab flow with the 157,448
PortScan flows of the same shape (1 fwd + 1 bwd packet) in
`Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`:

| Feature | CIC-IDS2017 (Java CICFlowMeter) | Live (Python cicflowmeter 0.5.0) | |
|---|---|---|---|
| `flow_duration` | 42–49 µs | 47.9 µs | ✅ |
| `flow_pkts_s` | ~41k–47k | 41,734 | ✅ |
| `init_fwd_win_byts` | 1024 (nmap's signature) | 1024 | ✅ |
| `init_bwd_win_byts` | 0 | 0 | ✅ |
| `totlen_fwd_pkts` | 0 or 2 | **60** | ❌ |
| `pkt_len_mean` | 2 – 3.3 | **57** | ❌ |
| `fwd_header_len` | 24 or 40 | **20** | ❌ |
| `syn_flag_cnt` / `rst_flag_cnt` | **always 0** | **1** | ❌ |

So the traffic itself looks very much like the training data, but the Python tool computes several
features with a different definition from the Java tool that produced the dataset:

- **Packet length:** Java = payload size only; Python = the whole packet.
- **Header length:** Java = TCP header (`dataofs × 4`); Python = IP header.
- **SYN/RST counts:** zero across the whole dataset (a known bug in the Java CICFlowMeter), so the model
  learned that they are always zero — any value of 1 looks anomalous to it.

**Planned fix:** a compatibility layer in `fast_flow_session.py → to_cic_units()` that computes these
features with the Java definitions, without changing the 69 features and without retraining. Then
re-test portscan + brute_force (hydra) + dos (hping3/slowloris) from Kali. **Only if an attack is still
missed after that do we retrain with lab data (v1.1).**

## 2. `cicflowmeter==0.2.0` in the docs

`INTEGRATION.md`, `mygate_ai/inference.py` (v1.0.0) and `schema.json["cicflowmeter_version"]` say
`0.2.0`. That version requires `numpy<2`, so installing it downgrades numpy from 2.5.3 to 1.26.4 and
breaks the model's pins, and its API differs from the one `FastFlowSession` uses.

**Correct:** `cicflowmeter==0.5.0` + `scapy==2.7.0` (tested on Python 3.12 with numpy 2.5.3 /
scikit-learn 1.6.1 / pandas 3.0.6 / scipy 1.18.1 → `pip check` is clean).
`load()` does not check `cicflowmeter_version`, so the wrong value in `schema.json` breaks nothing,
but it is misleading.

## 3–5. Fixed in MyGate (branch `fix/setup-paths`)

- **Units:** Python outputs time in seconds, CIC uses µs → `to_cic_units()` converts
  `flow_duration`, the IAT features and active/idle.
- **Flow pairing:** an RST used to close flows other than its own; and the first packet was counted
  twice (`_SingleCountFlow`).
- **Alert flood:** `ai_pipeline.py` now ignores any flow below `AI_ALERT_THRESHOLD` (0.5),
  with a 60-second cooldown per `(src_ip, label)`.
