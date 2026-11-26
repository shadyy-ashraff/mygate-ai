"""
schema.py — الأسماء الموحّدة للـ features + خريطة أعمدة لكل مصدر.

الفكرة: كل مصدر بيسمّي الأعمدة بطريقته
    CIC-IDS2017 : "Flow Duration"
    CIC-IDS2018 : "Flow Duration" / "Dst Port" (مختصر)
    live (VM)   : "flow_duration"   ← cicflowmeter==0.2.0
فبنحوّل الكل لاسم واحد رسمي (canonical). اخترنا أسماء الـ live نفسها
كأسماء رسمية، لأن ده اللي الموديل هيشوفه في الـ production.

لإضافة داتاست جديدة: ضيف dict جديد في COLUMN_MAPS وخلاص.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------
# CIC-IDS2017 (الأسماء بعد .strip() للمسافات)  ->  canonical
# ---------------------------------------------------------------------
CIC2017 = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwr_flag_count",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Header Length.1": "fwd_header_len_dup",  # مكرر — بيتشال من config
    "Fwd Avg Bytes/Bulk": "fwd_byts_b_avg",
    "Fwd Avg Packets/Bulk": "fwd_pkts_b_avg",
    "Fwd Avg Bulk Rate": "fwd_blk_rate_avg",
    "Bwd Avg Bytes/Bulk": "bwd_byts_b_avg",
    "Bwd Avg Packets/Bulk": "bwd_pkts_b_avg",
    "Bwd Avg Bulk Rate": "bwd_blk_rate_avg",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
    "Label": "label",
}

# ---------------------------------------------------------------------
# live — cicflowmeter==0.2.0 على الـ VM. الأسماء هي نفسها الـ canonical.
# ---------------------------------------------------------------------
LIVE_META = ["src_ip", "dst_ip", "src_port", "protocol", "timestamp"]
CANONICAL_FEATURES = [
    v for v in CIC2017.values() if v not in ("label", "fwd_header_len_dup")
]
LIVE = {name: name for name in CANONICAL_FEATURES}

# ---------------------------------------------------------------------
# كل المصادر في مكان واحد
# ---------------------------------------------------------------------
COLUMN_MAPS = {
    "cic2017": CIC2017,
    "live": LIVE,
    # "cic2018": CIC2018,   # v2
}


def to_canonical(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """يحوّل أي DataFrame من أي مصدر للأسماء الموحّدة ويشيل الأعمدة الغريبة."""
    mapping = COLUMN_MAPS[source]
    df = df.copy()
    df.columns = df.columns.str.strip()  # مسافات CIC-IDS2017
    df = df.rename(columns=mapping)
    keep = [c for c in df.columns if c in set(mapping.values())]
    return df[keep]
