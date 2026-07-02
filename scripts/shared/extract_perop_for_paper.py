"""Extract single-operator per-op numbers in the paper Table 3/4 format.

Prints, per length, SHAFT vs Ours: comm (GiB) + LAN / WAN-1 / WAN-2 (s),
averaged over layers [0,5,11]. Network model matches the existing tables
(LAN 3Gbps/0.5ms, WAN-1 400Mbps/4ms, WAN-2 100Mbps/80ms). comm in GiB = bytes/2^30.
"""

import glob
import json
import re
import statistics

GIB = 2 ** 30


def load_per_length(kind):
    """kind in {'softmax','layernorm'} -> {length: [rows]}."""
    out = {}
    for p in glob.glob(f"artifacts/benchmark/len*_single_softmax_layernorm_*/operators/{kind}_len*.json"):
        m = re.search(r"len(\d+)_single", p)
        if not m:
            continue
        out[int(m.group(1))] = json.load(open(p))["results"]
    return out


def agg(rows, case):
    picked = [r for r in rows if r["case"] == case]
    if not picked:
        return None
    return {
        "gib": statistics.mean(r["comm_bytes"] for r in picked) / GIB,
        "lan": statistics.mean(r["lan_3g_0p5ms_time_s"] for r in picked),
        "wan1": statistics.mean(r["wan_400m_4ms_time_s"] for r in picked),
        "wan2": statistics.mean(r["wan_100m_80ms_time_s"] for r in picked),
        "rounds": max(r["rounds"] for r in picked),
    }


soft = load_per_length("softmax")
ln = load_per_length("layernorm")

print("=== Table 3 (DPSS Softmax, single operator) ===")
print("| len | method | comm_GiB | LAN_s | WAN-1_s | WAN-2_s | rounds |")
for L in sorted(soft):
    shaft = agg(soft[L], "ode_clip_i16")
    ours = agg(soft[L], "scaled_k2_i8")
    if shaft:
        print(f"| {L} | SHAFT | {shaft['gib']:.3f} | {shaft['lan']:.2f} | {shaft['wan1']:.2f} | {shaft['wan2']:.2f} | {shaft['rounds']:.0f} |")
    if ours:
        print(f"| {L} | Ours  | {ours['gib']:.3f} | {ours['lan']:.2f} | {ours['wan1']:.2f} | {ours['wan2']:.2f} | {ours['rounds']:.0f} |")

print("\n=== Table 4 (MaskInv-LN, single operator) ===")
print("| len | method | comm_GiB | LAN_s | WAN-1_s | WAN-2_s | rounds |")
for L in sorted(ln):
    shaft = agg(ln[L], "NR")
    ours = agg(ln[L], "MLFormer")
    if shaft:
        print(f"| {L} | SHAFT | {shaft['gib']:.4f} | {shaft['lan']:.3f} | {shaft['wan1']:.3f} | {shaft['wan2']:.3f} | {shaft['rounds']:.0f} |")
    if ours:
        print(f"| {L} | Ours  | {ours['gib']:.4f} | {ours['lan']:.3f} | {ours['wan1']:.3f} | {ours['wan2']:.3f} | {ours['rounds']:.0f} |")
