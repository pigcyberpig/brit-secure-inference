#!/usr/bin/env python3
"""Aggregate the supernet benchmark campaign into CSVs + a JSON for plotting.

Reads the three suite trees under artifacts/benchmark/supernet_bench_20260619/
and the bert-base reference under blb_network_replay_20260617/, writing:
  aggregate/microbench.csv
  aggregate/bert_large.csv
  aggregate/gpt2.csv
  aggregate/bert_base_reference.csv
  aggregate/all_results.json

Also emits invariant checks (comm_bytes/rounds network-independent; throttle
took effect) to stdout.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT  # noqa: E402

SUPERNET_ROOT = QUEST_ROOT / "artifacts" / "benchmark" / "supernet_bench_20260619"
BLB_ROOT = QUEST_ROOT / "artifacts" / "benchmark" / "blb_network_replay_20260617"
PROFILE_ORDER = ["lan_3g_0p3ms", "blb_lan", "blb_wan1", "blb_wan2", "blb_wan3"]
PROFILE_LABEL = {
    "lan_3g_0p3ms": "LAN-3G/0.3ms",
    "blb_lan": "LAN/1G/0.3ms",
    "blb_wan1": "WAN1/400M/4ms",
    "blb_wan2": "WAN2/100M/4ms",
    "blb_wan3": "WAN3/100M/80ms",
}


def agg_microbench(root: Path) -> list[dict]:
    rows = []
    for backend in ("gpu", "cpu"):
        for profile in PROFILE_ORDER:
            for op in ("softmax", "layernorm"):
                # support both layouts: <b>/<p>/<op>/<op>.json and <b>/<p>/<op>.json
                j = root / "microbench" / backend / profile / op / f"{op}.json"
                if not j.exists():
                    j = root / "microbench" / backend / profile / f"{op}.json"
                if not j.exists():
                    continue
                data = json.loads(j.read_text())
                for r in data.get("results", []):
                    rows.append({
                        "profile": profile,
                        "profile_label": PROFILE_LABEL[profile],
                        "backend": backend,
                        "op": op,
                        "case": r["case"],
                        "comm_bytes": int(r["comm_bytes"]),
                        "comm_mb": round(r["comm_mb"], 4),
                        "rounds": int(r["rounds"]),
                        "compute_s": round(r["compute_time_s"], 6),
                        "comm_time_s": round(r.get("comm_time_s", 0.0), 6),
                        "wall_time_s": round(r.get("wall_time_s", 0.0), 6),
                    })
    return rows


def agg_bert_large(root: Path) -> list[dict]:
    rows = []
    for case_dir in sorted((root / "bert_large").glob("*/*/*")):
        if not case_dir.is_dir():
            continue
        backend = case_dir.parent.parent.name
        profile = case_dir.parent.name
        case = case_dir.name
        summaries = sorted(case_dir.glob("summary_*.json"))
        if not summaries:
            continue
        d = json.loads(summaries[0].read_text())
        pf = d.get("private_forward", {})
        rows.append({
            "profile": profile,
            "profile_label": PROFILE_LABEL.get(profile, profile),
            "backend": backend,
            "case": case,
            "running_time_s": round(d.get("running_time_s", 0.0), 3),
            "total_comm_bytes": int(pf.get("total_comm_bytes", 0)),
            "total_comm_gb": round(pf.get("total_comm_bytes", 0) / 1e9, 4),
            "total_comm_rounds": int(pf.get("total_comm_rounds", 0)),
            "total_comm_time_s": round(pf.get("total_comm_time_s", 0.0), 3),
            "softmax_comm_gb": round(pf.get("softmax_comm_bytes", 0) / 1e9, 4),
            "layernorm_comm_gb": round(pf.get("layernorm_comm_bytes", 0) / 1e9, 4),
            "gelu_comm_gb": round(pf.get("gelu_comm_bytes", 0) / 1e9, 4),
            "matmul_comm_gb": round(pf.get("matmul_comm_bytes", 0) / 1e9, 4),
            "layernorm_rounds": int(pf.get("layernorm_comm_rounds", 0)),
        })
    return rows


def agg_gpt2(root: Path) -> list[dict]:
    rows = []
    for case_dir in sorted((root / "gpt2").glob("*/*/*")):
        if not case_dir.is_dir():
            continue
        backend = case_dir.parent.parent.name
        profile = case_dir.parent.name
        case = case_dir.name
        j = case_dir / "summary.json"
        if not j.exists():
            continue
        d = json.loads(j.read_text())
        rows.append({
            "profile": profile,
            "profile_label": PROFILE_LABEL.get(profile, profile),
            "backend": backend,
            "case": case,
            "elapsed_s": round(d.get("elapsed_s", 0.0), 4),
            "bytes_per_party_mb": round(d.get("bytes_per_party_mb", 0.0), 4),
            "rounds": int(d.get("rounds", 0)),
        })
    return rows


def agg_bert_base_reference() -> list[dict]:
    """bert-base len128 from the 20260617 campaign (gpu 4 cases + cpu 2 cases)."""
    rows = []
    case_dirs = list(BLB_ROOT.glob("gpu/*/len128/*")) + list(BLB_ROOT.glob("cpu/*/len128/*"))
    for case_dir in sorted(case_dirs):
        if not case_dir.is_dir():
            continue
        # layout: BLB_ROOT/{backend}/{profile}/len128/{case}
        backend = case_dir.parent.parent.parent.name
        profile = case_dir.parent.parent.name
        case = case_dir.name
        summaries = sorted(case_dir.glob("summary_*.json"))
        if not summaries:
            continue
        d = json.loads(summaries[0].read_text())
        pf = d.get("private_forward", {})
        rows.append({
            "profile": profile,
            "profile_label": PROFILE_LABEL.get(profile, profile),
            "backend": backend,
            "case": case,
            "running_time_s": round(d.get("running_time_s", 0.0), 3),
            "total_comm_gb": round(pf.get("total_comm_bytes", 0) / 1e9, 4),
            "total_comm_rounds": int(pf.get("total_comm_rounds", 0)),
            "total_comm_time_s": round(pf.get("total_comm_time_s", 0.0), 3),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def invariant_checks(micro: list[dict], bert: list[dict], gpt2: list[dict]) -> list[str]:
    msgs = []
    # Microbench: comm_bytes & rounds must be network-independent per (op,case,backend).
    by_key = {}
    for r in micro:
        by_key.setdefault((r["op"], r["case"], r["backend"]), []).append(r)
    for key, items in by_key.items():
        comms = {r["comm_bytes"] for r in items}
        rnds = {r["rounds"] for r in items}
        if len(comms) > 1:
            msgs.append(f"WARN comm_bytes varies across networks for {key}: {comms}")
        if len(rnds) > 1:
            msgs.append(f"WARN rounds varies across networks for {key}: {rnds}")
        # throttle took effect: comm_time_s should grow lan->wan3
        times = {r["profile"]: r["comm_time_s"] for r in items}
        if "lan_3g_0p3ms" in times and "blb_wan3" in times:
            if times["blb_wan3"] <= times["lan_3g_0p3ms"]:
                msgs.append(f"WARN comm_time did not grow lan->wan3 for {key}: {times}")
    # BERT-large: softmax config took effect (shaft softmax_comm != both softmax_comm)
    sha = [r for r in bert if r["case"] == "shaft_original"]
    bot = [r for r in bert if r["case"] == "both_optimized"]
    if sha and bot:
        if abs(sha[0]["softmax_comm_gb"] - bot[0]["softmax_comm_gb"]) < 1e-6:
            msgs.append("WARN bert-large softmax_comm identical shaft vs both (config may not have applied)")
    if not msgs:
        msgs.append("invariant checks: OK")
    return msgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(SUPERNET_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    agg = root / "aggregate"

    micro = agg_microbench(root)
    bert = agg_bert_large(root)
    gpt2 = agg_gpt2(root)
    base = agg_bert_base_reference()

    write_csv(agg / "microbench.csv", micro)
    write_csv(agg / "bert_large.csv", bert)
    write_csv(agg / "gpt2.csv", gpt2)
    write_csv(agg / "bert_base_reference.csv", base)

    (agg / "all_results.json").write_text(json.dumps({
        "microbench": micro,
        "bert_large": bert,
        "gpt2": gpt2,
        "bert_base_reference": base,
    }, indent=2, ensure_ascii=False))

    print(f"microbench rows={len(micro)} (expect 60)")
    print(f"bert_large  rows={len(bert)} (expect 15)")
    print(f"gpt2        rows={len(gpt2)} (expect 20)")
    print(f"bert_base   rows={len(base)} (reference)")
    print("--- invariant checks ---")
    for m in invariant_checks(micro, bert, gpt2):
        print(m)
    print(f"\naggregate -> {agg}")


if __name__ == "__main__":
    main()
