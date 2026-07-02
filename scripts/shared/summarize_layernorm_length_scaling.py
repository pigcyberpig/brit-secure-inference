"""Length-scaling summary for single-operator LayerNorm (NR=SHAFT vs MLFormer=ours).

Globs operators/layernorm_len<L>.json across the len{L}_single_softmax_layernorm_*
benchmark dirs and writes a per-op scaling table (comm / rounds / 3-net time)
alongside the e2e length_scaling.md.
"""

import argparse
import glob
import json
import os
import re
import statistics


def load(path):
    with open(path) as handle:
        return json.load(handle)["results"]


def agg(rows, case):
    picked = [r for r in rows if r["case"] == case]
    keys = (
        "comm_mb",
        "rounds",
        "compute_time_s",
        "lan_3g_0p5ms_time_s",
        "wan_400m_4ms_time_s",
        "wan_100m_80ms_time_s",
    )
    return {k: statistics.mean(r[k] for r in picked) for k in keys}


def pct(new, base):
    return (1.0 - new / base) * 100.0 if base else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        default="artifacts/benchmark",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/benchmark/length_scaling_softmax_layernorm_20260528",
    )
    args = parser.parse_args()

    paths = glob.glob(
        f"{args.benchmark_root}/len*_single_softmax_layernorm_*/operators/layernorm_len*.json"
    )
    by_len = {}
    for p in paths:
        m = re.search(r"len(\d+)_single", p)
        if not m:
            continue
        by_len[int(m.group(1))] = load(p)

    lines = []
    lines.append("# LayerNorm 单算子 Length-Scaling：NR (SHAFT) vs MLFormer (我们的方法)\n")
    lines.append(
        "单次 layernorm（非整样本累加），真实 BERT 残差流 hidden state `(L,768)`，layers [0,5,11] 均值。"
        "comm/rounds 由形状决定；compute/三网时间为 3 层均值。"
        "网络：LAN 3Gbps/0.5ms、WAN-4ms 400Mbps/4ms、WAN-80ms 100Mbps/80ms。\n"
    )
    lines.append(
        "| length | NR comm_MB | NR rounds | NR LAN_ms | NR WAN4_ms | NR WAN80_ms | "
        "MLF comm_MB | MLF rounds | MLF LAN_ms | MLF WAN4_ms | MLF WAN80_ms | comm↓ | rounds↓ |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for L in sorted(by_len):
        nr = agg(by_len[L], "NR")
        ml = agg(by_len[L], "MLFormer")
        lines.append(
            f"| {L} | {nr['comm_mb']:.2f} | {nr['rounds']:.0f} | "
            f"{nr['lan_3g_0p5ms_time_s']*1e3:.1f} | {nr['wan_400m_4ms_time_s']*1e3:.1f} | "
            f"{nr['wan_100m_80ms_time_s']*1e3:.1f} | {ml['comm_mb']:.2f} | {ml['rounds']:.0f} | "
            f"{ml['lan_3g_0p5ms_time_s']*1e3:.1f} | {ml['wan_400m_4ms_time_s']*1e3:.1f} | "
            f"{ml['wan_100m_80ms_time_s']*1e3:.1f} | {pct(ml['comm_mb'], nr['comm_mb']):.1f}% | "
            f"{pct(ml['rounds'], nr['rounds']):.1f}% |"
        )
    lines.append("")
    lines.append(
        "解读：layernorm 单算子的轮数缩减（NR 26 → MLFormer 5，~80.8%）在所有长度上稳定；"
        "通信量近乎相等（MLFormer 仅低 ~1%，因 layernorm 通信由方差 matmul 主导、与 sqrt 方法几乎无关），"
        "随序列长度线性增长（32→256：1.2→9.6 MB）。在 WAN-80ms 等高延迟网络下，轮数缩减带来 ~1.9x 单算子加速。"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/layernorm_perop_scaling.md"
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(open(out_path).read())


if __name__ == "__main__":
    main()
