"""Summarize len256 single-operator (softmax + layernorm) benchmark into one table.

Reads operators/softmax_len256.json and operators/layernorm_len256.json and
writes operators/len256_single_op_summary.md comparing SHAFT baseline vs our
method on comm / rounds / LAN / WAN-4ms / WAN-80ms. comm & rounds are
shape-determined (layer-invariant); compute & network times are averaged over
the measured layers.
"""

import argparse
import json
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
    out = {k: statistics.mean(r[k] for r in picked) for k in keys}
    out["layers"] = sorted({r["layer"] for r in picked})
    return out


def pct(new, base):
    return (1.0 - new / base) * 100.0 if base else float("nan")


def fmt_row(op, label_shaft, shaft, label_ours, ours):
    comm_red = pct(ours["comm_mb"], shaft["comm_mb"])
    rnd_red = pct(ours["rounds"], shaft["rounds"])
    return (
        f"| {op} | {label_shaft} | {shaft['comm_mb']:.2f} | {shaft['rounds']:.0f} | "
        f"{shaft['lan_3g_0p5ms_time_s']*1e3:.1f} | {shaft['wan_400m_4ms_time_s']*1e3:.1f} | "
        f"{shaft['wan_100m_80ms_time_s']*1e3:.1f} | — | — |\n"
        f"| {op} | {label_ours} | {ours['comm_mb']:.2f} | {ours['rounds']:.0f} | "
        f"{ours['lan_3g_0p5ms_time_s']*1e3:.1f} | {ours['wan_400m_4ms_time_s']*1e3:.1f} | "
        f"{ours['wan_100m_80ms_time_s']*1e3:.1f} | {comm_red:.1f}% | {rnd_red:.1f}% |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operators-dir",
        default="artifacts/benchmark/len256_single_softmax_layernorm_20260615/operators",
    )
    args = parser.parse_args()
    base = args.operators_dir.rstrip("/")

    softmax = load(f"{base}/softmax_len256.json")
    layernorm = load(f"{base}/layernorm_len256.json")

    sm_shaft = agg(softmax, "ode_clip_i16")
    sm_ours = agg(softmax, "scaled_k2_i8")
    ln_shaft = agg(layernorm, "NR")
    ln_ours = agg(layernorm, "MLFormer")

    lines = []
    lines.append("# len256 单算子基准：SHAFT vs 我们的方法\n")
    lines.append(
        "单次 softmax / 单次 layernorm（不是整样本累加）。真实 BERT 激活，layers "
        f"{sm_shaft['layers']}。comm 与 rounds 由形状决定（与层无关、跨层一致）；"
        "compute 与三网时间为 3 层均值。网络：LAN 3Gbps/0.5ms、WAN-4ms 400Mbps/4ms、WAN-80ms 100Mbps/80ms。\n"
    )
    lines.append(
        "| op | method | comm_MB | rounds | LAN_ms | WAN4_ms | WAN80_ms | comm↓ | rounds↓ |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(fmt_row("softmax", "ode_clip_i16 (SHAFT)", sm_shaft, "scaled_k2_i8 (ours)", sm_ours))
    lines.append(fmt_row("layernorm", "NR (SHAFT)", ln_shaft, "MLFormer (ours)", ln_ours))
    lines.append("")
    lines.append("## softmax 全部 case（附录，3 层均值）\n")
    lines.append("| case | comm_MB | rounds | LAN_ms | WAN4_ms | WAN80_ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for case in ["ode_clip_i16", "scaled_k2_i8", "scaled_k2_i12", "scaled_k2_i16"]:
        if any(r["case"] == case for r in softmax):
            a = agg(softmax, case)
            lines.append(
                f"| {case} | {a['comm_mb']:.2f} | {a['rounds']:.0f} | "
                f"{a['lan_3g_0p5ms_time_s']*1e3:.1f} | {a['wan_400m_4ms_time_s']*1e3:.1f} | "
                f"{a['wan_100m_80ms_time_s']*1e3:.1f} |"
            )
    out_path = f"{base}/len256_single_op_summary.md"
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(open(out_path).read())


if __name__ == "__main__":
    main()
