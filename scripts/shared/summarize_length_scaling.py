#!/usr/bin/env python3
"""Build a compact length-scaling comparison from len32/64/128 metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


LENGTHS = (32, 64, 128, 256)
BASE = Path("artifacts/benchmark")


def fmt(value: float) -> str:
    return f"{value:.2f}"


def fmt_opt(value: float | None) -> str:
    return "" if value is None else fmt(value)


def load_rows(length: int) -> dict:
    matches = sorted(BASE.glob(f"len{length}_single_softmax_layernorm_*/metrics.json"))
    if not matches:
        raise FileNotFoundError(f"no metrics.json for len{length} under {BASE}")
    data = json.loads(matches[-1].read_text())
    return {row["name"]: row for row in data["end_to_end"]}


def main() -> None:
    rows = []
    for length in LENGTHS:
        by_name = load_rows(length)
        base = by_name["shaft_original"]
        both = by_name["both_optimized"]
        soft = by_name["softmax_only"]
        ln = by_name["layernorm_only"]
        i16_both = by_name.get("i16_both_optimized")
        rows.append(
            {
                "length": length,
                "baseline_comm_gib": base["total_comm_gib"],
                "both_comm_gib": both["total_comm_gib"],
                "comm_reduction_pct": (1 - both["total_comm_bytes"] / base["total_comm_bytes"]) * 100,
                "baseline_rounds": base["total_comm_rounds"],
                "both_rounds": both["total_comm_rounds"],
                "round_reduction_pct": (1 - both["total_comm_rounds"] / base["total_comm_rounds"]) * 100,
                "lan_baseline_s": base["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"],
                "lan_both_s": both["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"],
                "lan_speedup": base["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"]
                / both["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"],
                "wan_mid_baseline_s": base["network_estimates"]["wan_400m_4ms"]["estimated_time_s"],
                "wan_mid_both_s": both["network_estimates"]["wan_400m_4ms"]["estimated_time_s"],
                "wan_mid_speedup": base["network_estimates"]["wan_400m_4ms"]["estimated_time_s"]
                / both["network_estimates"]["wan_400m_4ms"]["estimated_time_s"],
                "wan_hard_baseline_s": base["network_estimates"]["wan_100m_80ms"]["estimated_time_s"],
                "wan_hard_both_s": both["network_estimates"]["wan_100m_80ms"]["estimated_time_s"],
                "wan_hard_speedup": base["network_estimates"]["wan_100m_80ms"]["estimated_time_s"]
                / both["network_estimates"]["wan_100m_80ms"]["estimated_time_s"],
                "softmax_comm_reduction_pct": (
                    1 - soft["softmax_comm_bytes"] / base["softmax_comm_bytes"]
                )
                * 100,
                "layernorm_round_reduction_pct": (
                    1 - ln["layernorm_comm_rounds"] / base["layernorm_comm_rounds"]
                )
                * 100,
                "k2i16_both_comm_gib": i16_both["total_comm_gib"] if i16_both else None,
                "k2i16_both_rounds": i16_both["total_comm_rounds"] if i16_both else None,
                "k2i16_lan_speedup": (
                    base["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"]
                    / i16_both["network_estimates"]["lan_3g_0p5ms"]["estimated_time_s"]
                    if i16_both
                    else None
                ),
                "k2i16_wan_mid_speedup": (
                    base["network_estimates"]["wan_400m_4ms"]["estimated_time_s"]
                    / i16_both["network_estimates"]["wan_400m_4ms"]["estimated_time_s"]
                    if i16_both
                    else None
                ),
                "k2i16_wan_hard_speedup": (
                    base["network_estimates"]["wan_100m_80ms"]["estimated_time_s"]
                    / i16_both["network_estimates"]["wan_100m_80ms"]["estimated_time_s"]
                    if i16_both
                    else None
                ),
                "k2i16_vs_k2i8_comm_ratio": (
                    i16_both["total_comm_bytes"] / both["total_comm_bytes"] if i16_both else None
                ),
                "k2i16_vs_k2i8_round_ratio": (
                    i16_both["total_comm_rounds"] / both["total_comm_rounds"] if i16_both else None
                ),
            }
        )

    out_dir = BASE / "length_scaling_softmax_layernorm_20260528"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "length_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Length Scaling: Softmax + LayerNorm",
        "",
        "| length | comm baseline -> both | rounds baseline -> both | LAN speedup | WAN-4ms speedup | WAN-80ms speedup | softmax comm reduction | LayerNorm round reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["length"]),
                    f"{fmt(row['baseline_comm_gib'])} -> {fmt(row['both_comm_gib'])} GiB ({fmt(row['comm_reduction_pct'])}%)",
                    f"{row['baseline_rounds']} -> {row['both_rounds']} ({fmt(row['round_reduction_pct'])}%)",
                    f"{fmt(row['lan_speedup'])}x",
                    f"{fmt(row['wan_mid_speedup'])}x",
                    f"{fmt(row['wan_hard_speedup'])}x",
                    f"{fmt(row['softmax_comm_reduction_pct'])}%",
                    f"{fmt(row['layernorm_round_reduction_pct'])}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: as padded sequence length increases, the optimized softmax covers a larger share of total communication, so end-to-end speedup rises. LayerNorm round reduction is stable across lengths, but its absolute impact is smaller when byte-heavy softmax/matmul costs dominate the network estimate.",
        ]
    )
    i16_rows = [row for row in rows if row["k2i16_both_comm_gib"] is not None]
    if i16_rows:
        lines.extend(
            [
                "",
                "## k2i16 Appendix",
                "",
                "| length | k2i8 both comm/rounds | k2i16 both comm/rounds | k2i16 vs k2i8 comm | k2i16 vs k2i8 rounds | k2i16 LAN speedup | k2i16 WAN-4ms speedup | k2i16 WAN-80ms speedup |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in i16_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["length"]),
                        f"{fmt(row['both_comm_gib'])} GiB / {row['both_rounds']}",
                        f"{fmt_opt(row['k2i16_both_comm_gib'])} GiB / {row['k2i16_both_rounds']}",
                        f"{fmt_opt(row['k2i16_vs_k2i8_comm_ratio'])}x",
                        f"{fmt_opt(row['k2i16_vs_k2i8_round_ratio'])}x",
                        f"{fmt_opt(row['k2i16_lan_speedup'])}x",
                        f"{fmt_opt(row['k2i16_wan_mid_speedup'])}x",
                        f"{fmt_opt(row['k2i16_wan_hard_speedup'])}x",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Interpretation: k2i16 is still better than the original ODE softmax, but it is consistently heavier than k2i8 in both communication bytes and rounds for the measured lengths.",
            ]
        )
    (out_dir / "length_scaling.md").write_text("\n".join(lines))
    print(out_dir / "length_scaling.md")


if __name__ == "__main__":
    main()
