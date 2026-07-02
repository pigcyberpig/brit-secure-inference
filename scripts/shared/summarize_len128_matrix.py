#!/usr/bin/env python3
"""Summarize single-sample softmax + LayerNorm communication runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

from estimate_network_profiles import NETWORK_PROFILES, estimate_all


RUNS = OrderedDict(
    [
        ("shaft_original", ("ode_clip_i16", "NR")),
        ("softmax_only", ("scaled_k2_i8", "NR")),
        ("layernorm_only", ("ode_clip_i16", "MLFormer")),
        ("both_optimized", ("scaled_k2_i8", "MLFormer")),
    ]
)

OPTIONAL_RUNS = OrderedDict(
    [
        ("softmax_i12_only", ("scaled_k2_i12", "NR")),
        ("i12_both_optimized", ("scaled_k2_i12", "MLFormer")),
        ("softmax_i16_only", ("scaled_k2_i16", "NR")),
        ("i16_both_optimized", ("scaled_k2_i16", "MLFormer")),
    ]
)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def find_summary(run_dir: Path, softmax_config: str) -> Path:
    path = run_dir / f"summary_{softmax_config}.json"
    if path.exists():
        return path
    matches = sorted(run_dir.glob("summary_*.json"))
    if not matches:
        raise FileNotFoundError(f"no summary JSON in {run_dir}")
    return matches[0]


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def fmt_s(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.2f} s"


def run_row(name: str, payload: dict, softmax_config: str, sqrt_method: str) -> dict:
    private = payload["private_forward"]
    total_time = float(private["total_time_s"])
    comm_time = float(private.get("total_comm_time_s", 0.0))
    compute_time = max(total_time - comm_time, 0.0)
    total_bytes = int(private["total_comm_bytes"])
    total_rounds = int(private["total_comm_rounds"])
    softmax_bytes = int(private.get("softmax_comm_bytes", 0))
    softmax_rounds = int(private.get("softmax_comm_rounds", 0))
    layernorm_bytes = int(private.get("layernorm_comm_bytes", 0))
    layernorm_rounds = int(private.get("layernorm_comm_rounds", 0))
    row = {
        "name": name,
        "softmax_config": softmax_config,
        "sqrt_method": sqrt_method,
        "samples_seen": int(payload.get("samples_seen", 0)),
        "wall_time_s": float(payload.get("running_time_s", total_time)),
        "total_time_s": total_time,
        "compute_time_s": compute_time,
        "total_comm_bytes": total_bytes,
        "total_comm_gib": bytes_to_gib(total_bytes),
        "total_comm_rounds": total_rounds,
        "total_comm_time_s": comm_time,
        "softmax_time_s": float(private.get("softmax_time_s", 0.0)),
        "softmax_comm_bytes": softmax_bytes,
        "softmax_comm_gib": bytes_to_gib(softmax_bytes),
        "softmax_comm_rounds": softmax_rounds,
        "layernorm_time_s": float(private.get("layernorm_time_s", 0.0)),
        "layernorm_comm_bytes": layernorm_bytes,
        "layernorm_comm_gib": bytes_to_gib(layernorm_bytes),
        "layernorm_comm_rounds": layernorm_rounds,
        "layernorm_comm_time_s": float(private.get("layernorm_comm_time_s", 0.0)),
        "non_softmax_comm_bytes": total_bytes - softmax_bytes,
        "non_softmax_comm_rounds": total_rounds - softmax_rounds,
        "metric": payload.get("metric", {}),
        "network_estimates": estimate_all(compute_time, total_bytes, total_rounds),
    }
    return row


def add_relative(rows: list[dict]) -> None:
    baseline = next(row for row in rows if row["name"] == "shaft_original")
    for row in rows:
        row["relative_to_shaft_original"] = {
            "comm_ratio": row["total_comm_bytes"] / baseline["total_comm_bytes"],
            "round_ratio": row["total_comm_rounds"] / baseline["total_comm_rounds"],
            "wall_time_ratio": row["wall_time_s"] / baseline["wall_time_s"],
            "comm_reduction_pct": (
                1.0 - row["total_comm_bytes"] / baseline["total_comm_bytes"]
            )
            * 100.0,
            "round_reduction_pct": (
                1.0 - row["total_comm_rounds"] / baseline["total_comm_rounds"]
            )
            * 100.0,
            "network_time_ratio": {},
        }
        for profile in NETWORK_PROFILES:
            current = row["network_estimates"][profile]["estimated_time_s"]
            base = baseline["network_estimates"][profile]["estimated_time_s"]
            row["relative_to_shaft_original"]["network_time_ratio"][profile] = current / base


def parse_softmax_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    current_layer = None
    rows = []
    line_re = re.compile(
        r"^(ode_clip_i16|scaled_k2_i8|scaled_k2_i12|scaled_k2_i16)\s+"
        r"(?P<comp_ms>[0-9.]+)\s+"
        r"(?P<comm_mb>[0-9.]+)\s+"
        r"(?P<rounds>[0-9.]+)\s+"
        r"(?P<lan_ms>[0-9.]+)\s+"
        r"(?P<wan_ms>[0-9.]+)"
    )
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("layer="):
            match = re.search(r"layer=(\d+)", line)
            current_layer = int(match.group(1)) if match else None
            continue
        match = line_re.match(line)
        if not match:
            continue
        comm_bytes = float(match.group("comm_mb")) * 1_000_000
        compute_time_s = float(match.group("comp_ms")) / 1000.0
        rounds = int(float(match.group("rounds")))
        rows.append(
            {
                "layer": current_layer,
                "case": match.group(1),
                "compute_time_s": compute_time_s,
                "comm_bytes": comm_bytes,
                "comm_mb": float(match.group("comm_mb")),
                "rounds": rounds,
                "network_estimates": estimate_all(compute_time_s, int(comm_bytes), rounds),
            }
        )
    return rows


def aggregate_softmax(rows: list[dict]) -> list[dict]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    out = []
    for case, items in grouped.items():
        if not items:
            continue
        scale = 12 / len(items)
        compute_time_s = sum(item["compute_time_s"] for item in items) * scale
        comm_bytes = int(sum(item["comm_bytes"] for item in items) * scale)
        rounds = int(sum(item["rounds"] for item in items) * scale)
        out.append(
            {
                "case": case,
                "measured_layers": [item["layer"] for item in items],
                "aggregation": f"sum_measured_layers_scaled_to_12_layers_by_{scale:.4g}",
                "compute_time_s": compute_time_s,
                "comm_bytes": comm_bytes,
                "comm_gib": bytes_to_gib(comm_bytes),
                "rounds": rounds,
                "network_estimates": estimate_all(compute_time_s, comm_bytes, rounds),
            }
        )
    return out


def make_layernorm_effect(rows: list[dict]) -> list[dict]:
    by_name = {row["name"]: row for row in rows}
    pairs = [
        ("NR_effective", by_name["shaft_original"]),
        ("MLFormer_effective", by_name["layernorm_only"]),
    ]
    out = []
    for case, row in pairs:
        out.append(
            {
                "case": case,
                "source": "e2e_layernorm_counters",
                "time_s": row["layernorm_time_s"],
                "comm_bytes": row["layernorm_comm_bytes"],
                "comm_gib": row["layernorm_comm_gib"],
                "rounds": row["layernorm_comm_rounds"],
                "network_estimates": estimate_all(
                    max(row["layernorm_time_s"] - row["layernorm_comm_time_s"], 0.0),
                    row["layernorm_comm_bytes"],
                    row["layernorm_comm_rounds"],
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "name",
        "softmax_config",
        "sqrt_method",
        "samples_seen",
        "wall_time_s",
        "compute_time_s",
        "total_comm_gib",
        "total_comm_rounds",
        "softmax_comm_gib",
        "softmax_comm_rounds",
        "layernorm_comm_gib",
        "layernorm_comm_rounds",
        "comm_ratio",
        "round_ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "softmax_config": row["softmax_config"],
                    "sqrt_method": row["sqrt_method"],
                    "samples_seen": row["samples_seen"],
                    "wall_time_s": row["wall_time_s"],
                    "compute_time_s": row["compute_time_s"],
                    "total_comm_gib": row["total_comm_gib"],
                    "total_comm_rounds": row["total_comm_rounds"],
                    "softmax_comm_gib": row["softmax_comm_gib"],
                    "softmax_comm_rounds": row["softmax_comm_rounds"],
                    "layernorm_comm_gib": row["layernorm_comm_gib"],
                    "layernorm_comm_rounds": row["layernorm_comm_rounds"],
                    "comm_ratio": row["relative_to_shaft_original"]["comm_ratio"],
                    "round_ratio": row["relative_to_shaft_original"]["round_ratio"],
                }
            )


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def network_headers() -> list[str]:
    return [
        str(profile.get("short_name", name))
        for name, profile in NETWORK_PROFILES.items()
    ]


def write_markdown(
    out_dir: Path, rows: list[dict], softmax_rows: list[dict], softmax_agg: list[dict], max_length: int
) -> None:
    e2e_rows = []
    for row in rows:
        rel = row["relative_to_shaft_original"]
        e2e_rows.append(
            [
                row["name"],
                row["softmax_config"],
                row["sqrt_method"],
                f"{row['wall_time_s']:.2f}",
                f"{row['total_comm_gib']:.2f}",
                f"{row['total_comm_rounds']}",
                f"{row['softmax_comm_gib']:.2f}",
                f"{row['softmax_comm_rounds']}",
                f"{row['layernorm_comm_gib']:.2f}",
                f"{row['layernorm_comm_rounds']}",
                f"{rel['comm_reduction_pct']:.1f}%",
                f"{rel['round_reduction_pct']:.1f}%",
            ]
        )
    network_headers_row = ["config"] + network_headers()
    network_rows = []
    for row in rows:
        network_rows.append(
            [row["name"]]
            + [
                fmt_s(row["network_estimates"][profile]["estimated_time_s"])
                for profile in NETWORK_PROFILES
            ]
        )
    softmax_table_rows = [
        [
            str(row["layer"]),
            row["case"],
            f"{row['comm_mb']:.1f}",
            str(row["rounds"]),
            *[
                fmt_s(row["network_estimates"][profile]["estimated_time_s"])
                for profile in NETWORK_PROFILES
            ],
        ]
        for row in softmax_rows
    ]
    softmax_agg_rows = [
        [
            row["case"],
            f"{row['comm_gib']:.2f}",
            str(row["rounds"]),
            *[
                fmt_s(row["network_estimates"][profile]["estimated_time_s"])
                for profile in NETWORK_PROFILES
            ],
            row["aggregation"],
        ]
        for row in softmax_agg
    ]
    text = [
        f"# Len={max_length} Single-Sample Softmax + LayerNorm Metrics",
        "",
        "## End-to-End",
        "",
        md_table(
            [
                "config",
                "softmax",
                "sqrt",
                "wall s",
                "total GiB",
                "rounds",
                "softmax GiB",
                "softmax rounds",
                "LayerNorm GiB",
                "LayerNorm rounds",
                "comm delta",
                "round delta",
            ],
            e2e_rows,
        ),
        "",
        "## Network Estimated End-to-End Time",
        "",
        md_table(network_headers_row, network_rows),
        "",
        "## Softmax-Only Per-Layer",
        "",
        md_table(
            ["layer", "case", "comm MB", "rounds"] + network_headers(),
            softmax_table_rows,
        ),
        "",
        "## Softmax-Only Aggregate To 12 Layers",
        "",
        md_table(
            ["case", "comm GiB", "rounds"] + network_headers() + ["aggregation"],
            softmax_agg_rows,
        ),
        "",
    ]
    (out_dir / "metrics.md").write_text("\n".join(text))


def write_summary(out_dir: Path, rows: list[dict], max_length: int) -> None:
    by_name = {row["name"]: row for row in rows}
    base = by_name["shaft_original"]
    both = by_name["both_optimized"]
    soft = by_name["softmax_only"]
    ln = by_name["layernorm_only"]
    lines = [
        f"# Len={max_length} Single-Sample Summary",
        "",
        f"- Baseline total communication: `{base['total_comm_gib']:.2f} GiB`, rounds `{base['total_comm_rounds']}`.",
        f"- Softmax-only optimization: `{soft['total_comm_gib']:.2f} GiB`, rounds `{soft['total_comm_rounds']}`.",
        f"- LayerNorm-only optimization: `{ln['total_comm_gib']:.2f} GiB`, rounds `{ln['total_comm_rounds']}`.",
        f"- Both optimizations: `{both['total_comm_gib']:.2f} GiB`, rounds `{both['total_comm_rounds']}`.",
    ]
    if "softmax_i12_only" in by_name:
        i12_soft = by_name["softmax_i12_only"]
        lines.append(
            f"- Softmax k2i12-only: `{i12_soft['total_comm_gib']:.2f} GiB`, rounds `{i12_soft['total_comm_rounds']}`."
        )
    if "i12_both_optimized" in by_name:
        i12_both = by_name["i12_both_optimized"]
        lines.append(
            f"- Softmax k2i12 + LayerNorm optimization: `{i12_both['total_comm_gib']:.2f} GiB`, rounds `{i12_both['total_comm_rounds']}`."
        )
    if "softmax_i16_only" in by_name:
        i16_soft = by_name["softmax_i16_only"]
        lines.append(
            f"- Softmax k2i16-only: `{i16_soft['total_comm_gib']:.2f} GiB`, rounds `{i16_soft['total_comm_rounds']}`."
        )
    if "i16_both_optimized" in by_name:
        i16_both = by_name["i16_both_optimized"]
        lines.append(
            f"- Softmax k2i16 + LayerNorm optimization: `{i16_both['total_comm_gib']:.2f} GiB`, rounds `{i16_both['total_comm_rounds']}`."
        )
    lines.extend(["", "Network estimated time for both optimizations vs baseline:"])
    for profile, settings in NETWORK_PROFILES.items():
        base_t = base["network_estimates"][profile]["estimated_time_s"]
        both_t = both["network_estimates"][profile]["estimated_time_s"]
        delta = base_t - both_t
        pct = (1 - both_t / base_t) * 100 if base_t else math.nan
        short_name = settings.get("short_name", profile)
        lines.append(f"- `{short_name}`: `{fmt_s(base_t)}` -> `{fmt_s(both_t)}` ({delta:.2f}s, {pct:.1f}%).")
    lines.append("")
    lines.append("LayerNorm-only values use SHAFT graph LayerNormalization counters from the end-to-end run.")
    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    out_dir = Path(args.run_dir)
    rows = []
    run_specs = OrderedDict(RUNS)
    for name, spec in OPTIONAL_RUNS.items():
        if (out_dir / "e2e" / name).exists():
            run_specs[name] = spec
    for name, (softmax_config, sqrt_method) in run_specs.items():
        run_dir = out_dir / "e2e" / name
        payload = read_json(find_summary(run_dir, softmax_config))
        rows.append(run_row(name, payload, softmax_config, sqrt_method))
    add_relative(rows)

    softmax_log = out_dir / "operators" / "softmax_len128.log"
    softmax_rows = parse_softmax_log(softmax_log)
    softmax_agg = aggregate_softmax(softmax_rows)

    metrics = {
        "network_profiles": NETWORK_PROFILES,
        "max_length": args.max_length,
        "end_to_end": rows,
        "operators": {
            "softmax_only_per_layer": softmax_rows,
            "softmax_only_aggregate_12_layers": softmax_agg,
            "layernorm_only": make_layernorm_effect(rows),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_csv(out_dir / "metrics.csv", rows)
    write_markdown(out_dir, rows, softmax_rows, softmax_agg, args.max_length)
    write_summary(out_dir, rows, args.max_length)
    manifest = {
        "run_id": out_dir.name,
        "max_length": args.max_length,
        "dataset": (
            "artifacts/experiment/softmax/sst2_scaled_tradeoff_20260524/"
            "half_subset_seed20260524/validation.parquet"
        ),
        "model": os.path.join(os.environ.get("DATA_ROOT", ""), "bert-base-cased-sst2"),
        "gpu_policy": "CUDA_VISIBLE_DEVICES=0 only",
        "runs": list(run_specs.keys()),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
