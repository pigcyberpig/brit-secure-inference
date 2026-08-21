#!/usr/bin/env python3
"""Summarize LayerNorm privacy-leakage statistics for paper revision.

Reads authoritative quest 003 metrics and writes a new dated analysis bundle
without touching existing artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats


DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "LEAKAGE_PROBE_ROOT",
        "artifacts/experiment/layernorm_mpcguard_leakage_probe",
    )
)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n == 1:
        return mean, mean
    if n == 2:
        # t critical df=1.
        tcrit = 12.706204736432095
    elif n == 3:
        tcrit = 4.302652729911275
    else:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
    se = float(arr.std(ddof=1) / math.sqrt(n))
    half = tcrit * se
    return mean - half, mean + half


def perm_pvalue(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    obs = float(arr.mean())
    if n == 1:
        return 1.0
    flips = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=float)
    sims = (flips * arr).mean(axis=1)
    return float(np.mean(np.abs(sims) >= abs(obs) - 1e-12))


def fmt_mean_ci(values: list[float]) -> str:
    lo, hi = ci95(values)
    return f"{np.mean(values):.4f} [{lo:.4f}, {hi:.4f}]"


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing protocol_probe_ring64/ and token_recovery/ metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/layernorm_leakage_stats_20260819",
    )
    args = parser.parse_args()

    source_root = args.source_root
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    protocol_metrics = load_metrics(
        source_root / "protocol_probe_ring64" / "metrics.json"
    )
    token_metrics = load_metrics(source_root / "token_recovery" / "metrics.json")

    protocol_rows = protocol_metrics["rows"]
    token_rows = token_metrics["sweep_rows"]
    cryp_rows = token_metrics["crypten_results"]

    protocol_focus = [
        r
        for r in protocol_rows
        if r["protocol"] == "protocol_z2"
        and r["z_max"] == 255
        and r["pair_type"] in {"quantile", "scale", "shift", "zero"}
    ]
    protocol_baseline = [
        r
        for r in protocol_rows
        if r["protocol"] in {"additive", "full_ring_mul"}
        and r["z_max"] == 0
        and r["pair_type"] in {"quantile", "scale", "shift", "zero"}
    ]

    token_main = [
        r
        for r in token_rows
        if r["protocol"] == "protocol_z2"
        and r["z_max"] == 255
        and r["clf"] == "ridge"
        and r["n_top_tokens"] in {5, 10, 20, 50}
    ]
    token_all_clfs = [
        r
        for r in token_rows
        if r["protocol"] == "protocol_z2"
        and r["z_max"] == 255
        and r["n_top_tokens"] in {5, 10, 20, 50}
    ]
    token_controls = [
        r
        for r in token_rows
        if r["protocol"] in {"oracle", "additive", "protocol_z2_party0"}
        and r["clf"] == "ridge"
        and r["n_top_tokens"] in {5, 10, 20, 50}
    ]

    protocol_summary = {}
    for pair in ("quantile", "scale", "shift", "zero"):
        rows = [r for r in protocol_focus if r["pair_type"] == pair]
        protocol_summary[pair] = {
            "n": len(rows),
            "acc_real_mean_ci": fmt_mean_ci([r["acc_real"] for r in rows]),
            "acc_ideal_mean_ci": fmt_mean_ci([r["acc_ideal"] for r in rows]),
            "gap_mean_ci": fmt_mean_ci([r["gap"] for r in rows]),
            "p_perm": perm_pvalue([r["gap"] for r in rows]),
        }

    baseline_summary = {}
    for prot in ("additive", "full_ring_mul"):
        rows = [r for r in protocol_baseline if r["protocol"] == prot]
        baseline_summary[prot] = {
            pair: fmt_mean_ci([r["gap"] for r in rows if r["pair_type"] == pair])
            for pair in ("quantile", "scale", "shift", "zero")
        }

    token_summary = {}
    for k in (5, 10, 20, 50):
        rows = [r for r in token_main if r["n_top_tokens"] == k]
        token_summary[k] = {
            "n": len(rows),
            "n_classes": int(rows[0]["n_classes"]) if rows else None,
            "freq_prior": float(rows[0]["freq_prior"]) if rows else None,
            "acc_real_mean_ci": fmt_mean_ci([r["acc_real"] for r in rows]),
            "acc_ideal_mean_ci": fmt_mean_ci([r["acc_ideal"] for r in rows]),
            "gap_mean_ci": fmt_mean_ci([r["gap"] for r in rows]),
            "p_perm": perm_pvalue([r["gap"] for r in rows]),
        }

    token_all_summary = {}
    for k in (5, 10, 20, 50):
        rows = [r for r in token_all_clfs if r["n_top_tokens"] == k]
        token_all_summary[k] = {
            "n": len(rows),
            "acc_real_mean_ci": fmt_mean_ci([r["acc_real"] for r in rows]),
            "acc_ideal_mean_ci": fmt_mean_ci([r["acc_ideal"] for r in rows]),
            "gap_mean_ci": fmt_mean_ci([r["gap"] for r in rows]),
            "p_perm": perm_pvalue([r["gap"] for r in rows]),
        }

    control_summary = {}
    for prot in ("oracle", "additive", "protocol_z2_party0"):
        rows = [r for r in token_controls if r["protocol"] == prot]
        control_summary[prot] = {
            k: {
                "acc_real_mean_ci": fmt_mean_ci([r["acc_real"] for r in rows if r["n_top_tokens"] == k]),
                "acc_ideal_mean_ci": fmt_mean_ci([r["acc_ideal"] for r in rows if r["n_top_tokens"] == k]),
                "gap_mean_ci": fmt_mean_ci([r["gap"] for r in rows if r["n_top_tokens"] == k]),
                "p_perm": perm_pvalue([r["gap"] for r in rows if r["n_top_tokens"] == k]),
            }
            for k in (5, 10, 20, 50)
            if any(r["n_top_tokens"] == k for r in rows)
        }

    summary = {
        "source": {
            "protocol_metrics": str(source_root / "protocol_probe_ring64" / "metrics.json"),
            "token_metrics": str(source_root / "token_recovery" / "metrics.json"),
        },
        "protocol_probe": protocol_summary,
        "protocol_baselines": baseline_summary,
        "token_recovery": token_summary,
        "token_recovery_all_clfs": token_all_summary,
        "token_controls": control_summary,
        "crypten_validation": {
            "n_rows": len(cryp_rows),
            "note": "Only one preserved seed is available in the stored artifact; CI/p-value should not be claimed from this file alone.",
            "rows": cryp_rows,
        },
        "paper_notes": {
            "threshold_protocol": 0.10,
            "threshold_token": 0.05,
            "threshold_note": "Operational thresholds only, not significance levels.",
            "token_candidate_rule": "Top-K frequent non-special tokens via Counter.most_common(K), then natural-frequency subsampling.",
            "class_balance_note": "Protocol probe labels are i.i.d. Bernoulli(0.5); token recovery uses the natural class imbalance of the selected top-K tokens and reports frequency prior.",
        },
    }

    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LayerNorm Privacy Leakage Stats",
        "",
        "## Protocol Probe",
        f"- Model: `bert-base-cased-sst2`",
        f"- Samples: `128` sentences, `3344` valid tokens, `83600` variance values",
        f"- Seeds: `3` (`20260609`, `20261618`, `20262627`)",
        f"- Attackers: Ridge (`ridge=1e-2`) and MLP (`hidden=(32,16)`, `epochs=80`, `lr=0.01`, `batch_size=64`)",
        "",
        "| pair | n | acc_real | acc_ideal | gap | exact perm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in ("quantile", "scale", "shift", "zero"):
        s = protocol_summary[pair]
        lines.append(
            f"| {pair} | {s['n']} | {s['acc_real_mean_ci']} | {s['acc_ideal_mean_ci']} | {s['gap_mean_ci']} | {s['p_perm']:.4f} |"
        )
    lines += [
        "",
        "## Token Recovery",
        f"- Model: `bert-base-cased-sst2`",
        f"- Samples: `256` sentences, `156350` `(token, variance)` pairs, `1984` unique tokens",
        f"- Seeds: `2` (`20260610`, `20261619`)",
        f"- Main attacker: Ridge (`ridge=1e-2`); MLP check: `hidden=(64,32)`, `epochs=80`, `lr=0.01`, `batch_size=64`",
        "",
        "| K | n | acc_real | acc_ideal | gap | exact perm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in (5, 10, 20, 50):
        s = token_summary[k]
        lines.append(
            f"| {k} | {s['n']} | {s['acc_real_mean_ci']} | {s['acc_ideal_mean_ci']} | {s['gap_mean_ci']} | {s['p_perm']:.4f} |"
        )
    lines += [
        "",
        "### Ridge + MLP robustness check",
        "| K | n | acc_real | acc_ideal | gap | exact perm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in (5, 10, 20, 50):
        s = token_all_summary[k]
        lines.append(
            f"| {k} | {s['n']} | {s['acc_real_mean_ci']} | {s['acc_ideal_mean_ci']} | {s['gap_mean_ci']} | {s['p_perm']:.4f} |"
        )
    lines += [
        "",
        "## CrypTen Validation",
        f"- Stored rows: {len(cryp_rows)}",
        "- Only a single preserved seed is available in the stored artifact; do not quote CI/p-values from it.",
        "",
        "## Reporting Notes",
        "- Thresholds: `0.10` (protocol probe) and `0.05` (token recovery) are operational rules, not significance levels.",
        "- Candidate tokens: top-K frequent non-special tokens via `Counter.most_common(K)`.",
        "- Class balance: protocol probe labels are balanced by Bernoulli sampling; token recovery is naturally imbalanced and should be contextualized with the frequency prior.",
    ]
    write_md(out / "stats.md", lines)


if __name__ == "__main__":
    main()
