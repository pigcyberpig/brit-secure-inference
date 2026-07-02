import argparse
import json
import math
from pathlib import Path

import numpy as np

from layernorm_mask_leakage import (
    auc_rank,
    digitize,
    entropy_bits,
    histogram_tv,
    make_quantile_edges,
    quantile_summary,
)


DEFAULT_VARIANCES = (
    "artifacts/experiment/layernorm_mask_leakage_20260527/"
    "main436/layernorm_variances.npy"
)


CONFIGS = [
    {
        "name": "int_z8_S1",
        "kind": "integer",
        "candidates": 255,
        "scale": 1.0,
        "eps": 1e-5,
        "note": "Original integer-Z implementation, Z in [1, 255].",
    },
    {
        "name": "int_z14_S10",
        "kind": "integer_scaled",
        "candidates": 16_383,
        "scale": 10.0,
        "eps": 1e-5,
        "note": "Scaled integer-Z variant from test_scaled_zbits14.py.",
    },
    {
        "name": "frac_5M_S50",
        "kind": "fractional",
        "candidates": 5_000_000,
        "scale": 50.0,
        "eps": 1e-5,
        "denominator": 2**16,
        "note": "Fractional-Z variant from FRACTIONAL_Z_RESULTS.md.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variance_file", default=DEFAULT_VARIANCES)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--x_bins", type=int, default=64)
    parser.add_argument("--w_bins", type=int, default=256)
    parser.add_argument("--sim_samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def sample_log_w(x_values, config, size, rng):
    x = rng.choice(x_values, size=size, replace=True)
    k = rng.integers(1, config["candidates"] + 1, size=size)
    x_eff = x / config["scale"] + config["eps"]
    # The fractional denominator is a public multiplicative constant and does
    # not affect MI or threshold distinguishability, so log(k) is sufficient.
    log_w = np.log(x_eff) + 2.0 * np.log(k)
    return x, log_w


def leakage_metrics(x_values, config, args, rng):
    x_edges = make_quantile_edges(x_values, args.x_bins)
    n_x_bins = len(x_edges) - 1
    x, log_w = sample_log_w(x_values, config, args.sim_samples, rng)
    x_bin = digitize(x, x_edges)

    split = args.sim_samples // 2
    train_w = log_w[:split]
    train_x = x_bin[:split]
    test_w = log_w[split:]
    test_x = x_bin[split:]
    test_x_raw = x[split:]

    w_edges = np.linspace(float(train_w.min()), float(train_w.max()), args.w_bins + 1)
    w_edges[0] = -np.inf
    w_edges[-1] = np.inf
    train_w_bin = digitize(train_w, w_edges)
    test_w_bin = digitize(test_w, w_edges)
    n_w_bins = len(w_edges) - 1

    joint = np.zeros((n_x_bins, n_w_bins), dtype=np.float64)
    np.add.at(joint, (train_x, train_w_bin), 1.0)
    joint += 1e-9
    joint_prob = joint / joint.sum()
    px = joint_prob.sum(axis=1)
    pw = joint_prob.sum(axis=0)
    expected = px[:, None] * pw[None, :]
    mi = float((joint_prob * np.log2(joint_prob / expected)).sum())
    hx = entropy_bits(px)

    likelihood = joint / joint.sum(axis=1, keepdims=True)
    posterior = likelihood[:, test_w_bin] * px[:, None]
    posterior /= posterior.sum(axis=0, keepdims=True)
    pred = posterior.argmax(axis=0)
    top1 = float((pred == test_x).mean())
    top5_idx = np.argpartition(posterior, -min(5, n_x_bins), axis=0)[-min(5, n_x_bins) :, :]
    top5 = float(np.any(top5_idx == test_x[None, :], axis=0).mean())
    posterior_entropy = float(np.mean(-np.sum(posterior * np.log2(np.clip(posterior, 1e-300, 1)), axis=0)))

    centers = []
    for i in range(n_x_bins):
        lo = x_edges[i]
        hi = x_edges[i + 1]
        if np.isneginf(lo):
            lo = np.min(x_values)
        if np.isposinf(hi):
            hi = np.max(x_values)
        centers.append(math.sqrt(max(lo, 1e-300) * max(hi, 1e-300)))
    centers = np.asarray(centers)
    rel_err = np.abs(centers[pred] - test_x_raw) / np.maximum(test_x_raw, 1e-300)

    return {
        "name": config["name"],
        "kind": config["kind"],
        "candidates": config["candidates"],
        "scale": config["scale"],
        "eps": config["eps"],
        "mutual_information_bits": mi,
        "x_entropy_bits": hx,
        "normalized_mi": mi / hx if hx > 0 else float("nan"),
        "posterior_entropy_bits": posterior_entropy,
        "top1_xbin_accuracy": top1,
        "top5_xbin_accuracy": top5,
        "map_relative_error_median": float(np.median(rel_err)),
        "map_relative_error_p90": float(np.quantile(rel_err, 0.9)),
        "note": config["note"],
    }


def pairwise_metrics(x_values, config, rng, pair_count=80000):
    q_pairs = [(0.01, 0.05), (0.25, 0.5), (0.5, 0.75), (0.95, 0.99)]
    out = []
    for qa, qb in q_pairs:
        xa = float(np.quantile(x_values, qa))
        xb = float(np.quantile(x_values, qb))
        k0 = rng.integers(1, config["candidates"] + 1, size=pair_count)
        k1 = rng.integers(1, config["candidates"] + 1, size=pair_count)
        w0 = np.log(xa / config["scale"] + config["eps"]) + 2.0 * np.log(k0)
        w1 = np.log(xb / config["scale"] + config["eps"]) + 2.0 * np.log(k1)
        labels = np.concatenate([np.zeros(pair_count), np.ones(pair_count)])
        scores = np.concatenate([w0, w1])
        out.append(
            {
                "q_low": qa,
                "q_high": qb,
                "x_low": xa,
                "x_high": xb,
                "auc_logw_threshold": auc_rank(labels, scores),
                "tv_logw_hist": histogram_tv(w0, w1, bins=512),
            }
        )
    return out


def write_markdown(path, report):
    lines = [
        "# LayerNorm Mask Config Leakage",
        "",
        "## Variance Prior",
        "",
        f"- Count: `{report['variance_count']}`",
        f"- p1 / median / p99: `{report['variance_summary']['p1']:.6g}` / "
        f"`{report['variance_summary']['p50']:.6g}` / `{report['variance_summary']['p99']:.6g}`",
        "",
        "## Config Comparison",
        "",
        "| config | candidates | scale | MI bits | normalized MI | top1 x-bin | top5 x-bin | median rel err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["leakage"]:
        lines.append(
            f"| {row['name']} | {row['candidates']} | {row['scale']:.0f} | "
            f"{row['mutual_information_bits']:.4f} | {row['normalized_mi']:.4f} | "
            f"{row['top1_xbin_accuracy']:.4f} | {row['top5_xbin_accuracy']:.4f} | "
            f"{row['map_relative_error_median']:.4f} |"
        )
    lines.extend(["", "## Pairwise Distinguishability", ""])
    for item in report["pairwise"]:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append("| quantiles | x_low | x_high | AUC(log W) | TV(hist) |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in item["pairs"]:
            lines.append(
                f"| p{row['q_low'] * 100:g} vs p{row['q_high'] * 100:g} | "
                f"{row['x_low']:.6g} | {row['x_high']:.6g} | "
                f"{row['auc_logw_threshold']:.4f} | {row['tv_logw_hist']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_values = np.load(args.variance_file).astype(np.float64)
    x_values = x_values[np.isfinite(x_values) & (x_values > 0)]

    report = {
        "variance_file": args.variance_file,
        "variance_count": int(x_values.size),
        "variance_summary": quantile_summary(x_values),
        "x_bins": args.x_bins,
        "w_bins": args.w_bins,
        "sim_samples": args.sim_samples,
        "configs": CONFIGS,
        "leakage": [leakage_metrics(x_values, cfg, args, rng) for cfg in CONFIGS],
        "pairwise": [
            {"name": cfg["name"], "pairs": pairwise_metrics(x_values, cfg, rng)}
            for cfg in CONFIGS
        ],
    }
    (output_dir / "config_leakage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "config_leakage.md", report)
    print(json.dumps({"output_dir": str(output_dir), "configs": [c["name"] for c in CONFIGS]}, indent=2))


if __name__ == "__main__":
    main()
